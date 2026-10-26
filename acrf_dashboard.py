"""
LogiData ACRF Dashboard - analysis engine.

All framework calculations are delegated to the published functions in
acrf_framework.py (confidence_discount, safety_stock_adjusted, rolling_mape,
adjusted_confidence_with_mape, supplier_concentration_hhi,
composite_capacity_risk_score). This module only prepares order data,
calls those functions per category, and returns JSON for the page.
"""

import io
import json
import math

import numpy as np
import pandas as pd

import acrf_framework as acrf

# Column roles the dashboard understands, with the DataCo names as defaults.
ROLES = {
    "date": ["order date (DateOrders)", "order_date", "date", "order date"],
    "category": ["Category Name", "category", "product category", "sku", "product"],
    "quantity": ["Order Item Quantity", "quantity", "qty", "units"],
    "price": ["Product Price", "unit_price", "price", "unit cost", "unit_cost"],
    "sales": ["Sales", "sales", "revenue", "amount"],
    "lead_real": ["Days for shipping (real)", "lead_time", "lead time", "days to ship", "shipping days"],
    "lead_sched": ["Days for shipment (scheduled)", "scheduled_lead_time", "scheduled days"],
    "late_flag": ["Late_delivery_risk", "late", "late_flag", "is_late"],
    "carrier": ["Shipping Mode", "carrier", "supplier", "shipping_mode", "vendor"],
    "order_id": ["Order Id", "order_id", "order id", "order number"],
}

_STATE = {"df": None, "source": None}


def _guess(columns):
    lower = {c.lower().strip(): c for c in columns}
    out = {}
    for role, names in ROLES.items():
        out[role] = next((lower[n.lower()] for n in names if n.lower() in lower), None)
    return out


def _parse_dates(s):
    d = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(s, errors="coerce", format="mixed")
    return d


def load(csv_text: str, source: str) -> str:
    """Load a CSV and return its columns plus a suggested column mapping."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [str(c) for c in df.columns]
    _STATE["df"] = df
    _STATE["source"] = source
    return json.dumps({"columns": list(df.columns), "rows": int(len(df)), "guess": _guess(df.columns)})


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _forecast_ma(q, window):
    """Simple moving-average forecast: each day forecast = mean of the prior `window` days."""
    return pd.Series(q).shift(1).rolling(window).mean().to_numpy()


def analyze(params_json: str) -> str:
    p = json.loads(params_json)
    m = p["mapping"]
    df = _STATE["df"]
    if df is None:
        raise ValueError("No data loaded.")
    for role in ("date", "category", "quantity"):
        if not m.get(role):
            raise ValueError(f"Please map the '{role}' column.")

    z = float(p.get("z", 1.65))
    top_n = int(p.get("top_n", 5))
    bench = int(p.get("benchmark", 730))
    ma_window = int(p.get("ma_window", 5))
    mape_window = int(p.get("mape_window", 30))
    use_feedback = bool(p.get("feedback", False))
    n_override = p.get("n_override")
    weights = p.get("weights", [0.40, 0.35, 0.25])
    fallback_lead = float(p.get("fallback_lead", 7))
    fallback_lead_sd = float(p.get("fallback_lead_sd", 0))

    d = pd.DataFrame({
        "date": _parse_dates(df[m["date"]]),
        "category": df[m["category"]].astype(str).str.strip(),
        "qty": _num(df[m["quantity"]]),
    })
    if m.get("price"):
        d["price"] = _num(df[m["price"]])
    if m.get("sales"):
        d["sales"] = _num(df[m["sales"]])
    if m.get("lead_real"):
        d["lead"] = _num(df[m["lead_real"]])
    if m.get("lead_sched"):
        d["lead_sched"] = _num(df[m["lead_sched"]])
    if m.get("late_flag"):
        d["late"] = _num(df[m["late_flag"]])
    elif m.get("lead_real") and m.get("lead_sched"):
        d["late"] = (d["lead"] > d["lead_sched"]).astype(float)
    if m.get("carrier"):
        d["carrier"] = df[m["carrier"]].astype(str).str.strip()

    d = d.dropna(subset=["date", "qty"])
    d = d[d["qty"] > 0]
    if d.empty:
        raise ValueError("No usable rows after reading dates and quantities.")
    d["day"] = d["date"].dt.normalize()
    days = pd.date_range(d["day"].min(), d["day"].max(), freq="D")
    n_days = len(days)
    n_used = int(n_override) if n_override else n_days

    ranking = d["category"].value_counts()
    all_cats = list(ranking.index)
    cats = all_cats if top_n <= 0 else all_cats[:top_n]
    sub = d[d["category"].isin(cats)]

    results = []
    for cat in cats:
        g = sub[sub["category"] == cat]
        q = g.groupby("day")["qty"].sum().reindex(days, fill_value=0).to_numpy(dtype=float)
        D, sdD = float(q.mean()), float(q.std(ddof=1)) if len(q) > 1 else 0.0

        if "lead" in g and g["lead"].notna().any():
            L, sdL = float(g["lead"].mean()), float(g["lead"].std(ddof=1)) if g["lead"].notna().sum() > 1 else 0.0
        else:
            L, sdL = fallback_lead, fallback_lead_sd

        if "price" in g and g["price"].notna().any():
            unit_value = float(g["price"].mean())
        elif "sales" in g and g["sales"].notna().any():
            unit_value = float(g["sales"].sum() / g["qty"].sum())
        else:
            unit_value = None

        # Component 1: data-sparsity-adjusted safety stock (published function)
        ss = acrf.safety_stock_adjusted(D, sdD, L, sdL, service_level_z=z,
                                        n_observations=n_used, enterprise_benchmark=bench)
        c_n = float(ss["confidence_discount"])
        var_component = math.sqrt(L * sdD ** 2 + D ** 2 * sdL ** 2)

        # Component 2: iterative forecast-error correction (published function)
        fc = _forecast_ma(q, ma_window)
        mask = ~np.isnan(fc)
        roll = acrf.rolling_mape(q[mask], fc[mask], window=mape_window) if mask.sum() > mape_window else []
        if roll:
            mape = float(roll[-1]["rolling_mape"])
            mape_factor = float(roll[-1]["mape_adjustment_factor"])
        else:
            mape, mape_factor = None, None
        c_adj = float(acrf.adjusted_confidence_with_mape(c_n, mape_factor)) if mape_factor else None
        ss_feedback = round(z * c_adj * var_component, 1) if c_adj is not None else None

        ss_classical = float(ss["ss_classical"])
        ss_acrf = ss_feedback if (use_feedback and ss_feedback is not None) else float(ss["ss_adjusted"])
        units_freed = round(ss_classical - ss_acrf, 1)
        capital_freed = round(units_freed * unit_value, 2) if unit_value is not None else None

        # Component 3: Composite Capacity-Risk Score (published functions)
        util = float(100 * (1 - g["late"].mean())) if "late" in g and g["late"].notna().any() else None
        reliability = max(0.0, 100 - mape) if mape is not None else None
        if "carrier" in g and g["carrier"].notna().any():
            dis, hhi = acrf.supplier_concentration_hhi(g["carrier"].value_counts(normalize=True))
            dis, hhi = float(dis), float(hhi)
        else:
            dis, hhi = None, None
        comps = [util, reliability, dis]
        if all(v is not None for v in comps):
            ccrs = acrf.composite_capacity_risk_score(util, reliability, dis, *weights)
            ccrs = {k: (v if isinstance(v, str) else float(v)) for k, v in ccrs.items()}
            ccrs["partial"] = False
        else:
            avail = [(v, w) for v, w in zip(comps, weights) if v is not None]
            if avail:
                tw = sum(w for _, w in avail)
                score = round(sum(v * w / tw for v, w in avail), 1)
                cat_label = "Low Risk" if score >= 70 else "Moderate Risk" if score >= 50 else "High Risk"
                ccrs = {"ccrs_score": score, "risk_category": cat_label, "partial": True,
                        "utilization_component": None, "forecast_component": None, "disruption_component": None}
            else:
                ccrs = None

        mape_series = [[int(r["period"]), float(r["rolling_mape"])] for r in roll]
        results.append({
            "category": cat,
            "records": int(len(g)),
            "units": float(g["qty"].sum()),
            "avg_daily_demand": round(D, 2), "std_daily_demand": round(sdD, 2),
            "lead_time": round(L, 2), "lead_time_sd": round(sdL, 2),
            "unit_value": round(unit_value, 2) if unit_value is not None else None,
            "c_n": c_n, "mape": mape, "mape_factor": mape_factor, "c_adj": c_adj,
            "ss_classical": ss_classical, "ss_adjusted": float(ss["ss_adjusted"]),
            "ss_feedback": ss_feedback, "ss_acrf": ss_acrf,
            "units_freed": units_freed,
            "reduction_pct": round(100 * units_freed / ss_classical, 1) if ss_classical else 0.0,
            "capital_freed": capital_freed,
            "on_time_pct": round(util, 1) if util is not None else None,
            "hhi": hhi, "disruption_score": dis,
            "forecast_reliability": round(reliability, 1) if reliability is not None else None,
            "ccrs": ccrs,
            "series": {
                "days": [x.strftime("%Y-%m-%d") for x in days],
                "actual": [float(v) for v in q],
                "forecast": [None if np.isnan(v) else round(float(v), 2) for v in fc],
                "mape_offset": int(np.argmax(mask)) if mask.any() else 0,
                "rolling_mape": mape_series,
            },
        })

    cap_vals = [r["capital_freed"] for r in results if r["capital_freed"] is not None]
    val = lambda key: sum(r[key] * r["unit_value"] for r in results if r["unit_value"] is not None)
    scores = [r["ccrs"]["ccrs_score"] for r in results if r["ccrs"]]
    ot = [r["on_time_pct"] for r in results if r["on_time_pct"] is not None]
    portfolio = {
        "categories": len(results),
        "categories_available": len(all_cats),
        "records": int(len(sub)),
        "records_total": int(len(d)),
        "orders": int(df.loc[sub.index, m["order_id"]].nunique()) if m.get("order_id") and m["order_id"] in df else None,
        "start": days[0].strftime("%b %d, %Y"), "end": days[-1].strftime("%b %d, %Y"),
        "n_days": n_days, "n_used": n_used,
        "c_n": float(acrf.confidence_discount(n_used, bench)),
        "capital_freed": round(sum(cap_vals), 2) if cap_vals else None,
        "ss_value_classical": round(val("ss_classical"), 2) if cap_vals else None,
        "ss_value_acrf": round(val("ss_acrf"), 2) if cap_vals else None,
        "units_classical": round(sum(r["ss_classical"] for r in results), 1),
        "units_acrf": round(sum(r["ss_acrf"] for r in results), 1),
        "ccrs_min": min(scores) if scores else None, "ccrs_max": max(scores) if scores else None,
        "ccrs_avg": round(sum(scores) / len(scores), 1) if scores else None,
        "on_time_avg": round(sum(ot) / len(ot), 1) if ot else None,
        "has": {"lead": "lead" in d, "value": any(r["unit_value"] is not None for r in results),
                "late": "late" in d, "carrier": "carrier" in d},
    }
    curve = [[k, float(acrf.confidence_discount(k, bench))] for k in
             [1, 7, 14, 30, 45, 60, 90, 120, 150, 180, 240, 300, 365, 450, 540, 630, 730, 850]]
    return json.dumps({"portfolio": portfolio, "categories": results, "curve": curve, "source": _STATE["source"]})

"""Streamlit page — drop into the TYCCLES dashboard as a new tab, or run alone:
    streamlit run app/inflation_page.py
Reads only files written by `python -m infcomp build/validate`; no computation here.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path("outputs")


def render():
    st.header("US Inflation Composite")
    hook_p = OUT / "inflation_regime_hook.json"
    if not hook_p.exists():
        st.warning("No build yet. Run `python -m infcomp verify`, confirm rows, then "
                   "`python -m infcomp build`.")
        return
    hook = json.loads(hook_p.read_text())
    status = hook["status"]
    (st.success if status == "SIGNAL" else st.info if status == "DESCRIPTIVE" else st.warning)(
        f"Status: **{status}**  ·  method: {hook['production_method']}  ·  as of {hook['as_of']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite (σ)", f"{hook['level_z']:+.2f}",
              None if hook["chg_3m"] is None else f"{hook['chg_3m']:+.2f} 3m")
    c2.metric("Trajectory", hook["trajectory"] or "n/a")
    c3.metric("Heating breadth", f"{hook['diffusion_heating_pct']:.0f}%")
    c4.metric("Top-2 share of |contrib|", f"{hook['concentration']['top_share']:.0%}",
              "carried by few" if hook["concentration"]["carried_by_few"] else None,
              delta_color="inverse")

    st.image(str(OUT / "inflation_dashboard.png"), use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["Ranked contributions", "History", "Redundancy diagnostic",
                              "Validation / series map"])
    with t1:
        st.dataframe(pd.read_csv(OUT / "ranked_contributions.csv"), use_container_width=True)
    with t2:
        h = pd.read_csv(OUT / "composite_history.csv", index_col=0, parse_dates=True)
        st.line_chart(h[["composite_z"]]); st.line_chart(h[["ltm_model", "ltm_final"]])
    with t3:
        if (OUT / "diag_summary.json").exists():
            st.json(json.loads((OUT / "diag_summary.json").read_text()))
            st.dataframe(pd.read_csv(OUT / "diag_near_duplicates.csv"))
    with t4:
        v = OUT / "validation_result.json"
        st.json(json.loads(v.read_text())) if v.exists() else st.write("Not yet validated.")
        m = Path("config/series_map_verified.csv")
        if m.exists():
            st.dataframe(pd.read_csv(m), use_container_width=True)


if __name__ == "__main__" or st.runtime.exists():
    render()

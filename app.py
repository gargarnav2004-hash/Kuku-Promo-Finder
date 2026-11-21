"""AI Promo Finder - Streamlit app.  Launch with:  streamlit run app.py"""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
FULL_RUN = ROOT / "outputs"
APP_RUN = ROOT / "runs" / "app_latest"
REVIEWS = ROOT / "reviews.csv"
REASONS = ["weak hook", "weak cliffhanger", "needs context", "bad timestamps", "unsupported", "duplicate"]

st.set_page_config(page_title="AI Promo Finder", layout="wide")


def load_run(d: Path):
    if not (d / "promos.json").exists():
        return None, None
    return json.loads((d / "promos.json").read_text()), json.loads((d / "run_summary.json").read_text())


def load_reviews() -> pd.DataFrame:
    if REVIEWS.exists():
        return pd.read_csv(REVIEWS, dtype=str).fillna("")
    return pd.DataFrame(columns=["run", "promo_id", "decision", "reason", "note", "ai_score", "reviewed_at"])


def save_review(run, pid, decision, reason, note, score):
    df = load_reviews()
    df = df[~((df["run"] == run) & (df["promo_id"] == pid))]
    row = {"run": run, "promo_id": pid, "decision": decision, "reason": reason, "note": note,
           "ai_score": str(score), "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    pd.concat([df, pd.DataFrame([row])]).to_csv(REVIEWS, index=False)


def promo_card(p, expanded=False):
    j = p.get("judge") or {}
    icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[p["status"]]
    title = f"#{p['rank']}  |  {p['score']}/100  |  {p['duration']}s  |  Episodes {', '.join(map(str, p['episodes']))}  |  {icon}"
    gt = p.get("gate")
    if gt:
        pf = lambda t: "PASS" if gt[t]["pass"] else "FAIL"
        title += f"  |  Hook: {pf('hook')}  |  Cliffhanger: {pf('cliffhanger')}"
        if p.get("near_miss"):
            title = f"NEAR MISS (gap {p.get('gap')})  |  " + title
    with st.expander(title, expanded=expanded):
        if p.get("near_miss"):
            st.warning(f"NEAR MISS: added only to reach the requested number of promos. It did NOT pass every criterion "
                       f"(gap to passing: {p.get('gap')}). Review carefully.")
        if gt:
            c1, c2 = st.columns(2)
            for col, t, label in ((c1, "hook", "Hook test"), (c2, "cliffhanger", "Cliffhanger test")):
                if gt[t]["pass"]:
                    col.success(f"{label}: PASS")
                else:
                    col.error(f"{label}: FAIL - " + "; ".join(gt[t]["reasons"]))
            nc = p.get("next_check") or {}
            st.caption(f"Ending type: {j.get('unresolved_type', '-')}.  Next line in script: "
                       f"{' || '.join(p.get('next_lines', [])) or '-'}  ->  {nc.get('reason', 'not checked')}")
        st.markdown("**Segments (in order)**")
        for k, s in enumerate(p["segments"], 1):
            st.markdown(f"{k}. Episode **{s['episode']}**  `{s['start_ts']}` -> `{s['end_ts']}`  ({s['end'] - s['start']:.1f}s)")
        st.markdown(f"**Hook:** {j.get('hook_text', '')}")
        st.markdown(f"**Cliffhanger:** {j.get('cliffhanger_text', '')}")
        st.markdown(f"**Why it works:** {j.get('why_it_works', '')}")
        st.markdown(f"**Risks:** {j.get('risks', '')}")
        st.caption("AI scores (1-5): " + ", ".join(f"{k}={j.get(k)}" for k in
                   ["hook", "cliffhanger", "standalone_clarity", "timestamp_quality", "promo_appeal"]))
        for v in p["validation"]:
            (st.error if v["level"] == "error" else st.warning)(f"{v['code']}: {v['msg']}")
        with st.popover("Source lines (screenplay evidence)"):
            st.code("\n".join(p["source_lines"]), language=None)


# ---------------------------------------------------------------- sidebar
page = st.sidebar.radio("Page", ["1. Promo Finder", "2. Human Review", "3. Quality"])
runs = {"Full run (outputs/)": FULL_RUN, "Last app run (runs/app_latest/)": APP_RUN}
available = [k for k, v in runs.items() if (v / "promos.json").exists()]
if not available:
    available = ["Full run (outputs/)"]
if st.session_state.pop("switch_to_app_run", False):
    st.session_state["run_name"] = "Last app run (runs/app_latest/)"
run_name = st.sidebar.selectbox("Results to show", available, key="run_name")
promos, summary = load_run(runs[run_name])
st.sidebar.caption("Judged from screenplay text only - not acting, visuals or sound.")

# ---------------------------------------------------------------- page 1
if page.startswith("1"):
    st.title("Promo Finder")
    shows = sorted([d for d in (ROOT / "data").iterdir() if d.is_dir() and list(d.glob("*.txt"))]) if (ROOT / "data").exists() else []
    if shows:
        from parse import load_show
        show = st.selectbox("Show (folder in data/)", shows, format_func=lambda d: d.name)
        eps = sorted(load_show(show).keys())
        sel = st.multiselect("Episodes to analyze", eps, default=eps[:3])
        st.caption("Cached episodes are instant. New episodes take ~5-10s each on the free Gemini tier.")
        if st.button("Analyze", type="primary", disabled=not sel):
            from pipeline import run_pipeline
            bar = st.progress(0.0, "Starting...")
            try:
                run_pipeline(sel, APP_RUN, show, progress=lambda f, m: bar.progress(min(1.0, f), m))
                st.session_state["switch_to_app_run"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        st.info("Put a show folder of screenplay .txt files inside data/ to analyze it.")

    st.divider()
    if promos is None:
        st.info("No results yet. Run an analysis above or `python run.py`.")
    else:
        s = summary
        st.subheader(f"Ranked promos - {s['show']} ({s['episodes_analyzed']} episodes)")
        c = st.columns(4)
        c[0].metric("Final promos", s["promos_ranked"])
        c[1].metric("Candidates built", s["candidates_built"])
        c[2].metric("Target length", f"{s.get('target_seconds', '-')}s")
        c[3].metric("Episodes covered", len(s.get("episodes_covered", [])))
        st.caption(f"Allowed length {s.get('duration_range', ['?', '?'])[0]}-{s.get('duration_range', ['?', '?'])[1]}s "
                   f"(cuts land on whole lines). Duplicates removed: {len(s['near_duplicates_removed'])}. "
                   f"Other judged promos not in the final list are in all_ranked_promos.json.")
        hide_fail = st.checkbox("Hide promos that fail rule checks", value=False)
        for i, p in enumerate(promos):
            if hide_fail and p["status"] == "fail":
                continue
            promo_card(p, expanded=i == 0)

# ---------------------------------------------------------------- page 2
elif page.startswith("2"):
    st.title("Human Review")
    if not promos:
        st.info("No promos to review yet.")
    else:
        rv = load_reviews()
        rv = rv[rv["run"] == run_name]
        done = dict(zip(rv["promo_id"], rv["decision"]))
        st.caption(f"{len(done)} of {len(promos)} reviewed. Saved to reviews.csv (survives restarts).")
        only_open = st.checkbox("Show only unreviewed", value=False)
        for p in promos:
            if only_open and p["id"] in done:
                continue
            promo_card(p)
            prev = rv[rv["promo_id"] == p["id"]]
            prev = prev.iloc[0].to_dict() if len(prev) else {}
            with st.form(f"f_{p['id']}"):
                cols = st.columns([2, 2, 3, 1])
                dec = cols[0].radio("Decision", ["Approve", "Reject"], horizontal=True,
                                    index=1 if prev.get("decision") == "Reject" else 0)
                reason = cols[1].selectbox("Reject reason", [""] + REASONS,
                                           index=([""] + REASONS).index(prev.get("reason", "")) if prev.get("reason", "") in REASONS else 0)
                note = cols[2].text_input("Note", value=prev.get("note", ""))
                if cols[3].form_submit_button("Save"):
                    if dec == "Reject" and not reason:
                        st.error("Pick a reject reason.")
                    else:
                        save_review(run_name, p["id"], dec, reason if dec == "Reject" else "", note, p["score"])
                        st.success(f"Saved: {dec}")
            if p["id"] in done:
                st.caption(f"Current decision: **{done[p['id']]}**")
            st.write("")

# ---------------------------------------------------------------- page 3
else:
    st.title("Quality")
    if not promos:
        st.info("No results yet.")
    else:
        rv = load_reviews()
        rv = rv[rv["run"] == run_name]
        appr, rej = rv[rv["decision"] == "Approve"], rv[rv["decision"] == "Reject"]
        c = st.columns(4)
        c[0].metric("Generated (ranked)", len(promos))
        c[1].metric("Reviewed", len(rv))
        c[2].metric("Approved", len(appr))
        c[3].metric("Rejected", len(rej))
        score = lambda d: pd.to_numeric(d["ai_score"], errors="coerce").mean()
        c = st.columns(2)
        c[0].metric("Avg AI score - approved", f"{score(appr):.1f}" if len(appr) else "-")
        c[1].metric("Avg AI score - rejected", f"{score(rej):.1f}" if len(rej) else "-")
        st.subheader("Top rejection reasons")
        if len(rej):
            st.bar_chart(rej["reason"].value_counts())
        else:
            st.caption("No rejections yet.")
        gs = summary.get("gate")
        if gs:
            st.subheader("Hook + cliffhanger gate")
            c = st.columns(4)
            c[0].metric("Judged candidates", summary["candidates_judged"])
            c[1].metric("Passed both tests", gs["passed_both"])
            c[2].metric("Failed hook test", gs["failed_hook_test"])
            c[3].metric("Failed cliffhanger test", gs["failed_cliffhanger_test"])
            st.caption(f"Failed both: {gs['failed_both']}. Next-line checks run: {gs['next_line_checked']}, "
                       f"of which the next script line resolved the tension: {gs['next_line_resolved']}. "
                       f"Thresholds (config.yaml): hook >= {gs['thresholds']['hook_min_score']}, "
                       f"cliffhanger >= {gs['thresholds']['cliffhanger_min_score']}.")
            if gs.get("final_list_short"):
                if gs.get("in_final_near_misses"):
                    st.warning(f"Only {gs['in_final_passed_both']} promos passed every criterion. The final list of "
                               f"{summary['promos_ranked']} adds the {gs['in_final_near_misses']} closest near-misses, labelled NEAR MISS.")
                else:
                    st.warning(f"Only {summary['promos_ranked']} promos passed into the final list (asked for "
                               f"{summary.get('final_promo_count')}). The bar was not lowered.")
            if gs["fail_reason_counts"]:
                st.bar_chart(pd.Series(gs["fail_reason_counts"]))
            rej_path = runs[run_name] / "rejected_promos.csv"
            if rej_path.exists():
                st.subheader("Rejected promos (click to open)")
                with st.expander(f"Rejected promos ({gs['rejected']}) - which test failed and why"):
                    st.dataframe(pd.read_csv(rej_path)[["id", "ai_score", "failed", "hook_fail_reasons",
                                                        "cliffhanger_fail_reasons", "last_line_of_promo", "next_lines_in_script"]],
                                 hide_index=True)
        st.subheader("Rule-check (validation) results")
        st.write(summary["validation_status_counts"])
        if summary["issue_counts"]:
            st.bar_chart(pd.Series(summary["issue_counts"]).sort_values(ascending=False))
        st.caption(f"Near-duplicates removed: {len(summary['near_duplicates_removed'])}.  "
                   f"Tagging failures: {len(summary['tag_failures'])}.  LLM usage: {summary['llm_usage']}")
        st.subheader("Export")
        table = pd.DataFrame([{
            "rank": p["rank"], "id": p["id"], "ai_score": p["score"], "status": p["status"], "duration_s": p["duration"],
            "segments": " | ".join(f"E{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]),
            "hook": p["judge"]["hook_text"], "cliffhanger": p["judge"]["cliffhanger_text"],
            "decision": dict(zip(rv["promo_id"], rv["decision"])).get(p["id"], ""),
            "reason": dict(zip(rv["promo_id"], rv["reason"])).get(p["id"], ""),
        } for p in promos])
        st.dataframe(table, hide_index=True)
        st.download_button("Download CSV (promos + review decisions)", table.to_csv(index=False), "promos_reviewed.csv")

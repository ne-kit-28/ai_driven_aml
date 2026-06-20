"""
AML compliance dashboard (microservice UI).

Views: Investigate · Suspicious accounts · Live monitor · Verification.
Click any node in a graph to send it to the investigation field at the top.

  DASH_SOURCE=parquet|trino   DATA_DIR=...   TRINO_HOST=trino
  BLOCKLIST_PATH=/tmp/aml_blocklist.parquet   (must be writable)
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""
import os
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from graph_query import ParquetSource, TrinoSource, trace, build_graph
from chain_select import select_chain
from llm_explain import build_evidence, build_flow_evidence, explain, FLOW_PROMPT

st.set_page_config(page_title="AML Graph", layout="wide")
MODES = ["Investigate", "Suspicious accounts", "Live monitor", "Verification"]
BLOCKLIST = Path(os.environ.get("BLOCKLIST_PATH", "/tmp/aml_blocklist.parquet"))
LEGEND = ("Node colour = account toxicity (red high, green low) · size = degree · "
          "red edges = high-risk transfers · white ring = investigated / blocked · "
          "click a node to investigate it.")


@st.cache_resource
def get_source():
    if os.environ.get("DASH_SOURCE", "parquet") == "trino":
        return TrinoSource(host=os.environ.get("TRINO_HOST", "trino"))
    return ParquetSource(os.environ.get("DATA_DIR", "data"))


def load_blocked(src) -> set:
    s = set()
    try:
        if BLOCKLIST.exists():
            s |= set(pd.read_parquet(BLOCKLIST)["account_id"])
    except Exception:
        pass
    try:
        s |= src.blocked_accounts()
    except Exception:
        pass
    return s


def block_accounts(accounts, reason="suspected fraud", officer="officer"):
    accounts = list(dict.fromkeys(a for a in accounts if a))
    if not accounts:
        return
    now = pd.Timestamp.utcnow().isoformat()
    rows = pd.DataFrame([{"account_id": a, "reason": reason, "officer": officer, "ts": now}
                         for a in accounts])
    try:
        df = pd.concat([pd.read_parquet(BLOCKLIST), rows]) if BLOCKLIST.exists() else rows
        df.drop_duplicates("account_id", keep="last").to_parquet(BLOCKLIST, index=False)
    except OSError:
        pass   # read-only FS (e.g. data mounted ro) — Kafka publish below is the source of truth
    bs = os.environ.get("KAFKA_BOOTSTRAP")
    if bs:
        try:
            from kafka import KafkaProducer
            p = KafkaProducer(bootstrap_servers=bs, api_version_auto_timeout_ms=5000,
                              value_serializer=lambda v: json.dumps(v).encode())
            for a in accounts:
                p.send("blocklist", {"account_id": a, "reason": reason, "ts": int(time.time())})
            p.flush(); p.close()
        except Exception as e:
            st.warning(f"blocklist publish failed: {e}")


def goto_investigate(account):
    """Deferred navigation: applied at the top of the next run, before widgets exist."""
    st.session_state["_set_acct"] = account
    st.session_state["_goto"] = "Investigate"
    st.rerun()


def graph_panel(edges, attrs, alert, key):
    """Render the interactive graph; a node click navigates to investigate it."""
    fig, g = build_graph(edges, attrs, alert=alert, blocked=blocked)
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=key,
                         config={"displayModeBar": False})
    st.caption(f"{g.number_of_nodes()} accounts · {g.number_of_edges()} relationships. {LEGEND}")
    try:
        pts = ev["selection"]["points"]
    except (TypeError, KeyError):
        pts = getattr(getattr(ev, "selection", None), "points", None) or []
    if pts:
        cd = pts[0].get("customdata")
        clicked = cd[0] if isinstance(cd, (list, tuple)) else cd
        if clicked and clicked != alert and clicked != st.session_state.get("acct"):
            goto_investigate(clicked)
    return g


# ---- deferred navigation / account selection (must run before widgets) ----
_goto = st.session_state.pop("_goto", None)
if _goto in MODES:
    st.session_state["nav"] = _goto
_set = st.session_state.pop("_set_acct", None)
if _set is not None:
    st.session_state["acct"] = _set
st.session_state.setdefault("nav", MODES[0])

try:
    src = get_source()
except FileNotFoundError:
    st.error("Scored data not found. Generate it (generator → train → score_export) "
             "or set `DASH_SOURCE=trino` to read the lake.")
    st.stop()

# lake readiness probe
try:
    _probe = src.top_alerts(1)
except Exception as e:
    st.error(f"Cannot read the lake via Trino. Did the seed and scoring services run?\n\n```\n{e}\n```")
    st.stop()
if _probe.empty:
    st.info("No accounts in the lake yet — run the seed service first.")
    st.stop()
try:
    default_acct = _probe["account_id"].iloc[0]
except Exception:
    default_acct = ""
st.session_state.setdefault("acct", default_acct)

blocked = load_blocked(src)

# ---- sidebar: modern view selector + replay ----
st.sidebar.title("AML Graph")
if hasattr(st, "segmented_control"):
    mode = st.sidebar.segmented_control("View", MODES, key="nav")
else:
    mode = st.sidebar.radio("View", MODES, key="nav", horizontal=False)
mode = mode or st.session_state.get("nav") or MODES[0]

if st.sidebar.toggle("Live replay (real-time simulation)", value=False):
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2500, key="tick")
    t0, t1 = src.ts_range
    if t1 <= t0:
        st.sidebar.warning("No SCORED data yet — run ETL + scoring.")
        src.max_ts = None
    else:
        step = max(1, (t1 - t0) // 40)
        st.session_state["cursor"] = min(t1, st.session_state.get("cursor", t0) + step)
        src.max_ts = st.session_state["cursor"]
        st.sidebar.progress((src.max_ts - t0) / max(1, t1 - t0), text="stream position")
else:
    src.max_ts = None

if "toxicity" in _probe and bool(_probe["toxicity"].isna().all()):
    st.warning("Accounts are seeded but not scored yet — start the scoring service.")

# ============================ INVESTIGATE ============================
if mode == "Investigate":
    top = st.columns([4, 1])
    acct = top[0].text_input("Account to investigate", key="acct",
                             placeholder="e.g. ACC0000123").strip()
    depth = top[1].number_input("Trace depth", 1, 15, 4)
    with st.sidebar.expander("Graph detail"):
        fanout = st.slider("edges per node (by risk)", 2, 15, 6)
        max_nodes = st.slider("max nodes", 20, 250, 90)

    if not acct:
        st.info("Enter an account, or click a node in any graph to investigate it.")
        st.stop()
    if not src.has_account(acct):
        st.error(f"Account `{acct}` not found.")
        st.stop()

    st.subheader(f"{acct} — ego-network (depth {depth})")
    edges, attrs = trace(src, acct, depth=int(depth), fanout=fanout, max_nodes=max_nodes)

    left, right = st.columns([4, 1.3])
    with left:
        g = graph_panel(edges, attrs, alert=acct, key="g_invest")
        if g.number_of_nodes() >= max_nodes:
            st.caption(f"Graph capped at max nodes = {max_nodes}; raise it in the sidebar to see more.")
    with right:
        tox = float(attrs.loc[acct, "toxicity"] or 0) if acct in attrs.index else 0.0
        st.metric("Account toxicity", f"{tox:.2f}")
        st.metric("Nodes / transfers", f"{g.number_of_nodes()} / {g.number_of_edges()}")
        if acct in blocked:
            st.error("Already blocked")
        else:
            reason = st.text_input("Block reason", "suspected dropper")
            if st.button("Block account", type="primary", use_container_width=True):
                block_accounts([acct], reason); st.success(f"{acct} blocked"); st.rerun()

        st.divider()
        st.markdown("**Block the whole chain**")
        chain_thr = st.slider("include nodes with toxicity ≥", 0.0, 1.0, 0.5, 0.05,
                              key="chain_thr", help="legit hubs are always excluded")
        chain = select_chain(edges, attrs, acct, tox_threshold=chain_thr)
        to_block = sorted(chain - blocked)
        st.caption(f"Chain: {len(chain)} accounts · {len(to_block)} not yet blocked")
        with st.expander("chain members"):
            st.write(to_block or "—")
        if to_block and st.button(f"Block chain ({len(to_block)})", use_container_width=True):
            block_accounts(to_block, "suspected fraud chain")
            st.success(f"Blocked {len(to_block)} accounts"); st.rerun()

    st.markdown("#### Transactions of this account (by risk)")
    incident = src.incident_edges({acct})
    inc = incident.sort_values("risk_score", ascending=False).head(50).copy()
    inc["ts"] = pd.to_datetime(inc["ts"], unit="s")
    st.dataframe(inc.rename(columns={"source_account": "from", "target_account": "to",
                                     "risk_score": "risk"}), hide_index=True, height=240)

    st.markdown("#### AI explanation")
    bcol = st.columns(2)
    if bcol[0].button("Why is this a likely dropper? (SAR)", key="explain_btn"):
        with st.spinner("Collecting evidence and asking the model…"):
            st.session_state["sar"] = explain(build_evidence(acct, edges, attrs, incident))
            st.session_state["sar_acct"] = acct
    if bcol[1].button("In / out flow analysis", key="flow_btn"):
        with st.spinner("Analysing the account's flows…"):
            cps = set(incident.source_account) | set(incident.target_account) | {acct}
            fattrs = src.node_attrs(cps)
            st.session_state["flow"] = explain(build_flow_evidence(acct, fattrs, incident), system=FLOW_PROMPT)
            st.session_state["flow_acct"] = acct
    if st.session_state.get("sar") and st.session_state.get("sar_acct") == acct:
        st.markdown("**SAR verdict:**"); st.markdown(st.session_state["sar"])
    if st.session_state.get("flow") and st.session_state.get("flow_acct") == acct:
        st.markdown("**Flow analysis:**"); st.markdown(st.session_state["flow"])

# ============================ SUSPICIOUS ACCOUNTS ============================
elif mode == "Suspicious accounts":
    st.subheader("Suspicious accounts, ranked by toxicity")
    if st.toggle("Live (auto-refresh 5s)", value=False, key="nodes_live"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="nodes_tick")
    thr = st.slider("toxicity threshold", 0.0, 1.0, 0.5, 0.05)
    acc = src.scored_accounts_full()
    acc = acc[acc.toxicity.notna()]
    flagged = acc[acc.toxicity >= thr].sort_values("toxicity", ascending=False)
    m = st.columns(3)
    m[0].metric("flagged (≥ threshold)", len(flagged))
    m[1].metric("blocked", len(blocked))
    m[2].metric("total scored", len(acc))
    st.dataframe(flagged[["account_id", "fraud_role", "is_fraud", "toxicity"]].head(200),
                 hide_index=True, height=320)
    cpick = st.columns([3, 1, 1])
    pick = cpick[0].selectbox("account", flagged["account_id"].tolist()) if len(flagged) else None
    if pick and cpick[1].button("Investigate", use_container_width=True):
        goto_investigate(pick)
    if pick and cpick[2].button("Block", type="primary", use_container_width=True):
        block_accounts([pick], "suspected dropper"); st.success(f"{pick} blocked"); st.rerun()

    st.markdown("#### Network of the most toxic accounts")
    ktop = st.slider("top toxic accounts to show", 5, 40, 15)
    seeds = flagged["account_id"].head(ktop).tolist()
    if seeds:
        ed = src.incident_edges(set(seeds)).sort_values("risk_score", ascending=False).head(150)
        nodes = set(ed.source_account) | set(ed.target_account) | set(seeds)
        graph_panel(ed, src.node_attrs(nodes), alert=None, key="g_susp")
    else:
        st.info("No accounts above the threshold.")

# ============================ VERIFICATION ============================
elif mode == "Verification":
    st.subheader("Verification — model vs ground truth")
    thr = st.slider("toxicity threshold", 0.0, 1.0, 0.5, 0.05)
    acc = src.scored_accounts_full()
    acc = acc[acc.toxicity.notna()]
    if acc.empty:
        st.info("No scores yet — run ETL + scoring."); st.stop()
    fr, le = acc[acc.is_fraud == 1], acc[acc.is_fraud == 0]
    m = st.columns(4)
    m[0].metric("recall (fraud ≥ thr)", f"{(fr.toxicity >= thr).mean():.2f}" if len(fr) else "—")
    m[1].metric("fraud mean toxicity", f"{fr.toxicity.mean():.2f}" if len(fr) else "—")
    m[2].metric("legit mean toxicity", f"{le.toxicity.mean():.2f}" if len(le) else "—")
    m[3].metric("blocked", len(blocked))
    st.markdown("#### Recall by typology")
    typ = fr.copy()
    typ["case"] = typ["typology_id"].fillna("?").str.split("_").str[0]
    rep = typ.groupby("case").apply(
        lambda g: pd.Series({"accounts": len(g), "caught (≥thr)": int((g.toxicity >= thr).sum()),
                             "recall": round((g.toxicity >= thr).mean(), 2),
                             "max_toxicity": round(g.toxicity.max(), 2)})).reset_index()
    st.dataframe(rep, hide_index=True)
    st.caption("Recall rising and legit mean toxicity falling after blocks means the system is working.")

# ============================ LIVE MONITOR ============================
else:
    st.subheader("Latest suspicious transactions")
    if st.toggle("Live (auto-refresh 5s)", value=False, key="mon_live"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="mon_tick")
    c = st.columns(4)
    thr = c[0].slider("risk threshold", 0.0, 1.0, 0.5, 0.05)
    limit = c[1].number_input("rows", 20, 2000, 200, step=20)
    only_susp = not c[2].toggle("show all", value=False)
    show_graph = c[3].toggle("show network", value=True)

    feed = src.recent(min_risk=thr, limit=int(limit), only_susp=only_susp).copy()
    susp_n = int(src.recent(min_risk=thr, limit=100000, only_susp=True).shape[0])
    m = st.columns(3)
    m[0].metric("suspicious tx (≥ thr)", susp_n)
    m[1].metric("rows shown", len(feed))
    top = src.top_alerts(1)
    if len(top):
        _tt = top["toxicity"].iloc[0]
        val = f"{top['account_id'].iloc[0][-5:]} · " + (f"{_tt:.2f}" if _tt is not None and _tt == _tt else "—")
    else:
        val = "—"
    m[2].metric("top toxic account", val)

    disp = feed.copy(); disp["ts"] = pd.to_datetime(disp["ts"], unit="s")
    st.dataframe(disp.rename(columns={"source_account": "from", "target_account": "to",
                                      "risk_score": "risk"}), hide_index=True, height=300)

    if show_graph and not feed.empty:
        st.markdown("#### Top suspicious network")
        gn = st.slider("transactions in graph", 50, 600, 200, 25, key="mon_graph_n")
        ed = src.top_risk_edges(gn)
        nodes = set(ed.source_account) | set(ed.target_account)
        graph_panel(ed, src.node_attrs(nodes), alert=None, key="g_mon")

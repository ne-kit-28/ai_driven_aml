"""
AML compliance dashboard (microservice UI).

Views: Investigate · Suspicious accounts · Live monitor · Verification.
Filters live in the sidebar; click any graph node to investigate it. Graphs are
draggable and zoomable (vis.js via streamlit-agraph).

  DASH_SOURCE=parquet|trino   DATA_DIR=...   TRINO_HOST=trino
  BLOCKLIST_PATH=/tmp/aml_blocklist.parquet   (must be writable)
"""
import os
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from graph_query import ParquetSource, TrinoSource, trace, build_graph
from chain_select import select_chain
from llm_explain import build_evidence, build_flow_evidence, explain, FLOW_PROMPT

st.set_page_config(page_title="AML Graph", layout="wide")
MODES = ["Investigate", "Suspicious accounts", "Live monitor", "Verification"]
BLOCKLIST = Path(os.environ.get("BLOCKLIST_PATH", "/tmp/aml_blocklist.parquet"))
# staticGraphWithDragAndDrop: keep our precomputed layout, no physics drift, still draggable + zoomable
GRAPH_CFG = Config(width=1150, height=560, directed=True, physics=False,
                   staticGraphWithDragAndDrop=True, nodeHighlightBehavior=True,
                   highlightColor="#F2A900", collapsible=False, backgroundColor="#10141A")
LEGEND = ("Node colour = account toxicity (red = high, green = low) · size = degree · "
          "edge colour = transaction risk · click a node to investigate · drag to rearrange · scroll to zoom.")


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
        pass   # read-only FS — Kafka publish below is the source of truth on the stand
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
    nspec, espec, g = build_graph(edges, attrs, alert=alert, blocked=blocked)
    nodes = [Node(id=n["id"], label=n["label"], size=n["size"], shape="dot", title=n["title"],
                  x=n["x"], y=n["y"], color={"background": n["color"], "border": n["border"]},
                  borderWidth=n["borderWidth"])
             for n in nspec]
    eds = [Edge(source=e["source"], target=e["target"], color=e["color"],
                width=e["width"], title=e["title"]) for e in espec]
    st.caption(f"{g.number_of_nodes()} accounts · {g.number_of_edges()} relationships")
    clicked = agraph(nodes=nodes, edges=eds, config=GRAPH_CFG)
    st.caption(LEGEND)
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

try:
    _probe = src.top_alerts(1)
except Exception as e:
    st.error(f"Cannot read the lake via Trino. Did the seed and scoring services run?\n\n```\n{e}\n```")
    st.stop()
if _probe.empty:
    st.info("No accounts in the lake yet — run the seed service first.")
    st.stop()
st.session_state.setdefault("acct", _probe["account_id"].iloc[0])
blocked = load_blocked(src)

# ---- sidebar: view selector + per-view filters (top to bottom) ----
st.sidebar.title("AML Graph")
mode = st.sidebar.radio("View", MODES, key="nav")   # vertical list, one option per line
mode = mode or st.session_state.get("nav") or MODES[0]
st.sidebar.divider()
st.sidebar.markdown("**Filters**")

if st.sidebar.toggle("Live replay (simulate real-time arrival)", value=False):
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
    depth = st.sidebar.slider("How many hops to trace", 1, 15, 4)
    fanout = st.sidebar.slider("Top transfers to follow per account", 2, 15, 6)
    max_nodes = st.sidebar.slider("Max accounts in the graph", 20, 250, 90)

    acct = st.text_input("Account to investigate", key="acct",
                         placeholder="e.g. ACC0000123").strip()
    if not acct:
        st.info("Enter an account, or click a node in any graph to investigate it.")
        st.stop()
    if not src.has_account(acct):
        st.error(f"Account `{acct}` not found.")
        st.stop()

    st.subheader(f"{acct} — ego-network (depth {depth})")
    edges, attrs = trace(src, acct, depth=int(depth), fanout=fanout, max_nodes=max_nodes)

    tox = float(attrs.loc[acct, "toxicity"] or 0) if acct in attrs.index else 0.0
    k = st.columns(3)
    k[0].metric("Account toxicity", f"{tox:.2f}")
    k[1].metric("Status", "BLOCKED" if acct in blocked else "active")
    k[2].metric("Accounts in view", len(attrs))

    g = graph_panel(edges, attrs, alert=acct, key="g_invest")
    if g.number_of_nodes() >= max_nodes:
        st.caption(f"Graph capped at {max_nodes} accounts; raise the limit in the sidebar to see more.")

    st.markdown("#### Take action")
    a1, a2 = st.columns(2)
    with a1:
        if acct in blocked:
            st.info("This account is already blocked.")
        else:
            reason = st.text_input("Block reason", "suspected dropper")
            if st.button("Block this account", type="primary", use_container_width=True):
                block_accounts([acct], reason); st.success(f"{acct} blocked"); st.rerun()
    with a2:
        chain_thr = st.slider("Chain: include accounts with toxicity ≥", 0.0, 1.0, 0.5, 0.05,
                              key="chain_thr", help="legitimate hubs are always excluded")
        chain = select_chain(edges, attrs, acct, tox_threshold=chain_thr)
        to_block = sorted(chain - blocked)
        around = [a for a in to_block if a != acct]            # the chain minus this account
        st.caption(f"Chain: {len(chain)} accounts · {len(to_block)} not yet blocked")
        if to_block and st.button(f"Block whole chain ({len(to_block)})", use_container_width=True):
            block_accounts(to_block, "suspected fraud chain")
            st.success(f"Blocked {len(to_block)} accounts"); st.rerun()
        if around and st.button(f"Block fraud around it, keep this account ({len(around)})",
                                use_container_width=True,
                                help="for a contaminated legit account: block the surrounding fraud so it recovers"):
            block_accounts(around, "fraud feeding a victim")
            st.success(f"Blocked {len(around)} surrounding accounts; {acct} kept"); st.rerun()

    st.markdown("#### Transactions of this account (highest risk first)")
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
            st.session_state["flow"] = explain(build_flow_evidence(acct, src.node_attrs(cps), incident),
                                               system=FLOW_PROMPT)
            st.session_state["flow_acct"] = acct
    if st.session_state.get("sar") and st.session_state.get("sar_acct") == acct:
        st.markdown("**SAR verdict:**"); st.markdown(st.session_state["sar"])
    if st.session_state.get("flow") and st.session_state.get("flow_acct") == acct:
        st.markdown("**Flow analysis:**"); st.markdown(st.session_state["flow"])

# ============================ SUSPICIOUS ACCOUNTS ============================
elif mode == "Suspicious accounts":
    thr = st.sidebar.slider("Minimum account toxicity", 0.0, 1.0, 0.5, 0.05)
    ktop = st.sidebar.slider("Accounts to show in the network", 5, 40, 15)
    if st.sidebar.toggle("Auto-refresh (5s)", value=False, key="nodes_live"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="nodes_tick")

    st.subheader("Suspicious accounts, ranked by toxicity")
    acc = src.scored_accounts_full()
    acc = acc[acc.toxicity.notna()]
    flagged = acc[acc.toxicity >= thr].sort_values("toxicity", ascending=False)
    m = st.columns(3)
    m[0].metric("Flagged (≥ threshold)", len(flagged))
    m[1].metric("Blocked", len(blocked))
    m[2].metric("Total scored", len(acc))

    st.dataframe(flagged[["account_id", "fraud_role", "is_fraud", "toxicity"]].head(200),
                 hide_index=True, height=320)
    cpick = st.columns([3, 1, 1])
    pick = cpick[0].selectbox("Pick an account", flagged["account_id"].tolist()) if len(flagged) else None
    if pick and cpick[1].button("Investigate", use_container_width=True):
        goto_investigate(pick)
    if pick and cpick[2].button("Block", type="primary", use_container_width=True):
        block_accounts([pick], "suspected dropper"); st.success(f"{pick} blocked"); st.rerun()

    st.markdown("#### Network of the most toxic accounts")
    seeds = flagged["account_id"].head(ktop).tolist()
    if seeds:
        ed = src.incident_edges(set(seeds)).sort_values("risk_score", ascending=False).head(150)
        nodes = set(ed.source_account) | set(ed.target_account) | set(seeds)
        graph_panel(ed, src.node_attrs(nodes), alert=None, key="g_susp")
    else:
        st.info("No accounts above the threshold.")

# ============================ VERIFICATION ============================
elif mode == "Verification":
    thr = st.sidebar.slider("Minimum account toxicity (alert threshold)", 0.0, 1.0, 0.5, 0.05)

    st.subheader("Verification — does the model catch the known fraud?")
    st.caption("Ground truth is the labelled fraud accounts. This view measures how well the model's "
               "account toxicity separates fraud from legitimate accounts.")
    acc = src.scored_accounts_full()
    acc = acc[acc.toxicity.notna()]
    if acc.empty:
        st.info("No scores yet — run ETL + scoring."); st.stop()
    fr, le = acc[acc.is_fraud == 1], acc[acc.is_fraud == 0]

    # rank-based ROC-AUC (threshold-independent; meaningful even before memory warms up)
    def roc_auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        r = pd.concat([pos, neg]).rank(method="average")
        return (r.iloc[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    auc = roc_auc(fr.toxicity, le.toxicity)
    tp = int((fr.toxicity >= thr).sum()); fn = len(fr) - tp
    fp = int((le.toxicity >= thr).sum()); tn = len(le) - fp
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    k = st.columns(4)
    k[0].metric("Ranking quality (ROC-AUC)", f"{auc:.3f}",
                help="How well toxicity ranks fraud above legit. 1.0 = perfect, 0.5 = random.")
    k[1].metric("Precision @ threshold", f"{prec:.0%}" if prec == prec else "—",
                help="Of accounts the model flags, how many are truly fraud.")
    k[2].metric("Recall @ threshold", f"{rec:.0%}" if rec == rec else "—",
                help="Of all fraud accounts, how many the model flags.")
    k[3].metric("Mean toxicity: fraud vs legit",
                f"{fr.toxicity.mean():.2f} / {le.toxicity.mean():.2f}" if len(fr) and len(le) else "—",
                help="The gap between these two is the separation the model achieves.")

    if len(fr) and fr.toxicity.max() < 0.10:
        st.info("Model memory is still warming up — toxicity is low across the board. Ranking quality "
                "above is already meaningful; threshold metrics will rise as scoring runs more cycles.")

    st.markdown("#### At the current alert threshold")
    cc = st.columns(4)
    cc[0].metric("Fraud caught", tp)
    cc[1].metric("Fraud missed", fn)
    cc[2].metric("False alarms", fp)
    cc[3].metric("Legit cleared", tn)

    st.markdown("#### Toxicity distribution — fraud vs legit")
    bedges = [i / 10 for i in range(11)]
    labels = [f"{int(bedges[i]*100)}–{int(bedges[i+1]*100)}%" for i in range(10)]
    frb = pd.cut(fr.toxicity, bedges, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    leb = pd.cut(le.toxicity, bedges, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    dist = pd.DataFrame({"fraud": frb, "legit": leb})
    st.bar_chart(dist, color=["#E4572E", "#2BB673"])
    st.caption("A working model pushes the red (fraud) bars to the right and the green (legit) bars to the left.")

    st.markdown("#### Recall by laundering typology")
    typ = fr.copy()
    typ["typology"] = typ["typology_id"].fillna("?").str.split("_").str[0]
    rep = typ.groupby("typology").apply(
        lambda g: pd.Series({"fraud accounts": len(g),
                             "caught (≥ thr)": int((g.toxicity >= thr).sum()),
                             "recall": round((g.toxicity >= thr).mean(), 2),
                             "max toxicity": round(g.toxicity.max(), 2)})).reset_index()
    st.dataframe(rep, hide_index=True)

# ============================ LIVE MONITOR ============================
else:
    thr = st.sidebar.slider("Minimum transaction risk", 0.0, 1.0, 0.5, 0.05)
    limit = st.sidebar.number_input("Rows to show", 20, 2000, 200, step=20)
    only_susp = not st.sidebar.toggle("Include low-risk transactions", value=False)
    show_graph = st.sidebar.toggle("Show network graph", value=True)
    gn = st.sidebar.slider("Max transactions in the network", 50, 600, 200, 25, key="mon_graph_n")
    if st.sidebar.toggle("Auto-refresh (5s)", value=False, key="mon_live"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="mon_tick")

    st.subheader("Latest suspicious transactions")
    feed = src.recent(min_risk=thr, limit=int(limit), only_susp=only_susp).copy()
    susp_n = int(src.recent(min_risk=thr, limit=100000, only_susp=True).shape[0])
    m = st.columns(3)
    m[0].metric("Suspicious tx (≥ threshold)", susp_n)
    m[1].metric("Rows shown", len(feed))
    top = src.top_alerts(1)
    if len(top):
        _tt = top["toxicity"].iloc[0]
        val = f"{top['account_id'].iloc[0][-5:]} · " + (f"{_tt:.2f}" if _tt is not None and _tt == _tt else "—")
    else:
        val = "—"
    m[2].metric("Top toxic account", val)

    disp = feed.copy(); disp["ts"] = pd.to_datetime(disp["ts"], unit="s")
    st.dataframe(disp.rename(columns={"source_account": "from", "target_account": "to",
                                      "risk_score": "risk"}), hide_index=True, height=300)

    if show_graph and not feed.empty:
        st.markdown("#### Network of the riskiest transactions")
        ed = src.top_risk_edges(gn)
        nodes = set(ed.source_account) | set(ed.target_account)
        graph_panel(ed, src.node_attrs(nodes), alert=None, key="g_mon")

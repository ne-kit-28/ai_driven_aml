"""
AML compliance dashboard (microservice UI). Two modes:
  🔎 Investigate — enter a wallet id, render its ego-graph to a chosen depth,
                   inspect its red (high-risk) transactions, and BLOCK it.
  📡 Monitor     — feed of the latest suspicious transactions across the lake.

  DASH_SOURCE=parquet|trino   DATA_DIR=...   TRINO_HOST=trino
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graph_query import ParquetSource, TrinoSource, trace, build_network
from llm_explain import build_evidence, build_flow_evidence, explain, FLOW_PROMPT

st.set_page_config(page_title="AML Graph", layout="wide", page_icon="🕸️")
BLOCKLIST = Path(os.environ.get("DATA_DIR", "data")) / "blocklist.parquet"


@st.cache_resource
def get_source():
    if os.environ.get("DASH_SOURCE", "parquet") == "trino":
        return TrinoSource(host=os.environ.get("TRINO_HOST", "trino"))
    return ParquetSource(os.environ.get("DATA_DIR", "data"))


def load_blocked() -> set:
    if BLOCKLIST.exists():
        return set(pd.read_parquet(BLOCKLIST)["account_id"])
    return set()


def block_account(account, reason, officer="officer"):
    row = pd.DataFrame([{"account_id": account, "reason": reason, "officer": officer,
                         "ts": pd.Timestamp.utcnow().isoformat()}])
    df = pd.concat([pd.read_parquet(BLOCKLIST), row]) if BLOCKLIST.exists() else row
    df.drop_duplicates("account_id", keep="last").to_parquet(BLOCKLIST, index=False)


def embed(net):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        net.write_html(f.name, notebook=False)
        components.html(open(f.name).read(), height=730, scrolling=False)


try:
    src = get_source()
except FileNotFoundError:
    st.error(
        "Scored data not found at `$DATA_DIR`.\n\n"
        "Generate it on the host (needs torch), then it appears via the mounted volume:\n"
        "```\npython src/generator/generate_graph.py --out data\n"
        "python src/ml/train_temporal.py --data data --out src/ml/artifacts\n"
        "python src/ml/score_export.py --data data --out data\n```\n"
        "…or set `DASH_SOURCE=trino` to read the lake.")
    st.stop()

st.sidebar.title("🕸️ AML Graph")
mode = st.sidebar.radio("Mode", ["🔎 Investigate account", "📡 Monitor suspicious"], key="mode")

# global live-replay (simulate real-time arrival)
if st.sidebar.toggle("▶ Live replay (real-time sim)", value=False):
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2500, key="tick")
    t0, t1 = src.ts_range
    if t1 <= t0:
        st.sidebar.warning("нет SCORED-данных — запусти ETL + scoring")
        src.max_ts = None
    else:
        step = max(1, (t1 - t0) // 40)
        st.session_state["cursor"] = min(t1, st.session_state.get("cursor", t0) + step)
        src.max_ts = st.session_state["cursor"]
        st.sidebar.progress((src.max_ts - t0) / max(1, t1 - t0), text="stream position")
else:
    src.max_ts = None

blocked = load_blocked()
LEGEND = ("🔴 high toxicity (likely dropper/mule) · 🟢 low (legit) · "
          "edge thickness/red = transaction risk · ⛔ blocked")

# lake readiness probe (Trino mode): tables may not exist / be scored yet
try:
    _probe = src.top_alerts(1)
except Exception as e:
    st.error("Cannot read the lake via Trino. Did the **seed** and **scoring** services run?\n\n"
             "Expected tables `scored_transactions` and `accounts_state`.\n\n"
             f"```\n{e}\n```")
    st.stop()
if _probe.empty:
    st.info("No accounts in the lake yet — run the **seed** service first.")
    st.stop()
if "toxicity" in _probe and bool(_probe["toxicity"].isna().all()):
    st.warning("Accounts are seeded but not scored yet — start the **scoring** service "
               "(`docker compose --profile scoring up -d scoring`).")

# ============================ INVESTIGATE ============================
if mode == "🔎 Investigate account":
    default_acct = src.top_alerts(1)["account_id"].iloc[0]
    acct = st.sidebar.text_input("Wallet / account id", key="acct",
                                 value=st.session_state.get("acct", default_acct)).strip()
    depth = st.sidebar.number_input("Trace depth (hops)", 1, 15, 4,
                                    help="left empty -> default 4")
    with st.sidebar.expander("advanced"):
        fanout = st.slider("edges per node (by risk)", 2, 15, 6)
        max_nodes = st.slider("max nodes", 20, 200, 80)

    if not src.has_account(acct):
        st.error(f"account `{acct}` not found"); st.stop()

    st.markdown(f"### 🔎 `{acct}` — ego-network, depth {depth}")
    edges, attrs = trace(src, acct, depth=int(depth), fanout=fanout, max_nodes=max_nodes)
    net, g = build_network(edges, attrs, alert=acct, blocked=blocked)

    left, right = st.columns([4, 1.3])
    with left:
        embed(net); st.caption(LEGEND)
        if g.number_of_nodes() >= max_nodes:
            st.caption(f"⚠️ Граф обрезан до лимита **Max nodes = {max_nodes}** — поэтому увеличение глубины "
                       f"дальше может не менять картинку. Подними «Max nodes» или снизь «edges per node» в *advanced*.")
    with right:
        tox = float(attrs.loc[acct, "toxicity"] or 0) if acct in attrs.index else 0.0
        st.metric("dropper toxicity", f"{tox:.2f}")
        st.metric("nodes / tx in view", f"{g.number_of_nodes()} / {g.number_of_edges()}")
        if acct in blocked:
            st.error("⛔ already BLOCKED")
        else:
            reason = st.text_input("block reason", "suspected dropper")
            if st.button("⛔ Block account", type="primary", use_container_width=True):
                block_account(acct, reason); st.success(f"{acct} blocked"); st.rerun()

    st.markdown("#### Transactions of this account (by risk)")
    incident = src.incident_edges({acct})
    inc = incident.sort_values("risk_score", ascending=False).head(50).copy()
    inc["ts"] = pd.to_datetime(inc["ts"], unit="s")
    st.dataframe(inc.rename(columns={"source_account": "from", "target_account": "to",
                                     "risk_score": "risk"}), hide_index=True, height=240)

    st.markdown("#### 🧠 AI (LLM)")
    bcol = st.columns(2)
    if bcol[0].button("SAR: почему подозрение на дроппера", key="explain_btn"):
        with st.spinner("Собираю улики из эго-графа и спрашиваю ИИ…"):
            st.session_state["sar"] = explain(build_evidence(acct, edges, attrs, incident))
            st.session_state["sar_acct"] = acct
    if bcol[1].button("Анализ входящих / исходящих", key="flow_btn"):
        with st.spinner("Разбираю потоки счёта…"):
            cps = set(incident.source_account) | set(incident.target_account) | {acct}
            fattrs = src.node_attrs(cps)
            st.session_state["flow"] = explain(build_flow_evidence(acct, fattrs, incident), system=FLOW_PROMPT)
            st.session_state["flow_acct"] = acct
    if st.session_state.get("sar") and st.session_state.get("sar_acct") == acct:
        st.markdown("**SAR — вердикт:**"); st.markdown(st.session_state["sar"])
    if st.session_state.get("flow") and st.session_state.get("flow_acct") == acct:
        st.markdown("**Разбор потоков (вход/выход):**"); st.markdown(st.session_state["flow"])
    with st.expander("улики SAR, отправленные модели"):
        st.code(build_evidence(acct, edges, attrs, incident))

# ============================ MONITOR ============================
else:
    st.markdown("### 📡 Latest suspicious transactions")
    if st.toggle("🔴 Live (auto-refresh, 5s)", value=False, key="mon_live"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="mon_tick")
        try:
            _t0, _t1 = src.ts_range
            st.caption(f"🔴 live — обновление каждые 5с · последняя активность: "
                       f"{pd.to_datetime(_t1, unit='s')} · новые SCORED-транзакции появляются по мере записи scoring-сервисом")
        except Exception:
            st.caption("🔴 live — обновление каждые 5с")
    c = st.columns(4)
    thr = c[0].slider("risk threshold", 0.0, 1.0, 0.5, 0.05)
    limit = c[1].number_input("rows", 20, 2000, 200, step=20)
    only_susp = not c[2].toggle("show ALL (not just suspicious)", value=False)
    show_graph = c[3].toggle("show suspicious network", value=True)

    feed = src.recent(min_risk=thr, limit=int(limit), only_susp=only_susp).copy()
    susp_n = int((src.recent(min_risk=thr, limit=100000, only_susp=True)).shape[0])
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

    # jump to investigate any account from the feed
    accts = sorted(set(feed["source_account"]) | set(feed["target_account"]))
    jc = st.columns([3, 1])
    pick = jc[0].selectbox("investigate account from feed", accts) if accts else None
    if pick and jc[1].button("🔎 Investigate", use_container_width=True):
        st.session_state["acct"] = pick; st.session_state["mode"] = "🔎 Investigate account"; st.rerun()

    if show_graph and not feed.empty:
        ed = src.top_risk_edges(80)
        nodes = set(ed.source_account) | set(ed.target_account)
        net, g = build_network(ed, src.node_attrs(nodes), alert=None, blocked=blocked)
        st.markdown("#### Top suspicious network")
        embed(net); st.caption(LEGEND)

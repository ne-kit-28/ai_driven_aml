"""
Graph retrieval + rendering for the AML dashboard.

trace(): follows the money path from an alert BOTH directions (upstream where
funds came from, downstream where they went) up to `depth` hops, expanding only
along the highest-risk edges per node. Legit hubs (many low-risk edges) are not
traversed through, so deep traversal reconstructs the laundering path end-to-end
instead of exploding into a hairball.

Sources share one interface:
  ParquetSource  — local scored parquet (offline demo, supports a time cursor)
  TrinoSource    — Iceberg via Trino (MVP / stand)
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
from pyvis.network import Network

TOX = cm.get_cmap("RdYlGn_r")
RISK = mcolors.LinearSegmentedColormap.from_list("risk", ["#9aa0a6", "#d62728"])


class ParquetSource:
    def __init__(self, data="data"):
        self.tx = pd.read_parquet(f"{data}/scored_transactions.parquet")
        self.acc = pd.read_parquet(f"{data}/scored_accounts.parquet").set_index("account_id")
        self.max_ts = None                       # time cursor for the real-time replay demo

    @property
    def ts_range(self):
        return int(self.tx.ts.min()), int(self.tx.ts.max())

    def _visible(self):
        return self.tx if self.max_ts is None else self.tx[self.tx.ts <= self.max_ts]

    def incident_edges(self, accts):
        t = self._visible()
        return t[t.source_account.isin(accts) | t.target_account.isin(accts)]

    def node_attrs(self, accts):
        return self.acc.reindex(list(accts))

    def top_alerts(self, n=15):
        a = self.acc.reset_index()
        return a.nlargest(n, "toxicity")[["account_id", "fraud_role", "toxicity"]]

    def recent(self, min_risk=0.0, limit=200, only_susp=True):
        t = self._visible()
        if only_susp:
            t = t[t.risk_score >= min_risk]
        return t.sort_values("ts", ascending=False).head(limit)[
            ["tx_id", "source_account", "target_account", "amount", "ts", "risk_score"]]

    def top_risk_edges(self, n=60):
        return self._visible().nlargest(n, "risk_score")

    def has_account(self, a):
        return a in self.acc.index


class TrinoSource:
    def __init__(self, host="trino", port=8080, user="dashboard",
                 catalog="iceberg", schema="banking"):
        import trino
        self.conn = trino.dbapi.connect(host=host, port=port, user=user,
                                        catalog=catalog, schema=schema)
        self.max_ts = None

    def _q(self, sql, params=None):
        cur = self.conn.cursor(); cur.execute(sql, params or [])
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)

    @property
    def ts_range(self):
        r = self._q("SELECT min(ts), max(ts) FROM scored_transactions")
        a, b = r.iloc[0, 0], r.iloc[0, 1]
        if a is None or b is None:      # no SCORED rows yet
            return 0, 0
        return int(a), int(b)

    def _ts_clause(self):
        return f" AND ts <= {int(self.max_ts)}" if self.max_ts is not None else ""

    def incident_edges(self, accts):
        accts = list(accts); ph = ",".join(["?"] * len(accts))
        sql = (f"SELECT tx_id, source_account, target_account, amount, ts, risk_score "
               f"FROM scored_transactions WHERE (source_account IN ({ph}) OR target_account IN ({ph}))"
               f"{self._ts_clause()}")
        return self._q(sql, accts + accts)

    def node_attrs(self, accts):
        accts = list(accts); ph = ",".join(["?"] * len(accts))
        return self._q(f"SELECT account_id, fraud_role, is_fraud, toxicity "
                       f"FROM accounts_state WHERE account_id IN ({ph})", accts).set_index("account_id")

    def top_alerts(self, n=15):
        return self._q(f"SELECT account_id, fraud_role, toxicity FROM accounts_state "
                       f"ORDER BY toxicity DESC LIMIT {n}")

    def recent(self, min_risk=0.0, limit=200, only_susp=True):
        conds = []
        if only_susp:
            conds.append(f"risk_score >= {min_risk}")
        if self.max_ts is not None:
            conds.append(f"ts <= {int(self.max_ts)}")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return self._q(f"SELECT tx_id, source_account, target_account, amount, ts, risk_score "
                       f"FROM scored_transactions {where} ORDER BY ts DESC LIMIT {limit}")

    def top_risk_edges(self, n=60):
        where = f"WHERE ts <= {int(self.max_ts)}" if self.max_ts is not None else ""
        return self._q(f"SELECT tx_id, source_account, target_account, amount, ts, risk_score "
                       f"FROM scored_transactions {where} ORDER BY risk_score DESC LIMIT {n}")

    def has_account(self, a):
        return not self._q("SELECT 1 FROM accounts_state WHERE account_id = ? LIMIT 1", [a]).empty


def trace(source, alert, depth=4, fanout=6, max_nodes=80):
    """Follow the top-risk edges from `alert`, both directions, up to `depth`."""
    nodes = {alert}; frontier = {alert}; kept = []
    for _ in range(depth):
        if not frontier or len(nodes) >= max_nodes:
            break
        e = source.incident_edges(frontier)
        if e.empty:
            break
        e = e.assign(risk_score=e.risk_score.fillna(0.0))
        admit = [e[(e.source_account == n) | (e.target_account == n)]
                 .sort_values("risk_score", ascending=False).head(fanout) for n in frontier]
        a = pd.concat(admit).drop_duplicates("tx_id")
        kept.append(a)
        new = (set(a.source_account) | set(a.target_account)) - nodes
        new = set(list(new)[: max(0, max_nodes - len(nodes))])
        nodes |= new; frontier = new
    if not kept:
        return pd.DataFrame(columns=source.incident_edges({alert}).columns), source.node_attrs({alert})
    ed = pd.concat(kept).drop_duplicates("tx_id")
    ed = ed[ed.source_account.isin(nodes) & ed.target_account.isin(nodes)]
    return ed, source.node_attrs(set(ed.source_account) | set(ed.target_account) | {alert})


def _hex(cmap, v):
    return mcolors.to_hex(cmap(float(max(0.0, min(1.0, v)))))


def build_network(edges, attrs, alert=None, blocked=()):
    g = nx.DiGraph()
    for a, row in attrs.iterrows():
        g.add_node(a, tox=float(row.get("toxicity", 0) or 0), role=str(row.get("fraud_role", "")),
                   fraud=int(row.get("is_fraud", 0) or 0))
    for _, e in edges.iterrows():
        g.add_edge(e.source_account, e.target_account,
                   amount=float(e.amount), risk=float(e.risk_score or 0))
    pos = nx.spring_layout(g, seed=42, k=1.3)
    if alert in pos:                                       # centre the investigated account
        cx, cy = pos[alert]; pos = {n: (p[0] - cx, p[1] - cy) for n, p in pos.items()}
    net = Network(height="720px", width="100%", directed=True,
                  bgcolor="#10141a", font_color="#e8e8e8", cdn_resources="in_line")
    net.toggle_physics(False)
    for n, d in g.nodes(data=True):
        is_blocked = n in blocked
        net.add_node(n, label=("⛔" if is_blocked else "") + n[-5:],
                     x=float(pos[n][0] * 950), y=float(pos[n][1] * 950),
                     size=12 + 2.0 * g.degree(n),
                     color={"background": _hex(TOX, d["tox"]),
                            "border": "#ffffff" if is_blocked or n == alert else "#333"},
                     borderWidth=6 if is_blocked else (5 if n == alert else 1),
                     title=f"{n}\nrole: {d['role']}\ntoxicity: {d['tox']:.2f}"
                           f"\nfraud(gt): {d['fraud']}" + ("\nBLOCKED" if is_blocked else ""))
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, color=_hex(RISK, d["risk"]), width=1 + 5 * d["risk"],
                     title=f"amount: {d['amount']:.0f}\nrisk: {d['risk']:.2f}")
    return net, g

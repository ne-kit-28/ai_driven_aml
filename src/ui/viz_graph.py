"""
Ego-graph visualization for the compliance dashboard.

Renders the k-hop neighbourhood around an alert: nodes coloured by toxicity
(dropper probability), edges coloured/sized by transaction risk. Two sources:
  --source trino    : pull the ego-graph from Iceberg via Trino   (MVP / stand)
  --source parquet  : read local scored parquet                   (offline demo)

  python src/ui/viz_graph.py --source parquet --role collector --hops 1 --png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
from pyvis.network import Network

TOX = cm.get_cmap("RdYlGn_r")          # 0 green -> 1 red
RISK = mcolors.LinearSegmentedColormap.from_list("risk", ["#b0b0b0", "#d62728"])


# ---------------- data sources (same interface) ----------------
class ParquetSource:
    def __init__(self, data="data"):
        self.tx = pd.read_parquet(f"{data}/scored_transactions.parquet")
        self.acc = pd.read_parquet(f"{data}/scored_accounts.parquet").set_index("account_id")

    def incident_edges(self, accts):
        t = self.tx
        return t[t.source_account.isin(accts) | t.target_account.isin(accts)]

    def node_attrs(self, accts):
        return self.acc.reindex(list(accts))


class TrinoSource:
    """MVP path: pull the ego-graph from Iceberg via Trino (no graph DB)."""
    def __init__(self, host="localhost", port=8080, user="viz",
                 catalog="iceberg", schema="banking"):
        import trino  # lazy: only needed on the stand
        self.conn = trino.dbapi.connect(host=host, port=port, user=user,
                                        catalog=catalog, schema=schema)

    def _q(self, sql, params):
        cur = self.conn.cursor(); cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)

    def incident_edges(self, accts):
        accts = list(accts); ph = ",".join(["?"] * len(accts))
        sql = (f"SELECT tx_id, source_account, target_account, amount, ts, risk_score "
               f"FROM transactions WHERE source_account IN ({ph}) OR target_account IN ({ph})")
        return self._q(sql, accts + accts)

    def node_attrs(self, accts):
        accts = list(accts); ph = ",".join(["?"] * len(accts))
        sql = (f"SELECT account_id, fraud_role, is_fraud, toxicity "
               f"FROM accounts_state WHERE account_id IN ({ph})")
        return self._q(sql, accts).set_index("account_id").reindex(accts)


# ---------------- ego-graph BFS (shared) ----------------
def ego(source, alert, hops=1, max_nodes=60):
    nodes = {alert}; frontier = {alert}; edges = []
    for _ in range(hops):
        e = source.incident_edges(frontier)
        edges.append(e)
        new = (set(e.source_account) | set(e.target_account)) - nodes
        # keep the graph legible: cap how many new nodes we admit per hop
        if len(nodes) + len(new) > max_nodes:
            new = set(list(new)[: max_nodes - len(nodes)])
        nodes |= new; frontier = new
        if len(nodes) >= max_nodes or not new:
            break
    ed = pd.concat(edges).drop_duplicates("tx_id")
    ed = ed[ed.source_account.isin(nodes) & ed.target_account.isin(nodes)]
    return ed, source.node_attrs(set(ed.source_account) | set(ed.target_account) | {alert})


def hex_of(cmap, v):
    return mcolors.to_hex(cmap(float(max(0.0, min(1.0, v)))))


def render(edges, attrs, alert, out_html):
    g = nx.DiGraph()
    for a, row in attrs.iterrows():
        tox = float(row.get("toxicity", 0) or 0)
        g.add_node(a, tox=tox, role=str(row.get("fraud_role", "")),
                   fraud=int(row.get("is_fraud", 0) or 0))
    for _, e in edges.iterrows():
        g.add_edge(e.source_account, e.target_account, amount=float(e.amount),
                   risk=float(e.risk_score or 0))

    pos = nx.spring_layout(g, seed=42, k=1.2)            # deterministic layout
    if alert in pos:                                     # centre the investigated account
        cx, cy = pos[alert]; pos = {n: (p[0] - cx, p[1] - cy) for n, p in pos.items()}
    net = Network(height="760px", width="100%", directed=True,
                  bgcolor="#10141a", font_color="#e8e8e8", cdn_resources="in_line")
    net.toggle_physics(False)
    for n, d in g.nodes(data=True):
        deg = g.degree(n)
        net.add_node(n, label=f"{n[-5:]}", x=float(pos[n][0] * 900), y=float(pos[n][1] * 900),
                     size=12 + 2.2 * deg, color=hex_of(TOX, d["tox"]),
                     borderWidth=4 if n == alert else 1,
                     title=f"{n}\nrole: {d['role']}\ntoxicity: {d['tox']:.2f}\nfraud(gt): {d['fraud']}")
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, color=hex_of(RISK, d["risk"]),
                     width=1 + 4 * d["risk"], title=f"amount: {d['amount']:.0f}\nrisk: {d['risk']:.2f}")
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(out_html, notebook=False)
    _inject_overlay(out_html, alert)
    return g


def _inject_overlay(path, alert):
    """Add a title + legend overlay on top of the pyvis canvas (for the defense slide)."""
    overlay = f"""
    <div style="position:absolute;top:12px;left:16px;color:#e8e8e8;font-family:sans-serif;z-index:9;">
      <div style="font-size:20px;font-weight:700;">AML graph — ego-network of alert {alert[-5:]}</div>
      <div style="font-size:12px;opacity:.8;">GNN toxicity scoring · node = account, edge = transaction</div>
    </div>
    <div style="position:absolute;bottom:14px;left:16px;color:#e8e8e8;font-family:sans-serif;
                font-size:13px;background:#0008;padding:10px 14px;border-radius:8px;z-index:9;">
      <div><span style="color:#d62728;">●</span> high toxicity (likely dropper/mule)
           &nbsp;&nbsp;<span style="color:#1a9850;">●</span> low toxicity (legit)</div>
      <div>edge thickness / red = transaction risk &nbsp;·&nbsp; thick border = investigated account</div>
    </div>"""
    html = Path(path).read_text()
    Path(path).write_text(html.replace("<body>", "<body>" + overlay, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["parquet", "trino"], default="parquet")
    ap.add_argument("--data", default="data")
    ap.add_argument("--alert", help="account_id to centre on (else auto-pick by role)")
    ap.add_argument("--role", default="collector", help="auto-pick alert of this role")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", default="data/ego_graph.html")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()

    src = ParquetSource(args.data) if args.source == "parquet" else TrinoSource()
    if args.alert:
        alert = args.alert
    else:                                   # auto-pick most toxic account of the role
        a = src.acc if args.source == "parquet" else None
        cand = a[a.fraud_role == args.role] if a is not None else None
        alert = cand["toxicity"].idxmax() if cand is not None and len(cand) else a["toxicity"].idxmax()

    edges, attrs = ego(src, alert, hops=args.hops)
    g = render(edges, attrs, alert, args.out)
    print(f"alert={alert} | nodes={g.number_of_nodes()} edges={g.number_of_edges()} -> {args.out}")

    if args.png:
        from playwright.sync_api import sync_playwright
        png = args.out.replace(".html", ".png")
        with sync_playwright() as p:
            br = p.chromium.launch(); pg = br.new_page(viewport={"width": 1400, "height": 820})
            pg.goto(f"file://{Path(args.out).resolve()}"); pg.wait_for_timeout(1500)
            pg.screenshot(path=png, full_page=False); br.close()
        print("png ->", png)


if __name__ == "__main__":
    main()

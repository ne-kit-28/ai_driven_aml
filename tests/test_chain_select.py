"""Chain/cluster selection: block the laundering chain, not just one node."""
import pandas as pd

from src.ui.chain_select import select_chain


def attrs_df(rows):
    # rows: {account_id: (toxicity, fraud_role)}
    return pd.DataFrame(
        [{"account_id": a, "toxicity": t, "fraud_role": r} for a, (t, r) in rows.items()]
    ).set_index("account_id")


def edges_df(pairs):
    return pd.DataFrame(pairs, columns=["source_account", "target_account"])


def test_picks_up_connected_toxic_nodes():
    edges = edges_df([("A", "B"), ("B", "C")])
    attrs = attrs_df({"A": (0.9, "dropper"), "B": (0.8, "mule"), "C": (0.7, "collector")})
    assert select_chain(edges, attrs, "A", tox_threshold=0.5) == {"A", "B", "C"}


def test_seed_always_included_even_if_low_toxicity():
    edges = edges_df([("A", "B")])
    attrs = attrs_df({"A": (0.1, "legit"), "B": (0.9, "mule")})
    assert select_chain(edges, attrs, "A", tox_threshold=0.5) == {"A", "B"}


def test_legit_low_toxicity_neighbor_is_not_blocked():
    edges = edges_df([("A", "B"), ("B", "V")])  # V is a contaminated legit victim
    attrs = attrs_df({"A": (0.9, "mule"), "B": (0.8, "mule"), "V": (0.2, "legit")})
    assert select_chain(edges, attrs, "A", tox_threshold=0.5) == {"A", "B"}


def test_hubs_excluded_even_when_toxic():
    edges = edges_df([("A", "H")])  # cash-out via a shared legit hub
    attrs = attrs_df({"A": (0.9, "collector"), "H": (0.9, "hub")})
    assert select_chain(edges, attrs, "A", tox_threshold=0.5) == {"A"}


def test_disconnected_toxic_node_not_pulled_in():
    edges = edges_df([("A", "B")])
    attrs = attrs_df({"A": (0.9, "mule"), "B": (0.8, "mule"), "Z": (0.95, "mule")})
    assert "Z" not in select_chain(edges, attrs, "A", tox_threshold=0.5)


def test_isolated_seed_returns_just_itself():
    assert select_chain(edges_df([]), attrs_df({"A": (0.9, "mule")}), "A") == {"A"}


def test_threshold_filters_borderline_nodes():
    edges = edges_df([("A", "B")])
    attrs = attrs_df({"A": (0.9, "mule"), "B": (0.4, "mule")})
    assert select_chain(edges, attrs, "A", tox_threshold=0.5) == {"A"}

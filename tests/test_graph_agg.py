"""Parallel transactions between the same pair collapse into one weighted edge."""
import pandas as pd

from src.ui.graph_agg import aggregate_edges


def edges(rows):
    return pd.DataFrame(rows, columns=["source_account", "target_account", "amount", "risk_score"])


def test_merges_parallel_edges_with_count_sum_and_peak_risk():
    df = edges([
        ("A", "B", 100.0, 0.9),
        ("A", "B", 50.0, 0.4),
        ("A", "B", 25.0, 0.99),
    ])
    out = aggregate_edges(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_tx"] == 3
    assert row["amount"] == 175.0
    assert row["risk_score"] == 0.99   # peak, not last


def test_keeps_distinct_pairs_separate():
    df = edges([("A", "B", 10.0, 0.5), ("B", "C", 20.0, 0.6), ("A", "B", 5.0, 0.7)])
    out = aggregate_edges(df).set_index(["source_account", "target_account"])
    assert out.loc[("A", "B"), "n_tx"] == 2
    assert out.loc[("B", "C"), "n_tx"] == 1


def test_direction_matters():
    df = edges([("A", "B", 10.0, 0.5), ("B", "A", 20.0, 0.6)])
    assert len(aggregate_edges(df)) == 2


def test_null_risk_treated_as_zero():
    df = edges([("A", "B", 10.0, None), ("A", "B", 10.0, None)])
    assert aggregate_edges(df).iloc[0]["risk_score"] == 0.0


def test_empty_input_returns_empty_frame_with_schema():
    out = aggregate_edges(edges([]))
    assert len(out) == 0
    assert list(out.columns) == ["source_account", "target_account", "amount", "risk_score", "n_tx"]

"""The exactly-once guarantee: no transaction is ever scored twice.

These cover the anti-join that backs live scoring (``IcebergIO.poll`` /
``write_tx`` in ``infer_stream``), independent of Iceberg and torch.
"""
import pandas as pd

from src.ml.exactly_once import select_unscored, updated_seen


def batch(ids):
    return pd.DataFrame({"tx_id": ids, "amount": [1.0] * len(ids)})


def test_filters_already_scored_and_preserves_order():
    df = batch(["t3", "t1", "t2"])
    out = select_unscored(df, {"t1"})
    assert out["tx_id"].tolist() == ["t3", "t2"]


def test_all_seen_returns_empty():
    df = batch(["t1", "t2"])
    assert len(select_unscored(df, {"t1", "t2"})) == 0


def test_empty_input_is_safe():
    empty = pd.DataFrame(columns=["tx_id", "amount"])
    assert len(select_unscored(empty, {"t1"})) == 0


def test_updated_seen_unions_batch_ids():
    assert updated_seen({"t1"}, batch(["t2", "t3"])) == {"t1", "t2", "t3"}


def test_updated_seen_does_not_mutate_input():
    seen = {"t1"}
    updated_seen(seen, batch(["t2"]))
    assert seen == {"t1"}


def test_no_double_scoring_across_cycles():
    seen = set()
    # cycle 1: a fresh batch is fully unscored, then recorded
    c1 = batch(["t1", "t2"])
    fresh1 = select_unscored(c1, seen)
    assert fresh1["tx_id"].tolist() == ["t1", "t2"]
    seen = updated_seen(seen, fresh1)
    # cycle 2: ETL re-presents t1/t2 (overlapping commit) plus a new t3
    c2 = batch(["t1", "t2", "t3"])
    fresh2 = select_unscored(c2, seen)
    assert fresh2["tx_id"].tolist() == ["t3"]   # only the genuinely new row
    seen = updated_seen(seen, fresh2)
    # cycle 3: nothing new -> nothing scored
    assert len(select_unscored(c2, seen)) == 0

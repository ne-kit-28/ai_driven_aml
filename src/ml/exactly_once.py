"""Pure core of the exactly-once scoring guarantee.

The live scorer must never score a transaction twice, even across restarts or
overlapping ETL commits. The mechanism is an anti-join on ``tx_id`` against the
set of already-scored ids. That logic is extracted here so it can be unit-tested
without a live Iceberg catalog or torch; ``IcebergIO`` in ``infer_stream`` uses
these helpers.
"""
from __future__ import annotations

import pandas as pd


def select_unscored(df: pd.DataFrame, seen) -> pd.DataFrame:
    """Return the rows of ``df`` whose ``tx_id`` has not been scored yet.

    Order is preserved; an empty input returns an empty frame unchanged.
    """
    if not len(df):
        return df
    return df[~df["tx_id"].isin(set(seen))]


def updated_seen(seen, df) -> set:
    """Return ``seen`` extended with the ``tx_id`` values of a scored batch."""
    out = set(seen)
    if len(df):
        out.update(df["tx_id"].tolist())
    return out

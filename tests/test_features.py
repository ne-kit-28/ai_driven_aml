"""Reference-implementation feature builders, tested on a tiny hand-checked graph."""
import numpy as np
import pandas as pd
import pytest

from src.ml.features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_edge_features,
    build_node_features,
)


@pytest.fixture
def graph():
    accounts = pd.DataFrame(
        {"account_id": ["A", "B", "C"], "opened_days_ago": [10, 20, 30]}
    )
    tx = pd.DataFrame(
        {
            "source_account": ["A", "A", "B"],
            "target_account": ["B", "C", "A"],
            "amount": [100.0, 9200.0, 50.0],
        }
    )
    return accounts, tx


def col(name):
    return NODE_FEATURE_NAMES.index(name)


def test_node_matrix_shape_and_dtype(graph):
    mat, idx = build_node_features(*graph)
    assert mat.shape == (3, len(NODE_FEATURE_NAMES))
    assert mat.dtype == np.float32
    assert idx == {"A": 0, "B": 1, "C": 2}


def test_degrees_and_counterparties(graph):
    mat, idx = build_node_features(*graph)
    assert mat[idx["A"], col("out_degree")] == 2
    assert mat[idx["A"], col("in_degree")] == 1
    assert mat[idx["A"], col("distinct_out_cp")] == 2
    assert mat[idx["C"], col("out_degree")] == 0  # C only receives


def test_structuring_ratio_flags_threshold_band(graph):
    mat, idx = build_node_features(*graph)
    # A sends one normal (100) and one in-band (9200) transfer -> 0.5
    assert mat[idx["A"], col("structuring_ratio")] == pytest.approx(0.5)
    assert mat[idx["B"], col("structuring_ratio")] == pytest.approx(0.0)


def test_net_flow_sign(graph):
    mat, idx = build_node_features(*graph)
    # A pays out far more than it receives -> negative net flow
    assert mat[idx["A"], col("log_net_flow_abs")] < 0


def test_edge_features_band_and_log():
    tx = pd.DataFrame({"amount": [100.0, 9200.0]})
    feats = build_edge_features(tx, logamt_mean=0.0, logamt_std=1.0)
    assert feats.shape == (2, len(EDGE_FEATURE_NAMES))
    log_i, band_i, z_i = (EDGE_FEATURE_NAMES.index(n) for n in EDGE_FEATURE_NAMES)
    assert feats[0, band_i] == 0.0  # 100 is outside the band
    assert feats[1, band_i] == 1.0  # 9200 is inside [9000, 9500]
    assert feats[0, log_i] == pytest.approx(np.log1p(100.0), rel=1e-6)


def test_edge_zscore_uses_fixed_stats():
    tx = pd.DataFrame({"amount": [100.0, 200.0, 300.0]})
    mu, sd = 5.0, 2.0
    feats = build_edge_features(tx, logamt_mean=mu, logamt_std=sd)
    z = EDGE_FEATURE_NAMES.index("amount_zscore")
    expected = (np.log1p(tx["amount"].to_numpy()) - mu) / (sd + 1e-9)
    np.testing.assert_allclose(feats[:, z], expected, rtol=1e-6)

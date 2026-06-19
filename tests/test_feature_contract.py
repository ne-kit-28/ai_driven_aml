"""The feature contract must stay in lock-step with the trained artifact.

If these break, the model is being fed columns in an order it was not trained
on, or normalization stats no longer line up with the feature vector. Either
way the scores are silently wrong, so this guards the most dangerous drift.
"""
import json
from pathlib import Path

from src.ml.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES

META = json.loads(
    (Path(__file__).resolve().parents[1] / "src/ml/artifacts/tgnlite_meta.json").read_text()
)


def test_node_feature_names_match_artifact():
    assert NODE_FEATURE_NAMES == META["node_feature_names"]


def test_edge_feature_names_match_artifact():
    assert EDGE_FEATURE_NAMES == META["edge_feature_names"]


def test_node_normalization_stats_have_one_entry_per_feature():
    assert len(META["node_mean"]) == len(NODE_FEATURE_NAMES)
    assert len(META["node_std"]) == len(NODE_FEATURE_NAMES)


def test_feature_names_are_unique():
    assert len(set(NODE_FEATURE_NAMES)) == len(NODE_FEATURE_NAMES)
    assert len(set(EDGE_FEATURE_NAMES)) == len(EDGE_FEATURE_NAMES)

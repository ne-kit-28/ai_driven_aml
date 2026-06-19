"""The blocklist feedback loop: a blocked account leaves the live graph.

These cover the shared exclusion invariant used by both the producer (stops
emitting incident edges) and the ETL (marks them BLOCKED before scoring),
without Kafka or Spark.
"""
from src.ingest.blocklist import edge_blocked, filter_blocked_edges


def test_clean_edge_is_allowed():
    assert edge_blocked("A", "B", set()) is False


def test_blocked_source_excludes_edge():
    assert edge_blocked("A", "B", {"A"}) is True


def test_blocked_destination_excludes_edge():
    assert edge_blocked("A", "B", {"B"}) is True


def test_filter_drops_only_incident_edges_and_keeps_order():
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    # block C: both edges touching C must go, the rest stay in order
    assert filter_blocked_edges(edges, {"C"}) == [("A", "B"), ("D", "A")]


def test_filter_handles_extra_edge_payload():
    edges = [("A", "B", 100.0), ("B", "C", 9200.0)]
    assert filter_blocked_edges(edges, {"B"}) == []


def test_block_removes_both_directions():
    edges = [("X", "A"), ("A", "Y")]   # A as destination and as source
    assert filter_blocked_edges(edges, {"A"}) == []

"""Pure core of the blocklist feedback loop.

A blocked account is excluded from the live graph: any transaction touching it
(as source or destination) is dropped. The same invariant shows up in two
places — the producer stops emitting such edges, and the ETL marks them
``BLOCKED`` via a Spark SQL ``MERGE`` so they never reach scoring. The rule is
extracted here so it can be unit-tested without Kafka or Spark.
"""
from __future__ import annotations


def edge_blocked(src, dst, blocked) -> bool:
    """True if either endpoint of the edge is on the blocklist."""
    return src in blocked or dst in blocked


def filter_blocked_edges(edges, blocked):
    """Drop every ``(src, dst, ...)`` edge incident to a blocked account.

    Mirrors the ETL exclusion: blocked-account edges are removed from the set
    that reaches scoring. Order of the surviving edges is preserved.
    """
    return [e for e in edges if not edge_blocked(e[0], e[1], blocked)]

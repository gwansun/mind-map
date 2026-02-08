"""Importance score calculation for knowledge graph nodes."""

import math
import time


def calculate_importance_score(
    connection_count: int,
    max_connections: int,
    last_interaction: float,
    lambda_decay: float = 0.05,
    time_unit_days: float = 1.0,
) -> float:
    """Calculate importance score using the formula: S = (C_node / C_max) * e^(-lambda * delta_t).

    Args:
        connection_count: Number of connections this node has (C_node)
        max_connections: Maximum connections any node has (C_max)
        last_interaction: Unix timestamp of last interaction
        lambda_decay: Decay constant (higher = faster decay)
        time_unit_days: Time unit for delta_t calculation

    Returns:
        Importance score between 0 and 1
    """
    if max_connections == 0:
        return 0.0

    # Connectivity factor: C_node / C_max
    connectivity = connection_count / max_connections

    # Time decay: e^(-lambda * delta_t)
    delta_t = (time.time() - last_interaction) / (86400 * time_unit_days)  # Convert to days
    time_decay = math.exp(-lambda_decay * delta_t)

    return connectivity * time_decay


def rank_nodes_by_importance(
    nodes: list[dict],
    max_connections: int,
    lambda_decay: float = 0.05,
) -> list[tuple[dict, float]]:
    """Rank a list of nodes by their importance scores.

    Args:
        nodes: List of node dictionaries with 'connection_count' and 'last_interaction'
        max_connections: Maximum connections any node has in the graph
        lambda_decay: Decay constant for importance calculation

    Returns:
        List of (node, score) tuples sorted by score descending
    """
    scored_nodes = []
    for node in nodes:
        score = calculate_importance_score(
            connection_count=node.get("connection_count", 0),
            max_connections=max_connections,
            last_interaction=node.get("last_interaction", time.time()),
            lambda_decay=lambda_decay,
        )
        scored_nodes.append((node, score))

    return sorted(scored_nodes, key=lambda x: x[1], reverse=True)

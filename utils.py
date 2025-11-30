import math


def cosine_sim(v1, v2) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns:
        float in range [-1, 1], where:
        -  1.0  = same direction
        -  0.0  = orthogonal
        - -1.0  = opposite direction
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], constant: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (constant + rank)
    return fused

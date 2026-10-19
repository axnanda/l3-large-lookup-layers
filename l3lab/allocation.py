"""Embedding allocation: how many of the ``n_emb`` embeddings each vocabulary token owns.

Both functions return ``list[int]`` of length ``V`` with every entry in ``[1, k]`` and ``sum == target``.
"""
from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Sequence, Tuple


def _check_feasible(V: int, n_emb: int, k: int) -> None:
    if V < 1:
        raise ValueError("V must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if n_emb < V:
        raise ValueError(f"n_emb={n_emb} < V={V}: every token needs at least one embedding")
    if n_emb > V * k:
        raise ValueError(f"n_emb={n_emb} > V*k={V * k}: cannot place that many embeddings with at most k per token")


def allocate_by_frequency(counts: Sequence[int], n_emb: int, k: int, method: str = "proportional") -> List[int]:
    """Greedy frequency-based allocation (the same spirit as the paper's ``train_lzw`` loop).

    Every token starts with one embedding; the remaining ``n_emb - V`` are handed out one at a time to the
    token that is "most frequent" among those still below ``k``:

    * ``method="proportional"`` (default): the token maximising ``counts[v] / alloc[v]`` (D'Hondt
      apportionment), i.e. allocations roughly proportional to frequency, clipped to ``[1, k]``.
    * ``method="rank"``: literally the most frequent token still below ``k`` (so the top tokens are
      filled to ``k`` first, less frequent ones keep a single embedding).

    Ties are broken by lower token id, so the result is deterministic. Zero-count tokens get extra
    embeddings only after every token with nonzero count has reached ``k``.
    """
    counts = [float(c) for c in counts]
    V = len(counts)
    _check_feasible(V, n_emb, k)
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    alloc = [1] * V
    remaining = n_emb - V
    if remaining == 0:
        return alloc
    if method == "proportional":
        # max-heap on counts[v]/alloc[v]; python heapq is a min-heap so push negatives
        heap: List[Tuple[float, int]] = [(-counts[v] / alloc[v], v) for v in range(V) if k > 1]
    elif method == "rank":
        heap = [(-counts[v], v) for v in range(V) if k > 1]
    else:
        raise ValueError(f"unknown method {method!r}")
    heapq.heapify(heap)
    while remaining > 0:
        _, v = heapq.heappop(heap)
        alloc[v] += 1
        remaining -= 1
        if alloc[v] < k:
            key = -counts[v] / alloc[v] if method == "proportional" else -counts[v]
            heapq.heappush(heap, (key, v))
    assert sum(alloc) == n_emb and max(alloc) <= k and min(alloc) >= 1
    return alloc


def lzw_phrase_counts(token_sequences: Iterable[Sequence[int]], V: int) -> Dict[Tuple[int, ...], int]:
    """The paper's LZW phrase counter, generalised to arbitrary integer sequences.

    Dictionary starts with every single token (count 0). Scanning a sequence from position ``last``, the
    longest phrase ``toks[last:cur-1]`` already in the dictionary gets ``+1`` and the one-token-longer phrase
    is inserted with count 1; scanning resumes after the inserted phrase (this "skip" is what the paper's
    loop does; it is not textbook LZW but it is what the allocation was trained with).

    One edge case of the paper's loop is handled differently on purpose: when the end of a sequence is
    reached while still inside a known phrase, the paper's code resets that phrase's count to 1
    (``lzw_counter[phrase] = 1``); here it is incremented instead, and a trailing single token is counted
    (the paper's loop drops it).
    """
    counter: Dict[Tuple[int, ...], int] = {(s,): 0 for s in range(V)}
    for toks in token_sequences:
        toks = [int(t) for t in toks]
        n = len(toks)
        last = 0
        while last < n:
            cur = last + 1
            while cur < n and tuple(toks[last:cur]) in counter:
                cur += 1
            phrase = tuple(toks[last:cur])
            if phrase in counter:          # ran off the end of the sequence inside a known phrase
                counter[phrase] += 1
            else:                          # phrase[:-1] is the longest known prefix
                counter[phrase[:-1]] += 1
                counter[phrase] = 1
            last = cur
    return counter


def train_lzw_generic(token_sequences: Iterable[Sequence[int]], V: int, k: int, target: int) -> List[int]:
    """The paper's ``train_lzw`` allocation for arbitrary integer token sequences.

    Phrases are sorted by count (descending, ties by insertion order as in the paper); walking down that
    list, the *last* token of each phrase receives one more embedding if it still has fewer than ``k``,
    until ``target`` embeddings are allocated. If the phrase list is exhausted first (the paper's loop would
    raise ``IndexError``), the remainder is filled with ``allocate_by_frequency`` on single-token counts.
    """
    _check_feasible(V, target, k)
    counter = lzw_phrase_counts(token_sequences, V)
    ordered = sorted(counter.items(), key=lambda x: x[1], reverse=True)   # stable sort => paper's order
    alloc = [1] * V
    n_alloc = V
    i = 0
    while n_alloc < target and i < len(ordered):
        v = ordered[i][0][-1]
        if alloc[v] < k:
            alloc[v] += 1
            n_alloc += 1
        i += 1
    if n_alloc < target:
        single = [counter[(v,)] for v in range(V)]
        # continue greedily from the current allocation
        extra = allocate_by_frequency([s + 1.0 for s in single], target - n_alloc + V, k, method="rank")
        # `extra` starts every token at 1; merge while respecting k
        for v in range(V):
            add = extra[v] - 1
            take = min(add, k - alloc[v], target - n_alloc)
            alloc[v] += take
            n_alloc += take
        v = 0
        while n_alloc < target:   # anything left over (only if k binds): round-robin
            if alloc[v] < k:
                alloc[v] += 1
                n_alloc += 1
            v = (v + 1) % V
    assert sum(alloc) == target and 1 <= min(alloc) and max(alloc) <= k
    return alloc

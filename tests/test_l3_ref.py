"""L3Ref vs the paper's blocked forward, AllocTable/allocation invariants, state_dict round trips."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import l3_layer  # noqa: E402  (the user's paper port; torch.compile is off unless L3_COMPILE=1)
from l3lab.allocation import allocate_by_frequency, lzw_phrase_counts, train_lzw_generic  # noqa: E402
from l3lab.l3_ref import AllocTable, L3Ref, masked_softmax, paper_inputs  # noqa: E402

torch.set_num_threads(1)

V, N_EMB, K, H, D_EMB, D_UP, B, T = 20, 60, 5, 16, 8, 12, 4, 8


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _random_alloc(V=V, n_emb=N_EMB, k=K):
    counts = allocate_by_frequency(torch.randint(1, 100, (V,)).tolist(), n_emb, k)
    return AllocTable.from_counts(counts)


# ---------------------------------------------------------------- AllocTable
def test_alloc_table_invariants():
    counts = [1, 3, 2, 5, 1]
    a = AllocTable.from_counts(counts)
    assert a.V == 5 and a.n_emb == 12 and a.k == 5
    assert a.bounds.tolist() == [0, 1, 4, 6, 11, 12]
    assert torch.equal(a.counts, torch.tensor(counts))
    assert a.mask.sum(1).tolist() == counts
    for v in range(a.V):
        ids = a.table[v][a.mask[v]].tolist()
        assert ids == list(range(a.bounds[v], a.bounds[v + 1]))
        assert (a.table[v][~a.mask[v]] == 0).all()
    ea = a.emb_alloc()
    assert ea.shape == (12,) and ea.tolist() == [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4]
    wide = AllocTable.from_counts(counts, k=8)
    assert wide.k == 8 and wide.table.shape == (5, 8) and torch.equal(wide.table[:, :5], a.table)
    with pytest.raises(ValueError):
        AllocTable.from_counts([1, 0, 2])
    with pytest.raises(ValueError):
        AllocTable.from_counts(counts, k=4)


# ---------------------------------------------------------------- allocation
@pytest.mark.parametrize("method", ["proportional", "rank"])
def test_allocate_by_frequency_invariants(method):
    g = torch.Generator().manual_seed(1)
    for V_, n_emb, k in [(20, 60, 5), (50, 50, 3), (10, 40, 4), (7, 20, 8), (512, 4096, 8), (256, 2048, 16)]:
        counts = torch.randint(0, 1000, (V_,), generator=g).tolist()
        alloc = allocate_by_frequency(counts, n_emb, k, method=method)
        assert len(alloc) == V_ and sum(alloc) == n_emb and min(alloc) >= 1 and max(alloc) <= k
        assert alloc == allocate_by_frequency(counts, n_emb, k, method=method)   # deterministic
        order = sorted(range(V_), key=lambda v: -counts[v])
        # monotone: a more frequent token never has fewer embeddings than a less frequent one
        for a_, b_ in zip(order, order[1:]):
            if counts[a_] > counts[b_]:
                assert alloc[a_] >= alloc[b_]
    with pytest.raises(ValueError):
        allocate_by_frequency([1, 2, 3], 2, 4)      # fewer than V
    with pytest.raises(ValueError):
        allocate_by_frequency([1, 2, 3], 13, 4)     # more than V*k


def test_lzw_phrase_counts_small():
    c = lzw_phrase_counts([[0, 1, 0, 1, 0, 1]], V=2)
    assert c[(0,)] == 1 and c[(1,)] == 1 and c[(0, 1)] == 2 and c[(0, 1, 0)] == 1


def test_train_lzw_generic_invariants():
    g = torch.Generator().manual_seed(2)
    seqs = [torch.randint(0, 10, (int(torch.randint(1, 30, (1,), generator=g)),), generator=g).tolist() for _ in range(100)]
    for target, k in [(30, 5), (50, 5), (10, 3), (45, 5)]:
        a = train_lzw_generic(seqs, 10, k, target)
        assert sum(a) == target and min(a) >= 1 and max(a) <= k
    a = train_lzw_generic([[0, 1], [0, 1]], 10, 5, 45)   # phrase list exhausted -> frequency fallback
    assert sum(a) == 45 and max(a) <= 5


# ---------------------------------------------------------------- L3Ref == paper
def _paper_and_ref():
    paper = l3_layer.L3(H, N_EMB, D_EMB, D_UP)
    with torch.no_grad():                 # non-trivial norm gains so norm weights are exercised too
        paper.norm_in.weight.uniform_(0.5, 1.5)
        paper.norm_out.weight.uniform_(0.5, 1.5)
    return paper, L3Ref.load_from_paper_layer(paper)


@pytest.mark.parametrize("bb", [32, 16, 8])
def test_l3ref_matches_paper_forward(bb):
    alloc = _random_alloc()
    paper, ref = _paper_and_ref()
    x = torch.randn(B, T, H)
    seqs = torch.randint(0, V, (B, T))
    inputs = paper_inputs(seqs, alloc)
    out_paper = paper(x, *inputs[1:], bb=bb)
    out_ref = ref(x, seqs, alloc)
    assert out_ref.shape == (B, T, H)
    diff = (out_paper - out_ref).abs().max().item()
    assert diff < 1e-5, (
        f"L3Ref (per-token masked softmax over the token's allocated embeddings) differs from the paper's "
        f"blocked forward with bb={bb}: max abs diff {diff:.3e}")


def test_l3ref_matches_paper_gradients():
    alloc = _random_alloc()
    paper, ref = _paper_and_ref()
    x = torch.randn(B, T, H)
    seqs = torch.randint(0, V, (B, T))
    inputs = paper_inputs(seqs, alloc)
    gp = torch.autograd.grad(paper(x, *inputs[1:], bb=32).pow(2).sum(), list(paper.parameters()))
    gr = torch.autograd.grad(ref(x, seqs, alloc).pow(2).sum(), list(ref.parameters()))
    for (name, _), a, b in zip(paper.named_parameters(), gp, gr):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), f"gradient mismatch on {name}"


def test_l3ref_degenerate_batches():
    """all positions the same token, and every token distinct"""
    alloc = _random_alloc()
    paper, ref = _paper_and_ref()
    x = torch.randn(B, T, H)
    for seqs in (torch.full((B, T), 3), torch.randperm(V)[: B * T].reshape(B, T) if V >= B * T else None):
        if seqs is None:
            continue
        inputs = paper_inputs(seqs, alloc)
        assert torch.allclose(paper(x, *inputs[1:], bb=16), ref(x, seqs, alloc), atol=1e-5)


def test_attention_probs_masked():
    alloc = _random_alloc()
    ref = L3Ref(H, N_EMB, D_EMB, D_UP)
    seqs = torch.randint(0, V, (B, T))
    p, comb = ref.attention(torch.randn(B, T, H), seqs, alloc)
    assert p.shape == (B, T, alloc.k) and comb.shape == (B, T, D_EMB)
    assert torch.allclose(p.sum(-1), torch.ones(B, T), atol=1e-6)
    assert (p[~alloc.mask[seqs]] == 0).all()
    assert torch.isfinite(masked_softmax(torch.randn(3, 4), torch.tensor([[True, False, False, False]] * 3))).all()


def test_state_dict_round_trip():
    paper = l3_layer.L3(H, N_EMB, D_EMB, D_UP)
    ref = L3Ref(H, N_EMB, D_EMB, D_UP)
    assert set(paper.state_dict()) == set(ref.state_dict())
    ref.load_state_dict(paper.state_dict(), strict=True)
    for k_, v in paper.state_dict().items():
        assert torch.equal(v, ref.state_dict()[k_])
    paper2 = l3_layer.L3(H, N_EMB, D_EMB, D_UP)
    paper2.load_state_dict(ref.state_dict(), strict=True)
    alloc = _random_alloc()
    x, seqs = torch.randn(B, T, H), torch.randint(0, V, (B, T))
    inputs = paper_inputs(seqs, alloc)
    assert torch.allclose(paper2(x, *inputs[1:], bb=32), ref(x, seqs, alloc), atol=1e-5)

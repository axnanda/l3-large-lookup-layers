"""LIF/spike primitives, tokenizers, and forward/backward of every model with tiny sizes."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from l3lab.l3_ref import AllocTable  # noqa: E402
from l3lab.models import (L3CharLM, L3Classifier, MLP3, SNNL3Block, SNNL3Classifier, SNNMLP3, WW_ROLES,  # noqa: E402
                          check_ww_layers)
from l3lab.snn import LIF, rate, spike, spike_stats, surrogate_sigmoid  # noqa: E402
from l3lab.tokenizers import CharTokenizer, PatchVQTokenizer, patchify  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------- snn primitives
def test_spike_exact_binary_and_surrogate_grad():
    v = torch.linspace(-3, 3, 101, requires_grad=True)
    s = spike(v, k=4.0)
    assert set(s.detach().unique().tolist()) == {0.0, 1.0}
    assert torch.equal(s.detach(), (v.detach() > 0).float())
    s.sum().backward()
    sg = surrogate_sigmoid(4.0)(v.detach())
    expected = 4.0 * sg * (1 - sg)
    assert torch.isfinite(v.grad).all() and (v.grad > 0).all()
    assert torch.allclose(v.grad, expected, atol=1e-6)


def test_lif_dynamics_and_grads():
    beta, theta = 0.9, 1.0
    lif = LIF(beta=beta, theta=theta, k=4.0, reset="soft")
    I = torch.full((6, 2, 3), 0.5, requires_grad=True)
    s, u = lif(I)
    assert s.shape == u.shape == (6, 2, 3)
    assert set(s.detach().unique().tolist()) <= {0.0, 1.0}
    # hand-rolled reference recurrence
    u_ref, s_ref, us, ss = 0.0, 0.0, [], []
    for t in range(6):
        u_ref = beta * u_ref + 0.5 - theta * s_ref
        s_ref = 1.0 if u_ref - theta > 0 else 0.0
        us.append(u_ref)
        ss.append(s_ref)
    assert torch.allclose(u[:, 0, 0], torch.tensor(us), atol=1e-6)
    assert s[:, 0, 0].tolist() == ss and ss[2] == 1.0 and ss[0] == 0.0
    (s.sum() + 0.1 * u.sum()).backward()
    assert torch.isfinite(I.grad).all() and I.grad.abs().sum() > 0
    assert torch.allclose(rate(s), s.mean(0))
    st = spike_stats(s)
    assert 0 <= st["rate"] <= 1 and 0 <= st["frac_silent_units"] <= 1
    for reset in ("hard", "none"):
        s2, _ = LIF(reset=reset)(torch.randn(4, 3, 5))
        assert set(s2.unique().tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------- tokenizers
def test_patch_vq_tokenizer_shapes_and_determinism():
    rng = np.random.RandomState(0)
    imgs = rng.randint(0, 256, size=(120, 28, 28)).astype(np.uint8)
    assert patchify(imgs).shape == (120, 16, 49) and patchify(imgs).max() <= 1.0
    tok = PatchVQTokenizer(patch=7, n_codes=16, seed=3).fit(imgs)
    assert tok.V == 16 and tok.n_patches == 16
    codes = tok.encode(imgs)
    assert codes.shape == (120, 16) and codes.dtype == torch.long and 0 <= codes.min() and codes.max() < 16
    tok2 = PatchVQTokenizer(patch=7, n_codes=16, seed=3).fit(imgs)
    assert np.array_equal(tok.centroids, tok2.centroids) and torch.equal(codes, tok2.encode(imgs))
    counts = tok.counts(imgs)
    assert counts.shape == (16,) and counts.sum() == 120 * 16
    tok3 = PatchVQTokenizer.from_state_dict(tok.state_dict())
    assert torch.equal(tok3.encode(imgs[:5]), codes[:5])


def test_char_tokenizer_round_trip():
    ct = CharTokenizer().fit("hello world")
    assert ct.V == 8
    ids = ct.encode("hello")
    assert ids.dtype == torch.long and ct.decode(ids) == "hello"
    with pytest.raises(ValueError):
        ct.encode("xyz")


# ---------------------------------------------------------------- models
def _check_model(model, inp, n_out):
    out = model(inp)
    assert out.shape == (inp.shape[0], n_out) and torch.isfinite(out).all()
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    check_ww_layers(model)
    roles = {r for _, _, r in model.ww_layers()}
    assert roles <= set(WW_ROLES)
    return out


def test_mlp3_and_snnmlp3():
    x = torch.rand(6, 784)
    _check_model(MLP3(h=32, dropout=0.1), x, 10)
    m = SNNMLP3(h=32, T=4)
    _check_model(m, x, 10)
    assert set(m.last_stats) == {"rate_1", "silent_1", "rate_2", "silent_2"}
    assert {r for _, _, r in m.ww_layers()} == {"mlp", "head"}


def test_l3classifier_forward_backward_and_determinism():
    kw = dict(V=32, n_emb=100, k=6, h=32, d_emb=16, d_up=24, n_layers=2)
    ids = torch.randint(0, 32, (5, 16))
    torch.manual_seed(1)
    m1 = L3Classifier(**kw)
    out1 = _check_model(m1, ids, 10)
    torch.manual_seed(1)
    m2 = L3Classifier(**kw)
    assert torch.equal(out1, m2(ids))
    roles = [r for _, _, r in m1.ww_layers()]
    assert roles == ["embed", "embed"] + ["l3_wk", "l3_wv", "l3_wup", "l3_wmix"] * 2 + ["head"]
    ps = m1.lookup_probs(ids)
    assert len(ps) == 2 and ps[0].shape == (5, 16, m1.alloc.k) and torch.allclose(ps[0].sum(-1), torch.ones(5, 16))
    # allocation travels with the state_dict
    alloc = AllocTable.from_counts([2] * 20 + [5] * 12, k=6)
    m3 = L3Classifier(**kw, alloc=alloc)
    m4 = L3Classifier(**kw)
    m4.load_state_dict(m3.state_dict())
    assert torch.equal(m4.alloc.table, alloc.table) and torch.equal(m3(ids), m4(ids))
    with pytest.raises(ValueError):
        L3Classifier(**kw, alloc=AllocTable.from_counts([3] * 32))   # n_emb mismatch


@pytest.mark.parametrize("mode", ["rate_query", "spiking_memory"])
@pytest.mark.parametrize("residual", ["stream", "input"])
def test_snnl3classifier_modes(mode, residual):
    counts = [1, 2, 3, 4, 5, 6, 7, 8] * 4
    alloc = AllocTable.from_counts(counts)
    m = SNNL3Classifier(V=32, n_emb=sum(counts), k=8, h=32, d_emb=16, d_up=24, n_layers=2, T=4, beta=0.9,
                        theta=1.0, k_sg=4.0, mode=mode, residual=residual, alloc=alloc)
    ids = torch.randint(0, 32, (5, 16))
    out = _check_model(m, ids, 10)
    st = m.last_stats
    for key in ("rate_in", "silent_in", "rate_l0", "silent_l0", "rate_l1", "silent_l1", "rate_out"):
        assert key in st and 0.0 <= st[key] <= 1.0
    if mode == "spiking_memory":
        for key in ("mem_rate_l0", "mem_frac_entries_fired_l0", "mem_frac_emb_fired_l1", "mem_mean_active_l1"):
            assert key in st and st[key] >= 0.0
    else:
        assert "mem_rate_l0" not in st
    torch.manual_seed(0)
    # second forward (fresh graph) gives identical logits: no hidden state leaks between calls
    assert torch.equal(out.detach(), m(ids).detach())


def test_snnl3block_masked_memory_never_spikes():
    counts = [1, 3, 5]
    alloc = AllocTable.from_counts(counts)              # k = 5, padding entries in rows 0 and 1
    blk = SNNL3Block(h=8, n_emb=9, d_emb=4, d_up=6, mode="spiking_memory", theta=-0.5)   # theta<0: zero current would fire
    tok = torch.tensor([[0, 1, 2], [2, 0, 1]])
    ids, mask = alloc.table[tok], alloc.mask[tok]
    keys, vals = blk.gather(ids)
    s = (torch.rand(2, 3, 8) > 0.5).float()
    state = None
    for _ in range(3):
        cur, state, s_mem = blk.step(s, keys, vals, mask, state)
        assert set(s_mem.unique().tolist()) <= {0.0, 1.0}
        assert (s_mem[~mask] == 0).all()
        assert cur.shape == (2, 3, 8)
    cur.sum().backward()
    assert blk.w_k.weight.grad is not None and torch.isfinite(blk.w_k.weight.grad).all()


def test_snnl3_score_scale_parameter():
    m = SNNL3Classifier(V=16, n_emb=48, k=3, h=16, d_emb=8, d_up=8, n_layers=1, T=3, score_scale=0.25)
    assert m.score_scale == 0.25 and m.blocks[0].score_scale == 0.25
    with pytest.raises(ValueError):
        SNNL3Classifier(V=16, n_emb=48, k=3, h=16, d_emb=8, d_up=8, mode="bogus")


def test_l3charlm():
    m = L3CharLM(V=40, n_emb=120, k=4, h=32, d_emb=16, d_up=32, n_layers=2, context=16)
    ids = torch.randint(0, 40, (3, 10))
    out = m(ids)
    assert out.shape == (3, 10, 40) and torch.isfinite(out).all()
    out.sum().backward()
    check_ww_layers(m)
    with pytest.raises(ValueError):
        m(torch.randint(0, 40, (1, 17)))

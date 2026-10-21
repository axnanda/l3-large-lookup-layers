"""Tests for l3lab.multimodal (need the local CLIP weights; skipped otherwise)."""
import os, pytest, torch
from l3lab.multimodal import clipmodel as cm

HAVE = os.path.exists(cm.DEFAULT_WEIGHTS)
pytestmark = pytest.mark.skipif(not HAVE, reason="CLIP weights not available")


@pytest.fixture(scope="module")
def clip():
    return cm.load_clip()


def test_block_paths_match_open_clip(clip):
    model, pre, tok = clip
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = model.encode_image(x); mine = cm.image_embed_from_block(model, cm.embed_images(model, x), 0)
        t = tok(["a photo of a cat", "a dog"]); reft = model.encode_text(t); minet = cm.text_embed(model, t)
    assert torch.allclose(ref, mine, atol=1e-5) and torch.allclose(reft, minet, atol=1e-5)


def test_spiking_act_conversion_identity():
    """With gain = theta, soft reset, beta = 1 and large T, out -> clamp(I, 0, theta)."""
    act = cm.SpikingAct(5, T=256, theta=1.0, u0_frac=0.5)
    I = torch.tensor([[-0.5, 0.1, 0.5, 0.9, 1.5]])
    out = act(I)
    assert torch.allclose(out, torch.tensor([[0.0, 0.1, 0.5, 0.9, 1.0]]), atol=0.01)


def test_spiking_act_grad_flows_through_surrogate():
    act = cm.SpikingAct(3, T=8, theta=1.0)
    I = torch.tensor([[0.2, 0.6, 1.2]], requires_grad=True)
    act(I).sum().backward()
    assert I.grad is not None and torch.isfinite(I.grad).all() and I.grad.abs().sum() > 0


def test_adapter_zero_init_is_identity(clip):
    model, _, _ = clip
    blk = cm.get_block(model, "image", 3)
    x = torch.randn(2, 50, 768)
    with torch.no_grad():
        ref = blk(x); cm.install_adapter(model, "image", 3, r=16, spiking=True, T=4); out = blk(x)
    assert torch.allclose(ref, out)
    assert isinstance(blk.mlp, cm.MLPWithAdapter)


def test_clip_matrices_roles(clip):
    model, _, _ = clip
    mats, roles = cm.clip_matrices(model)
    assert len(mats) == 2 + 2 * 12 * 6 + 2
    assert roles["image.b00.mlp_fc1"] == "mlp_fc1" and mats["image.b00.mlp_fc1"].shape == (3072, 768)
    assert mats["text.token_embed"].shape == (49408, 512)

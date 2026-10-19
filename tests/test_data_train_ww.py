"""Agent B tests: data loaders, train_model, ww_analysis, sweep helpers and scripts (fast, < 2 min)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(1)

from l3lab import data as ldata  # noqa: E402
from l3lab import sweeps  # noqa: E402
from l3lab import train as ltrain  # noqa: E402
from l3lab import ww_analysis as wa  # noqa: E402
from standin_models import MLP3  # noqa: E402  (Agent B's stand-in for l3lab.models.MLP3)

FASHION_AVAILABLE = all((ldata.FASHION_DIR / f).exists() for f in ldata.FASHION_FILES.values())
needs_fashion = pytest.mark.skipif(not FASHION_AVAILABLE, reason="Fashion-MNIST files missing in data/fashion")


@pytest.fixture(scope="module")
def fashion():
    if not FASHION_AVAILABLE:
        pytest.skip("Fashion-MNIST files missing")
    return ldata.load_fashion_mnist()


@pytest.fixture(scope="module")
def small_train(fashion):
    xt, yt, xv, yv = ldata.make_split(fashion["x_train"], fashion["y_train"], n_val=10000, seed=0)
    X = ldata.to_tensor(ldata.flatten_images(xt[:512]))
    Y = ldata.to_tensor(yt[:512])
    XV = ldata.to_tensor(ldata.flatten_images(xv[:256]))
    YV = ldata.to_tensor(yv[:256])
    return X, Y, XV, YV


# ----------------------------------------------------------------------------- data
@needs_fashion
def test_fashion_loader_shapes_and_labels(fashion):
    assert fashion["x_train"].shape == (60000, 28, 28) and fashion["x_train"].dtype == np.uint8
    assert fashion["x_test"].shape == (10000, 28, 28)
    assert fashion["y_train"].shape == (60000,) and fashion["y_train"].dtype == np.int64
    assert fashion["y_test"].shape == (10000,)
    assert fashion["y_train"].min() == 0 and fashion["y_train"].max() == 9
    assert fashion["x_train"].max() == 255
    assert len(np.unique(fashion["y_test"])) == 10
    assert fashion["x_train"].flags.writeable


@needs_fashion
def test_make_split_disjoint_and_deterministic(fashion):
    xt, yt, xv, yv = ldata.make_split(fashion["x_train"], fashion["y_train"], n_val=10000, seed=3)
    xt2, yt2, xv2, yv2 = ldata.make_split(fashion["x_train"], fashion["y_train"], n_val=10000, seed=3)
    assert xt.shape == (50000, 28, 28) and xv.shape == (10000, 28, 28)
    assert len(yt) == 50000 and len(yv) == 10000
    assert np.array_equal(yv, yv2) and np.array_equal(xt[:10], xt2[:10])
    xt3, *_ = ldata.make_split(fashion["x_train"], fashion["y_train"], n_val=10000, seed=4)
    assert not np.array_equal(xt[:100], xt3[:100])


def test_patchify_shape_range_and_order():
    x = (np.arange(2 * 28 * 28) % 256).astype(np.uint8).reshape(2, 28, 28)
    p = ldata.patchify(x, patch=7)
    assert p.shape == (2, 16, 49) and p.dtype == np.float32
    assert p.min() >= 0.0 and p.max() <= 1.0
    # patch 5 = row 1, col 1 of the 4x4 grid
    expected = x[1, 7:14, 7:14].astype(np.float32).ravel() / 255.0
    assert np.allclose(p[1, 5], expected)
    flat = ldata.flatten_images(x)
    assert flat.shape == (2, 784) and flat.dtype == np.float32 and flat.max() <= 1.0


def test_load_tinyshakespeare():
    if not ldata.TINYSHAKESPEARE_PATH.exists():
        pytest.skip("tinyshakespeare.txt missing")
    text = ldata.load_tinyshakespeare()
    assert isinstance(text, str) and len(text) > 100000


# ------------------------------------------------------------------------- ww_analysis
def test_analyze_model_rows_extras_summary():
    torch.manual_seed(0)
    m = MLP3(h=64)
    rows, summary = wa.analyze_model(m, min_evals=50, randomize=True, mp_fit=True, tag="t")
    assert len(rows) == 3
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"fc1", "fc2", "fc3"}
    assert by_name["fc1"]["role"] == "mlp" and by_name["fc3"]["role"] == "head"
    for r in rows:
        for key in wa.WW_COLUMNS + wa.EXTRA_COLUMNS + ["role", "meets_min_evals"]:
            assert key in r, key
        if r["num_evals"] >= 50:
            assert r["alpha"] is not None and np.isfinite(r["alpha"]) and r["alpha"] > 0
            assert r["meets_min_evals"]
            for key in ("top_sv_energy", "effective_rank", "spectral_gap", "frac_dead_rows",
                        "mean_dir_overlap_left", "mean_dir_overlap_right"):
                assert r[key] is not None and np.isfinite(r[key]), key
            assert 0 < r["top_sv_energy"] <= 1 and 1 <= r["effective_rank"] <= r["M"] + 1e-6
    assert by_name["fc1"]["N"] == 784 and by_name["fc1"]["M"] == 64 and by_name["fc1"]["num_evals"] == 64
    assert by_name["fc3"]["num_evals"] == 10 and not by_name["fc3"]["meets_min_evals"]
    # summary
    assert summary["n_layers"] == 3 and summary["n_layers_min_evals"] == 2
    a50 = np.mean([by_name["fc1"]["alpha"], by_name["fc2"]["alpha"]])
    assert summary["alpha_min50_avg"] == pytest.approx(a50)
    assert summary["alpha_avg"] == pytest.approx(a50)
    if by_name["fc3"]["alpha"] is not None:
        assert summary["alpha_default_avg"] == pytest.approx(np.mean([r["alpha"] for r in rows]))
    assert summary["alpha_by_role"]["mlp"] == pytest.approx(a50)
    assert summary["alpha_role_mlp"] == pytest.approx(a50)
    for key in ("alpha_weighted_avg", "log_spectral_norm_avg", "n_over_trained", "n_under_trained",
                "frac_alpha_lt2", "frac_alpha_2_3", "frac_alpha_gt6", "time_sec"):
        assert key in summary
    assert summary["tag"] == "t"
    json.dumps(rows), json.dumps(summary)  # serialisable


def test_analyze_model_nested_mapping_and_embedding():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(60, 64)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(2)])
            self.head = nn.Linear(64, 10)

        def ww_layers(self):
            return [("embed", self.emb, "embed")] + [(f"blk{i}.lin", b[0], "mlp") for i, b in enumerate(self.blocks)] + \
                   [("head", self.head, "head")]

    torch.manual_seed(0)
    rows, summary = wa.analyze_model(Nested(), min_evals=50, randomize=False, mp_fit=False)
    names = [r["name"] for r in rows]
    assert names == ["embed", "blk0.lin", "blk1.lin", "head"]
    assert [r["role"] for r in rows] == ["embed", "mlp", "mlp", "head"]
    assert [r["module_path"] for r in rows] == ["emb", "blocks.0.0", "blocks.1.0", "head"]
    assert all(r["matched"] for r in rows)
    assert rows[0]["layer_type"] == "embedding" and rows[0]["N"] == 64 and rows[0]["M"] == 60
    assert summary["alpha_by_role"]["embed"] == pytest.approx(rows[0]["alpha"])
    assert summary["alpha_by_role"]["head"] is None  # 10 evals < min_evals
    assert summary["alpha_by_role_all"]["head"] == pytest.approx(rows[3]["alpha"])


def test_analyze_matrices():
    torch.manual_seed(0)
    mats = {"tab.a": torch.randn(300, 64), "b": torch.randn(80, 32)}
    rows, summary = wa.analyze_matrices(mats, roles={"tab.a": "embed"}, randomize=False, mp_fit=False)
    assert [r["name"] for r in rows] == ["tab.a", "b"]
    assert [r["role"] for r in rows] == ["embed", "matrix"]
    assert rows[0]["N"] == 300 and rows[0]["M"] == 64 and rows[0]["alpha"] is not None
    assert rows[0]["n_rows"] == 300 and rows[0]["n_cols"] == 64
    assert summary["n_layers"] == 2


def test_compute_extras_known_matrices():
    W = torch.ones(20, 30)  # rank 1, top direction = all-ones on both sides
    e = wa.compute_extras(W)
    assert e["top_sv_energy"] == pytest.approx(1.0)
    assert e["effective_rank"] == pytest.approx(1.0, abs=1e-4)
    assert e["ones_overlap_left"] == pytest.approx(1.0, abs=1e-5)
    assert e["mean_dir_overlap_right"] == pytest.approx(1.0, abs=1e-5)
    assert e["mean_dir_overlap_left"] == pytest.approx(1.0, abs=1e-5)
    assert e["frac_dead_rows"] == 0.0
    torch.manual_seed(0)
    W = torch.randn(40, 40)
    W[:4] = 0.0
    e = wa.compute_extras(W)
    assert e["frac_dead_rows"] == pytest.approx(0.1)
    assert e["spectral_gap"] >= 1.0 and 1.0 < e["effective_rank"] <= 40
    assert 0.0 <= e["mean_dir_overlap_left"] <= 1.0 + 1e-6


# ------------------------------------------------------------------------------- train
@needs_fashion
def test_train_model_one_epoch_with_ww(small_train, tmp_path):
    X, Y, XV, YV = small_train
    torch.manual_seed(0)
    model = MLP3(h=64)
    hist = ltrain.train_model(model, X, Y, XV, YV, epochs=1, batch_size=64, lr=1e-3, weight_decay=0.0,
                              optimizer="adam", momentum=0.9, seed=0, log_every_epoch=False, ww_every=1,
                              ww_kwargs=dict(min_evals=50, randomize=False, mp_fit=False), out_dir=tmp_path,
                              test_x=XV[:100], test_y=YV[:100], ww_at_init=True)
    assert hist["epoch"] == [1] and len(hist["train_loss"]) == 1 and len(hist["val_acc"]) == 1
    assert 0.0 <= hist["val_acc"][0] <= 1.0 and 0.0 <= hist["train_acc"][0] <= 1.0
    assert len(hist["test_acc"]) == 1
    assert [e["epoch"] for e in hist["ww"]] == [0, 1]
    rows = hist["ww"][-1]["rows"]
    assert len(rows) == 3 and all("alpha" in r for r in rows)
    fc1 = next(r for r in rows if r["name"] == "fc1")
    assert fc1["num_evals"] == 64 and fc1["alpha"] is not None and np.isfinite(fc1["alpha"])
    assert hist["ww"][-1]["summary"]["alpha_avg"] is not None
    assert hist["final"]["epochs_run"] == 1 and "gap" in hist["final"] and "test_acc" in hist["final"]
    assert hist["n_params"] == sum(p.numel() for p in model.parameters())
    json.dumps(hist)
    assert (tmp_path / "final.pt").exists() and (tmp_path / "history.json").exists()
    sd = torch.load(tmp_path / "final.pt")
    MLP3(h=64).load_state_dict(sd)
    loss, acc = ltrain.evaluate(model, XV, YV, batch_size=128)
    assert acc == pytest.approx(hist["final"]["val_acc"]) and np.isfinite(loss)


@needs_fashion
def test_train_model_sgd_no_ww_and_stop_fn(small_train):
    X, Y, XV, YV = small_train
    torch.manual_seed(1)
    model = MLP3(h=32)
    calls = []
    hist = ltrain.train_model(model, X, Y, XV, YV, epochs=3, batch_size=128, lr=5e-2, optimizer="sgd",
                              momentum=0.9, seed=1, log_every_epoch=False, ww_every=0, ww_final=False,
                              stop_fn=lambda: calls.append(1) or len(calls) >= 2)
    assert hist["ww"] == [] and hist["stopped_early"] and hist["final"]["epochs_run"] == 2
    assert hist["train_loss"][1] < hist["train_loss"][0] + 0.5


def test_train_model_sequence_logits():
    """Char-LM style model: logits (b,t,V) with targets (b,t) go through the same loop."""
    class TinyLM(nn.Module):
        def __init__(self, V=16, h=24):
            super().__init__()
            self.emb = nn.Embedding(V, h)
            self.lin = nn.Linear(h, V)

        def forward(self, ids):
            return self.lin(torch.relu(self.emb(ids)))

    torch.manual_seed(0)
    ids = torch.randint(0, 16, (64, 8))
    tgt = (ids + 1) % 16
    hist = ltrain.train_model(TinyLM(), ids, tgt, ids[:16], tgt[:16], epochs=2, batch_size=16, lr=1e-2,
                              log_every_epoch=False, ww_every=0, ww_final=False)
    assert len(hist["val_acc"]) == 2 and 0 <= hist["val_acc"][-1] <= 1


def test_to_jsonable():
    out = ltrain.to_jsonable({"a": np.float32(1.5), "b": torch.tensor(2), "c": np.array([1, 2]), "d": float("nan"),
                              "e": (np.int64(3),), "f": torch.tensor([1.0, 2.0])})
    assert out == {"a": 1.5, "b": 2, "c": [1, 2], "d": None, "e": [3], "f": [1.0, 2.0]}
    json.dumps(out)


# ----------------------------------------------------------------------------- sweeps
def test_sweep_helpers(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 20)
    yn, mask = sweeps.corrupt_labels(y, 0.3, rng, n_classes=10)
    assert mask.sum() == 60 and (yn[mask] != y[mask]).all() and (yn[~mask] == y[~mask]).all()
    assert yn.min() >= 0 and yn.max() <= 9
    y0, m0 = sweeps.corrupt_labels(y, 0.0, rng)
    assert np.array_equal(y0, y) and not m0.any()
    idx = sweeps.subsample_indices(1000, 100, rng)
    assert len(idx) == 100 and len(np.unique(idx)) == 100 and np.all(np.diff(idx) > 0)
    assert len(sweeps.subsample_indices(50, 100, rng)) == 50
    torch.manual_seed(0)
    m = MLP3(h=16)
    w0 = m.fc1.weight.detach().clone()
    sweeps.apply_init_gain(m, 2.0)
    assert torch.allclose(m.fc1.weight, 2 * w0)
    store = sweeps.RunStore(tmp_path / "sweep")
    rid = sweeps.run_id_for(0, 3)
    assert rid == "run_s0_0003" and not store.exists(rid)
    store.save(rid, {"run_id": rid, "x": np.float32(1.0)})
    assert store.exists(rid) and store.load(rid)["x"] == 1.0 and len(store.load_all()) == 1
    b = sweeps.Budget(0)
    assert not b.exhausted(1e9)
    b = sweeps.Budget(1e-6)
    assert b.exhausted(next_estimate_sec=1.0)  # 1 s estimate exceeds a 60 us budget
    mod = sweeps.import_models(str(ROOT / "tests" / "standin_models.py"))
    assert mod.MLP3 is not None
    assert sweeps.construct(MLP3, h=8, bogus_kwarg=1).fc1.out_features == 8
    arch = sweeps.sample_l3_arch(np.random.default_rng(1))
    assert arch["V"] <= arch["n_emb"] <= arch["V"] * arch["k"]


def _load_script(name):
    return sweeps.load_module_from_path(ROOT / "scripts" / name)


def test_sweep_scripts_import_and_argparse(tmp_path):
    for name in ("run_mlp_sweep.py", "run_l3_sweep.py", "run_snn_sweep.py"):
        mod = _load_script(name)
        args = mod.parse_args(["--smoke", "--out", str(tmp_path / name), "--seed", "5", "--max_minutes", "1"])
        assert args.smoke and args.seed == 5 and args.max_minutes == 1.0
        rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
        cfgs1 = [mod.sample_config(rng1, i, args) for i in range(4)]
        cfgs2 = [mod.sample_config(rng2, i, args) for i in range(4)]
        assert cfgs1 == cfgs2  # deterministic config sampling
        assert all(c["epochs"] == 1 for c in cfgs1)  # smoke
    l3 = _load_script("run_l3_sweep.py")
    args = l3.parse_args(["--out", str(tmp_path / "l3"), "--epochs_max", "7"])
    rng = np.random.default_rng(0)
    cfgs = [l3.sample_config(rng, i, args) for i in range(20)]
    for c in cfgs:
        assert c["V"] <= c["n_emb"] <= c["V"] * c["k"] and 1 <= c["epochs"] <= 7
        assert c["h"] in (128, 256) and c["n_layers"] in (1, 2, 3)
    snn = _load_script("run_snn_sweep.py")
    args = snn.parse_args(["--out", str(tmp_path / "snn"), "--model_type", "both"])
    rng = np.random.default_rng(0)
    cfgs = [snn.sample_config(rng, i, args) for i in range(30)]
    assert {c["model"] for c in cfgs} == {"SNNMLP3", "SNNL3Classifier"}
    assert all(c["T"] in (4, 8, 16) and c["k_sg"] in (1, 2, 4, 8) and c["beta"] in (0.5, 0.9) for c in cfgs)
    assert all(c["mode"] in ("rate_query", "spiking_memory") for c in cfgs if c["model"] == "SNNL3Classifier")
    an = _load_script("analyze_sweeps.py")
    assert an.parse_args(["--in", "x", "--out", "y"]).out == "y"


@needs_fashion
def test_mlp_sweep_smoke_and_analyze(tmp_path):
    mod = _load_script("run_mlp_sweep.py")
    out = tmp_path / "smoke"
    rc = mod.main(["--smoke", "--n_runs", "1", "--out", str(out), "--model_module", str(ROOT / "tests" / "standin_models.py"),
                   "--no_randomize", "--no_mp_fit", "--quiet_epochs"])
    assert rc == 0
    files = sorted((out / "runs").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["model"] == "MLP3" and payload["config"]["epochs"] == 1 and payload["config"]["train_subset"] == 512
    assert payload["ww_rows"] and payload["ww_summary"]["alpha_avg"] is not None
    assert "val_acc" in payload["final"] and "train_acc_clean" in payload["final"]
    # resumable: second call skips the existing run
    rc = mod.main(["--smoke", "--n_runs", "1", "--out", str(out), "--model_module", str(ROOT / "tests" / "standin_models.py")])
    assert rc == 0 and len(sorted((out / "runs").glob("*.json"))) == 1
    an = _load_script("analyze_sweeps.py")
    rc = an.main(["--in", str(out), "--out", str(tmp_path / "figs"), "--min_runs_corr", "1"])
    assert rc == 0
    assert (tmp_path / "figs" / "summary.md").exists() and (tmp_path / "figs" / "runs_table.csv").exists()
    assert (tmp_path / "figs" / "alpha_hist_by_role.png").exists()


def test_real_models_mlp3_if_available():
    """Integration check against Agent A's MLP3 (skipped until l3lab/models.py exists)."""
    try:
        from l3lab.models import MLP3 as RealMLP3  # noqa: WPS433
    except ImportError:
        pytest.skip("l3lab.models not available yet")
    torch.manual_seed(0)
    m = RealMLP3(h=64)
    layers = m.ww_layers()
    assert all(len(t) == 3 and isinstance(t[1], nn.Module) for t in layers)
    rows, summary = wa.analyze_model(m, min_evals=50, randomize=False, mp_fit=False)
    assert rows and all(r["matched"] for r in rows) and summary["alpha_avg"] is not None
    out = m(torch.rand(4, 784))
    assert out.shape == (4, 10)

"""The measurement layer must preserve outputs and keep evaluation out of training."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_costs", ROOT / "scripts/benchmark_costs.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def load_instrumented(module, meter):
    path = ROOT / "src" / (module.replace(".", "/") + ".py")
    transform = bench.Instrument(module)
    tree = transform.visit(ast.parse(path.read_text()))
    namespace = {"__name__": module, "__package__": module.rpartition(".")[0], "_cost_meter": meter}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace, transform.counts


def test_probe_instrumentation_preserves_training_and_counts_batches():
    from merge_and_rebase.eval.linear_probe import train_zeroshot_head_probe
    from merge_and_rebase.models.openclip_classifier import OpenClipClassifier

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)

        def encode_image(self, images):
            return self.linear(images)

    torch.manual_seed(5)
    clf = OpenClipClassifier(Model(), None, None)
    clf._zs_text_features = torch.randn(2, 4)
    loader = DataLoader(TensorDataset(torch.randn(7, 3), torch.tensor([0, 1, 0, 1, 0, 1, 0])), batch_size=3)
    kwargs = dict(device="cpu", steps=2, eval_loaders={"val": loader}, log_every=1)
    reference = train_zeroshot_head_probe(clf, loader, **kwargs)
    meter = bench.Meter(torch, warmup=0)
    namespace, counts = load_instrumented("merge_and_rebase.eval.linear_probe", meter)
    with meter.phase("pipeline"):
        actual = namespace["train_zeroshot_head_probe"](clf, loader, **kwargs)
    result = meter.finish()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert result["phases"]["training"]["calls"] == 2
    assert result["phases"]["evaluation"]["calls"] == 3
    assert result["batches"]["train/batch_size=3"]["images_processed"] == 12
    assert result["batches"]["train/batch_size=1"]["images_processed"] == 2
    assert counts["phase:training"] == 1
    assert result["phases"]["probe_pipeline"]["exclusive_seconds"] >= 0


def test_real_modules_still_expose_required_measurement_points():
    # Fail if a repository refactor silently removes KD/extraction instrumentation.
    for module, required in [
        (
            "merge_and_rebase.eval.kd_rebase",
            {"phase:training", "batch:train", "phase:teacher_precompute", "batch:teacher_forward"},
        ),
        (
            "merge_and_rebase.rebase.methods.steer",
            {"phase:extraction", "batch:extraction", "phase:refract_prepare", "phase:stage2_fit"},
        ),
    ]:
        path = ROOT / "src" / (module.replace(".", "/") + ".py")
        instrument = bench.Instrument(module)
        tree = instrument.visit(ast.parse(path.read_text()))
        compile(ast.fix_missing_locations(tree), str(path), "exec")
        assert required <= instrument.counts.keys()


def test_nested_gpu_resets_retain_parent_and_global_peaks():
    class Cuda:
        allocated = reserved = 0

        def is_available(self):
            return True

        def synchronize(self):
            pass

        def max_memory_allocated(self):
            return self.allocated

        def max_memory_reserved(self):
            return self.reserved

        def reset_peak_memory_stats(self):
            self.allocated = self.reserved = 0

    cuda = Cuda()
    meter = bench.Meter(SimpleNamespace(cuda=cuda), device="cuda:0")
    with meter.phase("pipeline"):
        cuda.allocated, cuda.reserved = 10, 20
        with meter.phase("child"):
            cuda.allocated, cuda.reserved = 100, 200
        cuda.allocated, cuda.reserved = 30, 60
    result = meter.finish()
    assert result["peak_allocated_bytes"] == 100
    assert result["peak_reserved_bytes"] == 200
    assert result["phases"]["pipeline"]["peak_allocated_bytes"] == 100
    assert result["phases"]["child"]["peak_allocated_bytes"] == 100


def test_matched_config_and_isolated_cache(tmp_path):
    args = bench.parser().parse_args([])
    base = {
        "source_clip_model": "A",
        "target_clip_model": "B",
        "target_clip_pretrained": "weights",
        "alpha_search": True,
        "linear_probe_head": True,
        "method_params": {"feature_cache_dir": "/production"},
    }
    lp = bench.make_config(base, "linear_probe", "SVHN", 33, args, tmp_path, tmp_path / "isolated")
    kd = bench.make_config(base, "kd", "SVHN", 33, args, tmp_path, tmp_path / "isolated")
    assert lp["seed"] == kd["kd"]["seed"] == kd["method_params"]["seed"] == 33
    assert lp["linear_probe_shots_per_class"] == kd["kd"]["few_shot"]
    assert lp["batch_size"] == kd["kd"]["batch_size"]
    assert not lp["linear_probe_head"] and not kd["alpha_search"]
    assert kd["method_params"]["feature_cache_dir"] == str(tmp_path / "isolated")
    assert base["method_params"]["feature_cache_dir"] == "/production"


def test_empty_steady_state_is_not_reported_as_zero():
    assert bench.stats([]) == {"n": 0}
    assert bench.stats([1, 2, 3])["mean"] == pytest.approx(2)


def test_kd_instrumented_pipeline_precaches_teacher_once(monkeypatch, tmp_path):
    """Execute the real KD loop on tiny CPU models, without downloads."""
    import json
    from copy import deepcopy

    from merge_and_rebase.data import vision_loaders
    from merge_and_rebase.eval import utils
    from merge_and_rebase.models.openclip_classifier import OpenClipClassifier

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(3, 4)

        def encode_image(self, images):
            return self.visual(images)

    class Dataset(TensorDataset):
        def __init__(self):
            super().__init__(torch.randn(8, 3), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
            self.split = {"label": self.tensors[1].tolist()}
            self.label_key = "label"

        def _map_label(self, value):
            return value

    torch.manual_seed(6)
    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=2)
    loaders = SimpleNamespace(train=loader, val=loader, test=loader, classnames=["a", "b"])

    def build(cfg):
        torch.manual_seed(10 if cfg.model_name == "source" else 11)
        clf = OpenClipClassifier(Model(), None, None)
        clf._zs_text_features = torch.eye(2, 4)
        return clf

    monkeypatch.setattr(OpenClipClassifier, "build", staticmethod(build))
    monkeypatch.setattr(OpenClipClassifier, "build_zeroshot_text_features", lambda *a, **kw: None)
    monkeypatch.setattr(vision_loaders, "load_hf_splits", lambda *a, **kw: None)
    monkeypatch.setattr(vision_loaders, "build_vision_loaders", lambda **kw: loaders)
    monkeypatch.setattr(utils, "eval_task_top1", lambda **kw: 0.5)
    cfg = {
        "source_clip_model": "source",
        "source_clip_pretrained": "stub",
        "target_clip_model": "target",
        "target_clip_pretrained": "stub",
        "device": "cpu",
        "tasks": "Cars",
        "batch_size": 2,
        "num_workers": 0,
        "tuned_ckpts": {"Cars": str(tmp_path / "teacher.pt")},
        "kd": {"few_shot": 2, "epochs": 3, "batch_size": 2, "seed": 33},
        "logging": {"use_wandb": False, "local_log_dir": str(tmp_path / "logs")},
    }
    torch.save({}, tmp_path / "teacher.pt")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    meter = bench.Meter(torch, warmup=0)
    namespace, _ = load_instrumented("merge_and_rebase.eval.kd_rebase", meter)
    namespace["_load_finetuned_source_vision"] = lambda source, *a: deepcopy(source)
    with meter.phase("pipeline"):
        namespace["main"](["vision", "--config", str(config_path)])
    result = meter.finish()
    assert result["phases"]["training"]["calls"] == 3
    assert result["phases"]["teacher_precompute"]["calls"] == 1
    assert result["batches"]["teacher_forward/batch_size=2"]["calls"] == 2
    assert result["batches"]["train/batch_size=2"]["calls"] == 6


def test_standard_residual_extraction_is_numerically_unchanged(monkeypatch):
    from merge_and_rebase.rebase.methods.steer import _collect_standard_split

    torch.manual_seed(7)
    source_pre = torch.nn.Linear(3, 4)
    source_ft = torch.nn.Linear(3, 4)
    target = torch.nn.Linear(3, 5)
    loader = DataLoader(TensorDataset(torch.randn(5, 3), torch.arange(5)), batch_size=2)
    kwargs = dict(
        clf_source_finetuned_visual=source_ft,
        clf_source_pretrained_visual=source_pre,
        target_visual=target,
        source_loader=loader,
        target_loader=loader,
        device=torch.device("cpu"),
    )
    reference = _collect_standard_split(**kwargs)
    meter = bench.Meter(torch, warmup=0)
    from merge_and_rebase.rebase import registry

    monkeypatch.setattr(registry, "register", lambda method: None)
    namespace, _ = load_instrumented("merge_and_rebase.rebase.methods.steer", meter)
    with meter.phase("pipeline"):
        actual = namespace["_collect_standard_split"](**kwargs)
    result = meter.finish()
    for key in reference:
        torch.testing.assert_close(reference[key], actual[key], rtol=0, atol=0)
    assert result["phases"]["extraction"]["calls"] == 1
    assert result["batches"]["extraction/batch_size=2"]["images_processed"] == 4
    assert result["batches"]["source_forward_pretrained/batch_size=2"]["calls"] == 2


def test_linear_residual_jvp_instrumentation_preserves_results(monkeypatch):
    from merge_and_rebase.rebase import registry
    from merge_and_rebase.rebase.methods.steer import _collect_linear_split

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
            self.proj = torch.nn.Parameter(torch.randn(3, 4))

        def forward(self, x):
            for block in self.transformer.resblocks:
                x = x + block(x)
            return x @ self.proj

    torch.manual_seed(12)
    source, target = Visual(), Visual()
    loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.arange(4)), batch_size=2)
    kwargs = dict(
        source_pretrained_visual=source,
        source_finetuned_params={n: p.detach().clone() + 0.01 for n, p in source.named_parameters()},
        target_visual=target,
        source_loader=loader,
        target_loader=loader,
        device=torch.device("cpu"),
    )
    reference = _collect_linear_split(**kwargs)
    meter = bench.Meter(torch, warmup=0)
    monkeypatch.setattr(registry, "register", lambda method: None)
    namespace, _ = load_instrumented("merge_and_rebase.rebase.methods.steer", meter)
    with meter.phase("pipeline"):
        actual = namespace["_collect_linear_split"](**kwargs)
    result = meter.finish()
    torch.testing.assert_close(reference, actual, rtol=0, atol=0)
    assert result["batches"]["source_jvp_full/batch_size=2"]["calls"] == 2
    assert result["batches"]["source_jvp_block/batch_size=2"]["calls"] == 4

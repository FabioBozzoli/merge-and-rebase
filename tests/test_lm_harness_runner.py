from __future__ import annotations

import sys
from types import ModuleType

import pytest


def test_harness_import_without_lm_eval() -> None:
    from merge_and_rebase.eval.lm_harness_runner import run

    assert callable(run)


def test_harness_raises_without_lm_eval(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lm_eval", None)

    from merge_and_rebase.eval.lm_harness_runner import run

    with pytest.raises(ImportError, match="lm-eval"):
        run(tasks=["hellaswag"], model=None, tokenizer=None)


def test_harness_with_mocked_lm_eval(monkeypatch) -> None:
    class _FakeResults:
        def get(self, key, default=None):
            if key == "results":
                return {
                    "hellaswag": {"acc,none": 0.42},
                    "piqa": {"acc_norm,none": 0.75},
                }
            return default

    class _FakeLMEval:
        def simple_evaluate(self, **kwargs):
            return _FakeResults()

    import torch.nn as nn

    fake_mod = ModuleType("lm_eval")
    fake_mod.simple_evaluate = lambda **kwargs: _FakeResults()
    monkeypatch.setitem(sys.modules, "lm_eval", fake_mod)

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    result = run(tasks=["hellaswag", "piqa"], model=stub_model, tokenizer=None)
    assert "hellaswag" in result
    assert result["hellaswag"] == 0.42
    assert "piqa_norm" in result
    assert result["piqa_norm"] == 0.75

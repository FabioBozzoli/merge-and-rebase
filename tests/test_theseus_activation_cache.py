import torch
import torch.nn as nn

from merge_and_rebase.rebase.methods.theseus import TheseusRebase


def test_theseus_reuses_activation_cache(tmp_path, monkeypatch) -> None:
    source = nn.Linear(2, 2, bias=False)
    target = nn.Linear(2, 2, bias=False)
    source.register_parameter("scalar", nn.Parameter(torch.tensor(1.0)))
    batches = [(torch.ones(2, 2),)]
    method = TheseusRebase()
    calls = 0

    from merge_and_rebase.rebase.methods import theseus

    original_collect = theseus.collect_activations

    def counted_collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(theseus, "collect_activations", counted_collect)
    kwargs = dict(
        source_model=source,
        target_model=target,
        source_dataloader=batches,
        target_dataloader=batches,
        device="cpu",
        n_batches=1,
        patch_qkv=False,
        verbose=False,
        activation_cache_dir=str(tmp_path),
        activation_cache_mode="auto",
    )
    first = method.prepare(**kwargs)
    second = method.prepare(**kwargs)

    assert calls == 1
    assert first["activation_registry"].keys() == second["activation_registry"].keys()
    assert len(list(tmp_path.glob("theseus_activations_*.pt"))) == 1

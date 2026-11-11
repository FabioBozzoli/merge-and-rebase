from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaForSequenceClassification, T5Config

from merge_and_rebase.rebase.text.steer_text import _collect_linear_split, _head_as_identity
from scripts.benchmark_text_costs import (
    ResidualExtractor,
    collect,
    fit_all,
    load_encoder,
    nearest_mean_head,
    save_results,
)
from scripts.dispatch_text_costs import dependency_jobs
from scripts.eval_t5_encoder_checkpoints import T5EncoderForSequenceClassification


def test_dispatch_waits_for_entire_retry_arrays_and_excludes_itself():
    assert dependency_jobs(["123_2", "123_3", "456", "789"], "789") == ["123", "456"]
    assert dependency_jobs(["789"], "789") == []
    with pytest.raises(ValueError):
        dependency_jobs(["123_[2-3]"], "789")


def t5(layers=1):
    return T5EncoderForSequenceClassification(
        T5Config(
            vocab_size=24,
            d_model=8,
            d_ff=16,
            d_kv=4,
            num_layers=layers,
            num_heads=2,
            dropout_rate=0.0,
            num_labels=3,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).eval()


def test_encoder_loading_preserves_pretrained_weights(tmp_path):
    original = t5().transformer
    original.save_pretrained(tmp_path)
    loaded = load_encoder(tmp_path, 3)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.transformer.state_dict()[key], value)


@pytest.mark.parametrize("kind", ["t5", "roberta"])
def test_measured_extraction_matches_production_and_fits_all_blocks(kind, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(33)
    source = t5()
    tuned = deepcopy(source)
    with torch.no_grad():
        for parameter in tuned.parameters():
            parameter.add_(0.001 * torch.randn_like(parameter))
    if kind == "t5":
        target = t5(layers=2)
    else:
        target = RobertaForSequenceClassification(
            RobertaConfig(
                vocab_size=24,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_labels=3,
                pad_token_id=0,
                max_position_embeddings=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        ).eval()
    batches = [
        {
            "input_ids": torch.tensor([[2, 3, 1], [4, 5, 1]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        },
        {
            "input_ids": torch.tensor([[6, 7, 1]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([2]),
        },
    ]
    loader = DataLoader(
        [{key: value[i] for key, value in batch.items()} for batch in batches for i in range(len(batch["labels"]))],
        batch_size=2,
    )
    expected = _collect_linear_split(
        source_pretrained=source,
        source_finetuned=tuned,
        target=target,
        source_loader=loader,
        target_loader=loader,
        device=torch.device("cpu"),
    )
    with _head_as_identity(source), _head_as_identity(tuned), _head_as_identity(target):
        extractor = ResidualExtractor(source, tuned)
        data, records = collect(extractor, target, batches, batches, torch.device("cpu"), warmup=1)
    for measured, original in (
        ("f_a", "features_A"),
        ("delta", "delta_A"),
        ("blocks_a", "delta_A_blocks"),
        ("f_b", "features_B"),
    ):
        torch.testing.assert_close(data[measured], expected[original])
    for block, value in data["blocks_b"].items():
        torch.testing.assert_close(value, expected["features_B_blocks"][block])
    assert [record["n_examples"] for record in records] == [2, 1]
    assert all(record["residual_s"] > 0 and record["features_B_s"] > 0 for record in records)
    assert extractor.num_groups + 1 == 3  # One residual block, output group, full JVP.
    config = {
        "stage1_lambda": 1.0,
        "stage1_regularization_mode": "transported",
        "ridge_lambda": 1.0,
        "block_group_strategy": "concat",
        "block_ridge_mode": "smoothed_residual",
        "rho": 0.9,
    }
    w_a = tuned.classification_head.out_proj.weight.detach().double()
    w_b = nearest_mean_head(data["f_b"], torch.arange(3), 3)
    fit = fit_all(data, w_a, w_b, config)
    assert fit["num_ridges"] == 2
    assert fit["ridge_input_dimensions"] == ([16, 8] if kind == "t5" else [8, 8])
    assert fit["ridge_total_s"] == pytest.approx(fit["ridge_preparation_and_stage1_s"] + fit["ridge_stage2_solver_s"])
    rows = [
        {"pair": "t5_to_t5_large", "repeat": 0, "residual_ms_per_example": 1.0, "features_B_ms_per_example": 2.0, **fit}
    ]
    save_results(tmp_path, rows)
    assert "1.0000 ± 0.0000" in (tmp_path / "table.md").read_text()
    assert (tmp_path / "runs.csv").is_file()

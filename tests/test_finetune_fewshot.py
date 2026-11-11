import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.data.vision_loaders import VisionLoaders
from merge_and_rebase.eval.kd_rebase import _vision_support_indices
from merge_and_rebase.finetune.train_vision import _few_shot_train_loaders


class LabeledDataset(TensorDataset):
    label_key = 'label'

    @property
    def split(self):
        return {'label': self.targets.tolist()}

    def _map_label(self, label):
        return int(label)

    @property
    def targets(self):
        return self.tensors[1]


def make_loaders():
    ds = LabeledDataset(torch.arange(240).float().unsqueeze(1), torch.arange(3).repeat_interleave(80))
    train = DataLoader(ds, batch_size=128, drop_last=True)
    return VisionLoaders(train, train, train, ['a', 'b', 'c'], 1, {'train': 240}, {})


@pytest.mark.parametrize('task', ['Cars', 'MNIST', 'SVHN'])
@pytest.mark.parametrize('shots', [1, 2, 5, 10, 20])
def test_support_matches_calibration_and_keeps_small_batches(task, shots):
    original = make_loaders()
    selected = _few_shot_train_loaders(original, task=task, shots=shots, seed=33)
    indices = selected.train.dataset.indices
    assert indices == _vision_support_indices(original.train.dataset, task, shots, 33)
    assert len(indices) == shots * 3
    assert torch.bincount(original.train.dataset.targets[indices]).tolist() == [shots] * 3
    assert sum(len(y) for _, y in selected.train) == shots * 3
    assert selected.val is original.val
    assert selected.test is original.test
    assert len(original.train.dataset) == 240
    assert selected.sizes['train'] == shots * 3


@pytest.mark.parametrize('shots', [0, -1, True, 1.5])
def test_invalid_shots_rejected(shots):
    with pytest.raises(ValueError, match='positive integer'):
        _few_shot_train_loaders(make_loaders(), task='Cars', shots=shots, seed=33)


def test_insufficient_examples_rejected():
    with pytest.raises(ValueError, match='need 81'):
        _few_shot_train_loaders(make_loaders(), task='Cars', shots=81, seed=33)

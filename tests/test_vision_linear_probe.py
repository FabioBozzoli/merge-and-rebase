from __future__ import annotations

import torch

from merge_and_rebase.eval.linear_probe import train_zeroshot_head_probe

from test_steer_rebase import _BUILD_CFG, _CLASSNAMES, _make_tiny_clf, _tiny_loaders


def _clf_with_head(depth: int = 2):
    torch.manual_seed(0)
    clf = _make_tiny_clf(depth=depth, out_dim=len(_CLASSNAMES))
    clf.build_zeroshot_text_features(_CLASSNAMES, _BUILD_CFG)
    return clf


def test_probe_requires_a_head_to_start_from() -> None:
    torch.manual_seed(0)
    clf = _make_tiny_clf(depth=2, out_dim=len(_CLASSNAMES))  # no build_zeroshot_text_features
    loaders = _tiny_loaders()
    try:
        train_zeroshot_head_probe(clf, loaders.train, device="cpu", steps=1)
    except RuntimeError as err:
        assert "build_zeroshot_text_features" in str(err)
    else:
        raise AssertionError("expected a RuntimeError when no zero-shot head is present")


def test_probe_trains_only_the_head_and_leaves_the_classifier_untouched() -> None:
    clf = _clf_with_head()
    loaders = _tiny_loaders()

    backbone_before = {n: p.detach().clone() for n, p in clf.named_parameters()}
    head_before = clf._zs_text_features.detach().clone()
    requires_grad_before = {n: p.requires_grad for n, p in clf.named_parameters()}

    trained = train_zeroshot_head_probe(clf, loaders.train, device="cpu", lr=0.05, steps=5)

    # Backbone frozen: not one parameter moved.
    for n, p in clf.named_parameters():
        assert torch.equal(p, backbone_before[n]), n
    # The classifier is handed back exactly as it came in: same head, same flags.
    assert torch.equal(clf._zs_text_features, head_before)
    assert {n: p.requires_grad for n, p in clf.named_parameters()} == requires_grad_before
    # The returned head has the right shape and actually moved.
    assert trained.shape == head_before.shape
    assert not torch.equal(trained, head_before.to(trained.dtype))


def test_probe_starts_from_the_zero_shot_head() -> None:
    """steer's correction is fit through pinv(w_b) of this exact head, so a probe
    that re-initialized it would throw that alignment away. lr=0 must be a no-op."""
    clf = _clf_with_head()
    loaders = _tiny_loaders()
    head_before = clf._zs_text_features.detach().clone()

    trained = train_zeroshot_head_probe(clf, loaders.train, device="cpu", lr=0.0, steps=2)

    assert torch.allclose(trained, head_before.to(trained.dtype))


def test_probe_fits_its_own_support_set() -> None:
    clf = _clf_with_head()
    loaders = _tiny_loaders()

    trained = train_zeroshot_head_probe(clf, loaders.train, device="cpu", lr=0.05, steps=200)

    from merge_and_rebase.eval.linear_probe import _probe_accuracy

    before = _probe_accuracy(clf, loaders.train, device="cpu", head=clf._zs_text_features)
    after = _probe_accuracy(clf, loaders.train, device="cpu", head=trained)
    assert after >= before

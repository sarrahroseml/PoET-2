"""CPU unit tests for the model-independent M3 pieces (config / schedule / optim / train).

The actual training loop runs only on the GPU box (it loads the real model via a lazy
import), but everything here is verifiable locally.
"""

import importlib.util
import sys

import pytest
import torch

from poet_2.training.config import TrainConfig
from poet_2.training.optim import build_optimizer
from poet_2.training.schedule import build_scheduler


def _sched_lrs(kind, warmup, total, min_lr_frac=0.0, base_lr=1.0, n=None):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base_lr)
    sched = build_scheduler(opt, warmup_steps=warmup, total_steps=total, kind=kind,
                            min_lr_frac=min_lr_frac)
    lrs = []
    for _ in range((n or total) + 1):
        lrs.append(sched.get_last_lr()[0])
        p.grad = torch.zeros_like(p)
        opt.step()
        sched.step()
    return lrs


def test_train_config_defaults_and_collator():
    cfg = TrainConfig(data_dir="x")
    assert cfg.dtype == "bf16" and cfg.optimizer == "adamw"
    cc = cfg.collator_config()
    assert cc.seq_mask_max == cfg.seq_mask_max and cc.rate_cap == cfg.rate_cap
    assert "peak_lr" in cfg.to_dict()


def test_warmup_cosine_shape():
    warmup, total, floor = 5, 20, 0.1
    lrs = _sched_lrs("cosine", warmup, total, min_lr_frac=floor)
    assert lrs[0] == pytest.approx(1 / warmup)          # first warmup step
    assert lrs[warmup - 1] == pytest.approx(1.0)        # peak at end of warmup
    assert lrs[warmup] == pytest.approx(1.0)            # cosine starts at 1.0
    assert lrs[total] == pytest.approx(floor, abs=1e-6) # decays to the floor
    # monotone up during warmup, down after
    assert all(lrs[i] <= lrs[i + 1] for i in range(warmup - 1))
    assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(warmup, total))


def test_inverse_sqrt_shape():
    lrs = _sched_lrs("inverse_sqrt", warmup=4, total=40)
    assert lrs[3] == pytest.approx(1.0)         # peak at end of warmup
    assert lrs[16] == pytest.approx((4 / 17) ** 0.5)  # 1/sqrt-ish decay
    assert lrs[39] < lrs[10]                     # keeps decaying


def test_build_optimizer_adamw_steps_params():
    p = torch.nn.Parameter(torch.ones(4))
    cfg = TrainConfig(data_dir="x", optimizer="adamw", peak_lr=0.1)
    opt = build_optimizer([p], cfg)
    p.grad = torch.ones_like(p)
    before = p.detach().clone()
    opt.step()
    assert not torch.equal(before, p.detach())  # AdamW updated the param


def test_build_optimizer_adafactor_dependency_behavior():
    p = torch.nn.Parameter(torch.ones(4))
    cfg = TrainConfig(data_dir="x", optimizer="adafactor", peak_lr=0.1)
    if importlib.util.find_spec("transformers") is None:
        with pytest.raises(ImportError):
            build_optimizer([p], cfg)
    else:  # pragma: no cover - depends on env
        assert build_optimizer([p], cfg) is not None


def test_build_optimizer_unknown_raises():
    cfg = TrainConfig(data_dir="x", optimizer="nope")
    with pytest.raises(ValueError):
        build_optimizer([torch.nn.Parameter(torch.zeros(1))], cfg)


def test_group_indices_by_budget():
    from poet_2.training.train import group_indices_by_budget

    counts = [10, 10, 10, 30, 5, 5]
    groups = group_indices_by_budget(counts, budget=25)
    assert [len(g) for g in groups] == [2, 1, 1, 2]  # [10,10] [10] [30] [5,5]
    assert sum(len(g) for g in groups) == len(counts)


def test_train_module_imports_without_flash_attn():
    # importing the loop must NOT import the model / flash_attn (lazy in load_trainable_model)
    sys.modules.pop("flash_attn", None)
    import poet_2.training.train  # noqa: F401

    assert "flash_attn" not in sys.modules
    assert hasattr(poet_2.training.train, "train")
    assert hasattr(poet_2.training.train, "load_trainable_model")

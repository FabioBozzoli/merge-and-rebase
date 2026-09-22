#!/usr/bin/env python
"""The t5-encoder ``steer_text`` grid: what it contains, what state each cell is in, how to submit it.

One **cell** is one run: (task, arm, stage-2 strategy, few_shot, seed) -- the
strategy being None for arms that have no stage 2. One **arm**
is a (checkpoint flavour, ``feature_regime``) pair. One **group** is a submission
batch -- the unit you launch, so ~1200 minutes of queue time go out a slice at a
time rather than all at once.

State is *derived*, never recorded: a cell is ``completed`` if its directory holds
a parseable ``summary.json``, ``running`` if its ``meta.json`` names a job still in
``squeue``, ``failed`` if the directory exists but neither holds, and ``missing``
otherwise. There is no hand-maintained list to drift out of sync with the disk.

There is deliberately **no per-flavour config file**. Everything is the one base
config plus dotted overrides, which is what ``resolve_config.py`` exists for --
so the flavour to checkpoint-directory and flavour to cache-directory mappings
live here, once, instead of being duplicated across near-identical JSONs. Every
override is archived into ``<exp-dir>/config.json`` at submit time, so a run stays
reconstructible from its own directory.

Usage:
    python scripts/slurm/grid.py status [--group G] [--per-job]
    python scripts/slurm/grid.py submit --group G [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = "configs/text_rebase_t5enc_steer.json"  # relative: submit_text_rebase.sh cds to REPO
BASE_CONFIG_THESEUS = "configs/text_rebase_t5enc_theseus.json"
BASE_CONFIG_LINPROBE = "configs/text_probe_t5large_nearest_mean.json"
WORK = Path("/work/intesasanpaolo_phd/merge-and-rebase")
RESULTS_ROOT = WORK / "t5enc_steer_text"
# theseus writes to its own root so that nothing -- a mistyped FORCE=1, a stray
# collection pass -- can reach the 378 finished steer runs. The grid *state* files
# still go under RESULTS_ROOT, which is the one place to look for the whole picture.
RESULTS_ROOT_THESEUS = WORK / "t5enc_theseus"
# The target-only control. One subdirectory per head initialization, so a later
# random-init sweep sits beside this one instead of inside it.
RESULTS_ROOT_LINPROBE = WORK / "t5-large_lin-probe" / "nearest_mean"
HEADS = WORK / "text_rebase_heads"
CONVERTED = WORK / "checkpoints_t5_converted"

TASKS = ("mnli", "sick", "rte", "scitail", "qnli", "snli")
SEEDS = (33, 54, 89)
# Both samplers draw an exact class-balanced support and raise rather than cap
# (steer.py:_few_shot, theseus.py:_class_balanced_indices), and sick's contradiction
# class has only 606 rows in the 4000-row train pool. Confirmed by jobs 112754
# (steer) and 112755 (theseus), both "Class 2 has 606 examples, need 1000".
# The next-smallest class anywhere is mnli's 1160, so every other task reaches 1000.
TASKS_FS1000 = tuple(t for t in TASKS if t != "sick")

# B's head is nearest-class-mean centroids over 300 support examples on t5-large.
# It depends on the *target*, not on A's checkpoint, so one set of six files serves
# every arm -- and it is deliberately NOT rebuilt per few_shot, so that axis varies
# only the steer stages' support and not B's head budget.
HEAD_FEWSHOT = 300


def head_file(task: str) -> Path:
    return HEADS / f"t5-large_{task}_nearest_mean_enc_seed33_fewshot{HEAD_FEWSHOT}.pt"


@dataclass(frozen=True)
class Arm:
    """A checkpoint flavour evaluated one way: a regime for steer, a method for theseus.

    The method belongs here rather than on the group because everything that
    changes with it -- which config to resolve, which results root to write, how
    many cores to ask for, whether there is a feature cache at all -- is a property
    of how this flavour is being evaluated.
    """

    flavour: str
    regime: str
    ckpt_root: Path | None  # None: this entrypoint has no source checkpoint
    cache_dir: Path | None  # None: this method caches no features
    method: str = "steer_text"
    base_config: str = BASE_CONFIG
    results_root: Path = RESULTS_ROOT
    cpus: int = 2
    entrypoint: str = "merge_and_rebase.eval.text_rebase"

    @property
    def key(self) -> str:
        return f"{self.flavour}:{self.regime}"


# The two arms. `tangent: true` in the training args is what decides the regime:
# ntk's deltas were fitted to f(t0) + J(t0)(t - t0), so `linear` is what reconstructs
# them -- confirmed empirically, identify_checkpoint_tasks.py --linearized moves every
# task from 0.34-0.68 (nonlinear forward) to 0.83-1.00.
#
# The separate cache_dir is load-bearing, not tidiness: _model_tag (text_rebase.py:543)
# keys the feature cache on the *base* model ids only, with no trace of which tuned
# checkpoint produced delta_A. Point two flavours at one cache dir and the second
# silently reads the first's deltas, with results that look entirely plausible.
ARMS = {
    a.key: a
    for a in (
        Arm("noreg-nonlinear", "standard", CONVERTED / "t5base_6text_noreg_nonlinear_v2", WORK / ".cache"),
        Arm("noreg-ntk", "linear", CONVERTED / "t5base_6text_noreg_ntk", WORK / ".cache_ntk"),
        # theseus transports weights instead of correcting features, so it has no
        # feature_regime and no feature cache: `cache_dir=None` is what tells the
        # cold-cache guard there is nothing to warm. cpus stays low because the Gram
        # and SVD run on the GPU in float32 (method_params.compute_*); if that is ever
        # turned off, this is the number that has to go back up -- measured at 26.3 min
        # a run on 2 CPU threads in float64 versus 6.5 at 16.
        Arm(
            "noreg-nonlinear", "theseus", CONVERTED / "t5base_6text_noreg_nonlinear_v2", None,
            method="theseus", base_config=BASE_CONFIG_THESEUS,
            results_root=RESULTS_ROOT_THESEUS, cpus=4,
        ),
        # t5-large alone, through its own entrypoint: a nearest-mean head fit on
        # the support set and then probed on it. No source model, no delta, no
        # feature cache -- hence ckpt_root/cache_dir None, which also short-circuits
        # the cold-cache guard.
        Arm(
            "linprobe-nm", "standard", None, None,
            method="linear_probe", base_config=BASE_CONFIG_LINPROBE,
            results_root=RESULTS_ROOT_LINPROBE,
            entrypoint="merge_and_rebase.eval.text_linear_probe",
        ),
    )
}

NONLINEAR = "noreg-nonlinear:standard"
NTK = "noreg-ntk:linear"
THESEUS = "noreg-nonlinear:theseus"
LINPROBE = "linprobe-nm:standard"


@dataclass(frozen=True)
class Group:
    name: str
    arm: str
    # steer_text's stage 2. None for arms that have no such stage (theseus
    # transports weights; the linear probe has no rebasin step at all), where any
    # value here would name a solver that never runs.
    strategy: str | None
    few_shot: int
    seeds: tuple[int, ...]
    note: str = ""
    # Only meaningful for `block_ridge`. "independent" is steer's own default, so
    # leaving it alone reproduces every group that ran before this axis existed --
    # byte-identical names, byte-identical overrides.
    block_ridge_mode: str = "independent"
    rho: float | None = None
    tasks: tuple[str, ...] = TASKS
    # Probe training budget. Only linear_probe cells vary it; 200 is what the first
    # 54 ran with, so it stays the default and those names keep their `ep200` tag.
    epochs: int = 200

    @property
    def size(self) -> int:
        return len(self.tasks) * len(self.seeds)


# Order matters: it is the launch order. The four nonlinear groups run first because
# their cache is already warm, so they return in minutes and exercise the whole chain
# before any expensive work. `ntk-warm` is the only slow batch -- it fills the six
# linear caches, and every ntk group after it is a cache hit.
GROUPS = (
    Group("nl-gr-fs300", NONLINEAR, "global_ridge", 300, SEEDS, "the original sweep; already complete"),
    Group("nl-gm-fs300", NONLINEAR, "global_mlp", 300, SEEDS, "the original sweep; already complete"),
    Group("nl-gr-fs200", NONLINEAR, "global_ridge", 200, SEEDS),
    Group("nl-gm-fs200", NONLINEAR, "global_mlp", 200, SEEDS),
    Group("nl-gr-fs500", NONLINEAR, "global_ridge", 500, SEEDS),
    Group("nl-gm-fs500", NONLINEAR, "global_mlp", 500, SEEDS, "arm noreg-nonlinear complete"),
    Group("ntk-warm", NTK, "global_ridge", 300, (33,), "fills the six linear caches; run this alone, first"),
    Group("ntk-gr-fs300", NTK, "global_ridge", 300, (54, 89), "the seeds ntk-warm left"),
    Group("ntk-gr-fs200", NTK, "global_ridge", 200, SEEDS),
    Group("ntk-gr-fs500", NTK, "global_ridge", 500, SEEDS),
    Group("ntk-gm-fs200", NTK, "global_mlp", 200, SEEDS),
    Group("ntk-gm-fs300", NTK, "global_mlp", 300, SEEDS),
    Group("ntk-gm-fs500", NTK, "global_mlp", 500, SEEDS),
    Group("ntk-br-fs200", NTK, "block_ridge", 200, SEEDS),
    Group("ntk-br-fs300", NTK, "block_ridge", 300, SEEDS),
    Group("ntk-br-fs500", NTK, "block_ridge", 500, SEEDS, "arm noreg-ntk complete"),
    # `smoothed_residual` re-fits the same thirteen blocks, carrying each block's
    # unexplained residual forward at rate `rho` instead of discarding it
    # (steer.py:_fit_block_ridge). Stage 1 and the features are untouched, so every
    # cell below is a cache hit against the same warm .cache_ntk the groups above used.
    Group("ntk-br-rho0.5-fs200", NTK, "block_ridge", 200, SEEDS, block_ridge_mode="smoothed_residual", rho=0.5),
    Group("ntk-br-rho0.5-fs300", NTK, "block_ridge", 300, SEEDS, block_ridge_mode="smoothed_residual", rho=0.5),
    Group("ntk-br-rho0.5-fs500", NTK, "block_ridge", 500, SEEDS, block_ridge_mode="smoothed_residual", rho=0.5),
    Group("ntk-br-rho1.0-fs200", NTK, "block_ridge", 200, SEEDS, block_ridge_mode="smoothed_residual", rho=1.0),
    Group("ntk-br-rho1.0-fs300", NTK, "block_ridge", 300, SEEDS, block_ridge_mode="smoothed_residual", rho=1.0),
    Group("ntk-br-rho1.0-fs500", NTK, "block_ridge", 500, SEEDS, block_ridge_mode="smoothed_residual", rho=1.0,
          note="arm noreg-ntk smoothed_residual complete"),
    # theseus. `few_shot` carries shots_per_class -- the same quantity (N per class)
    # under the name the other method uses for it, so the support axis stays one axis.
    Group("th-spc200", THESEUS, "theseus", 200, SEEDS),
    Group("th-spc300", THESEUS, "theseus", 300, SEEDS),
    Group("th-spc500", THESEUS, "theseus", 500, SEEDS, note="theseus arm complete"),
    # Extension: a third rho, and support size 1000 on the three configurations below.
    Group("ntk-br-rho0.9-fs200", NTK, "block_ridge", 200, SEEDS, block_ridge_mode="smoothed_residual", rho=0.9),
    Group("ntk-br-rho0.9-fs300", NTK, "block_ridge", 300, SEEDS, block_ridge_mode="smoothed_residual", rho=0.9),
    Group("ntk-br-rho0.9-fs500", NTK, "block_ridge", 500, SEEDS, block_ridge_mode="smoothed_residual", rho=0.9),
    Group("ntk-br-rho0.9-fs1000", NTK, "block_ridge", 1000, SEEDS, block_ridge_mode="smoothed_residual", rho=0.9,
          tasks=TASKS_FS1000),
    Group("nl-gm-fs1000", NONLINEAR, "global_mlp", 1000, SEEDS, tasks=TASKS_FS1000),
    Group("th-spc1000", THESEUS, "theseus", 1000, SEEDS, tasks=TASKS_FS1000),
    # The t5-large linear-probe control: no stage 2, hence strategy None.
    # The probe control, swept over support size x training budget. 1e-4 was still
    # improving monotonically at epoch 200 (job 114274), so the budget is an axis
    # rather than a constant.
    Group("lp-fs200-ep200", LINPROBE, None, 200, SEEDS),
    Group("lp-fs300-ep200", LINPROBE, None, 300, SEEDS),
    Group("lp-fs500-ep200", LINPROBE, None, 500, SEEDS),
    Group("lp-fs200-ep300", LINPROBE, None, 200, SEEDS, epochs=300),
    Group("lp-fs300-ep300", LINPROBE, None, 300, SEEDS, epochs=300),
    Group("lp-fs500-ep300", LINPROBE, None, 500, SEEDS, epochs=300),
    Group("lp-fs200-ep400", LINPROBE, None, 200, SEEDS, epochs=400),
    Group("lp-fs300-ep400", LINPROBE, None, 300, SEEDS, epochs=400),
    Group("lp-fs500-ep400", LINPROBE, None, 500, SEEDS, epochs=400),
    Group("lp-fs200-ep500", LINPROBE, None, 200, SEEDS, epochs=500),
    Group("lp-fs300-ep500", LINPROBE, None, 300, SEEDS, epochs=500),
    Group("lp-fs500-ep500", LINPROBE, None, 500, SEEDS, epochs=500),
    # Support 1000 at the best setting only (lr 1e-4, 500 epochs). sick cannot
    # reach 1000 per class, hence TASKS_FS1000 -- same 606-example constraint the
    # steer and theseus fs1000 groups hit.
    Group("lp-fs1000-ep500", LINPROBE, None, 1000, SEEDS, epochs=500, tasks=TASKS_FS1000,
          note="linear-probe control complete"),
)

# (MEM, TIME). block_ridge is the outlier: need_blocks makes it load the train and
# test block tensors and promote them to float64, then solve thirteen 2048x2048
# ridge systems. The other two never load the block tensors at all, even in a
# linear-regime arm (steer.py:_load_or_compute_split).
RESOURCES = {
    "global_ridge": ("8G", "00:15:00"),
    "global_mlp": ("8G", "00:15:00"),
    # Measured on the snli/fs500 probe (job 111507): 1:31 elapsed, 5.03 GB MaxRSS --
    # the heaviest cell of the class, since five of six tasks share train=4000/test=1800
    # and only rte is smaller. The 24G/30min this used to ask for was arithmetic, not
    # measurement, and ~5x too generous; a smaller ask schedules sooner.
    "block_ridge": ("12G", "00:15:00"),
}
# A cell whose feature cache is still cold pays for the collection itself. In the
# linear regime that is 13 masked jvps plus one full jvp per batch through t5-base
# against one plain forward in `standard` -- roughly 30x the source-side cost.
# Measured on the rte fill (job 111309): 8:42 elapsed, 4.16 GB MaxRSS for 2490 train
# + 249 test examples. The other five tasks carry 4000 train rows and up to 2000 test,
# ~1.3x the examples, so this leaves roughly 3x headroom on both -- deliberately tighter
# than the 32G/2h the first one asked for, because a smaller ask schedules sooner.
COLD = {"standard": ("16G", "00:40:00"), "linear": ("16G", "00:40:00")}
# Measured on the two ends of the class: rte/spc200 (job 112137) at 2:24, and the
# heaviest cell snli/spc500 (job 112145) at 3:52 with 7.27 GB MaxRSS, of which
# prepare is 147 s. Calibration size barely moves it because the Gram and SVD run on
# the GPU; what is left is model loading and evaluation, which every cell pays alike.
# The binding constraint is GPU memory, not this ask: at method_params.batch_size=16
# the activation hooks need 17.0 GiB against a 15.74 GiB card and OOM (hence
# batch_size=4, worth 2.8 GiB instead of 11.3).
THESEUS_RESOURCES = ("12G", "00:12:00")
# Measured on the heaviest cell, snli/fs500 through eval.text_linear_probe
# (job 114554): 1:27 and 1.55 GB MaxRSS, 1.51 GB on the GPU. It is that small
# because the probe caches the head's inputs once instead of re-running the
# encoder every epoch -- the same cell cost 25:09 before that (job 113978), and
# 2:15 through steer_text at alpha=0 (job 114526). Wall time still scales with
# support size, so the smaller cells get less; 4G is over twice the measured peak.
LINPROBE_RESOURCES = {
    200: ("4G", "00:06:00"), 300: ("4G", "00:08:00"), 500: ("4G", "00:10:00"),
    # fs500/ep500 measured at 2:18 worst case (1.56 GB RAM, 1.80 GB GPU); doubling
    # the support roughly doubles both the one feature pass and the per-epoch head
    # solve, and memory is dominated by the model either way.
    1000: ("4G", "00:15:00"),
}


def resources_for(cell: Cell, *, cold: bool) -> tuple[str, str]:
    arm = ARMS[cell.arm]
    if arm.method == "theseus":
        return THESEUS_RESOURCES
    if arm.method == "linear_probe":
        return LINPROBE_RESOURCES[cell.few_shot]
    return COLD[arm.regime] if cold else RESOURCES[cell.strategy]


@dataclass(frozen=True)
class Cell:
    group: str
    arm: str
    task: str
    strategy: str | None
    few_shot: int
    seed: int
    block_ridge_mode: str = "independent"
    rho: float | None = None
    epochs: int = 200

    @property
    def fit_tag(self) -> str:
        """The name field that distinguishes a non-default block_ridge fit.

        Empty for "independent", which is what keeps the 270 directories that
        predate this axis addressable under the names they already have -- and
        makes a collision with them structurally impossible rather than merely
        unlikely.
        """
        if self.block_ridge_mode == "independent":
            return ""
        return f"_smoothres-rho{self.rho}"

    @property
    def support_tag(self) -> str:
        """``fs`` for steer's few_shot, ``spc`` for theseus's shots_per_class."""
        return "spc" if ARMS[self.arm].method == "theseus" else "fs"

    @property
    def name(self) -> str:
        a = ARMS[self.arm]
        if a.method == "theseus":
            return f"{self.task}_theseus_{a.flavour}_spc{self.few_shot}_seed{self.seed}"
        if a.method == "linear_probe":
            return (
                f"{self.task}_linprobe_t5-large_nearest-mean_"
                f"fs{self.few_shot}_ep{self.epochs}_seed{self.seed}"
            )
        return (
            f"{self.task}_{a.regime}_{a.flavour}_{self.strategy}"
            f"{self.fit_tag}_fs{self.few_shot}_seed{self.seed}"
        )

    @property
    def directory(self) -> Path:
        return ARMS[self.arm].results_root / self.name

    def descriptor(self, *, underscore_flavour: bool = False) -> str:
        """The middle of a status line: everything but the task, support and seeds.

        The flavour is the only field whose hyphen is cosmetic, so the underscore
        swap is scoped to it -- a blanket replace would also rewrite the fit label
        (`smoothed_residual-rho0.5`), which is not the same name.
        """
        a = ARMS[self.arm]
        flavour = a.flavour.replace("-", "_") if underscore_flavour else a.flavour
        if a.method == "theseus":
            return f"theseus {flavour}"
        if a.method == "linear_probe":
            return f"linprobe t5-large nearest_mean ep{self.epochs}"
        return f"{a.regime} {flavour} {_field(self.strategy, self.fit_label)}"

    @property
    def rollup_key(self) -> tuple[str, str, str, str | None, str, str, int, int]:
        a = ARMS[self.arm]
        return (
            self.task, a.method, a.regime, a.flavour, self.strategy, self.fit_label,
            # Before few_shot, so the compact report collapses support size and seed
            # into braces while keeping one line per training budget.
            self.epochs, self.few_shot,
        )

    @property
    def compact_key(self) -> tuple[str, str, str, str | None, str, str, int]:
        """``rollup_key`` without the support size, for the one-line-per-config report."""
        return self.rollup_key[:-1]

    @property
    def fit_label(self) -> str:
        """How the block_ridge coefficients were fit, spelled out for a report.

        Empty for every non-block_ridge cell, where the parameter is unused and
        printing a default would suggest an axis that does not exist.
        """
        if self.strategy != "block_ridge":
            return ""
        if self.block_ridge_mode == "independent":
            return "independent"
        return f"{self.block_ridge_mode}-rho{self.rho}"


def all_cells() -> list[Cell]:
    return [
        Cell(g.name, g.arm, task, g.strategy, g.few_shot, seed, g.block_ridge_mode, g.rho, g.epochs)
        for g in GROUPS
        for task in g.tasks
        for seed in g.seeds
    ]


# --------------------------------------------------------------------------
# State, derived from disk and Slurm
# --------------------------------------------------------------------------

_ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "SUSPENDED"}


def active_job_ids() -> set[str]:
    """Job ids of this user that Slurm still considers live."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A %T"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    ids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in _ACTIVE:
            ids.add(parts[0])
    return ids


def _meta(cell: Cell) -> dict:
    path = cell.directory / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def state_of(cell: Cell, live: set[str]) -> tuple[str, str | None]:
    """``(state, slurm_job_id)``. Nothing here trusts a recorded status."""
    directory = cell.directory
    if not directory.is_dir():
        return "missing", None
    job_id = str(_meta(cell).get("slurm_job_id") or "") or None
    summary = directory / "summary.json"
    payload: dict | None = None
    if summary.is_file():
        try:
            loaded = json.loads(summary.read_text())
            payload = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            return "failed", job_id  # truncated summary: the job died mid-write
    # A summary that merely parses is not a finished run. The run logger writes a
    # stub ({"run_logging": ...}) the moment it starts, so a job that dies later --
    # a CUDA OOM inside prepare, say -- leaves valid JSON with no results in it and
    # would otherwise read as `completed` forever. Completion means test_results,
    # which is also the key collect_t5enc_results.py needs to build a record.
    if payload is not None and payload.get("test_results"):
        return "completed", job_id
    if job_id and job_id in live:
        return "running", job_id
    return "failed", job_id


def cache_is_warm(arm: Arm, task: str) -> bool:
    """Does this arm already hold train+test features for ``task``?

    Mirrors ``steer._cache_split_dir`` and ``text_rebase._model_tag``; the tags are
    the *base* model ids with the model kind appended, which is why every flavour
    needs its own ``cache_dir``. ``status`` cross-checks the rule against a cell it
    knows completed, so a drift shows up as a warning rather than as a silent
    "cold" that makes every submission ask for two hours.
    """
    if arm.cache_dir is None:
        return True  # this method caches nothing; there is no cold state to guard
    cfg = json.loads((REPO / arm.base_config).read_text())
    kind = str(cfg["model_kind"]).strip().lower()
    suffix = "" if kind == "sequence_classification" else f"__{kind}"
    source = str(cfg["source_model_name_or_path"]).replace("/", "__") + suffix
    target = str(cfg["target_model_name_or_path"]).replace("/", "__") + suffix
    base = arm.cache_dir / f"{source}_to_{target}" / task / arm.regime
    return all((base / split).is_dir() for split in ("train", "test"))


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def overrides_for(cell: Cell, *, save_artifacts: bool) -> list[str]:
    arm = ARMS[cell.arm]
    if arm.method == "theseus":
        return [
            f"tasks={cell.task}",
            f"seed={cell.seed}",
            f"tuned_ckpts.{cell.task}={arm.ckpt_root / cell.task}",
            f"target_task_heads={head_file(cell.task)}",
            f"method_params.shots_per_class={cell.few_shot}",
            # The one that actually reaches theseus. text_rebase.py's shim branch
            # (:1175) splats **method_params without injecting the run's seed the way
            # the steer branch does (:1214), so theseus would otherwise use its own
            # default of 0 and all three "seeds" would be the same run.
            f"method_params.seed={cell.seed}",
            # 702 MB a run, 37 GB over the grid, and exactly reproducible from this
            # config. Only the smoke cell asks for it.
            f"save_transported_tvs_dir={cell.directory}" if save_artifacts else "save_transported_tvs_dir=null",
        ]
    if arm.method == "linear_probe":
        return [
            f"tasks={cell.task}",
            # The probe's support draw: balanced_indices(..., seed=cfg["seed"]).
            f"seed={cell.seed}",
            f"linear_probe_shots_per_class={cell.few_shot}",
            f"linear_probe_epochs={cell.epochs}",
        ]
    items = [
        f"tasks={cell.task}",
        f"seed={cell.seed}",
        f"tuned_ckpts.{cell.task}={arm.ckpt_root / cell.task}",
        f"method_params.feature_cache_dir={arm.cache_dir}/",
        f"method_params.feature_regime={arm.regime}",
        f"method_params.stage_2_strategy={cell.strategy}",
        f"method_params.few_shot={cell.few_shot}",
        f"target_task_heads={head_file(cell.task)}",
    ]
    if cell.block_ridge_mode != "independent":
        # resolve_config.py only guards *parent* paths, and steer_text.prepare ends
        # in `**kwargs: Any` / `del kwargs` -- so a misspelled leaf here would be
        # written into config.json, silently dropped at prepare(), and the run would
        # quietly fit "independent" while looking like a success. The probe cell
        # compares stage2_test_acc against its independent twin to catch exactly that.
        items.append(f"method_params.block_ridge_mode={cell.block_ridge_mode}")
        items.append(f"method_params.rho={cell.rho}")
    if cell.strategy == "block_ridge" and not save_artifacts:
        # 13 x [2048, 1024] float64 is ~218 MB a run, and it is exactly reproducible
        # from the cached features plus this config. null beats submit_text_rebase.sh's
        # --set-if-absent default, which runs before the positional overrides.
        items.append("save_steer_artifacts_dir=null")
    return items


def submit(cell: Cell, *, cold: bool, save_artifacts: bool, dry_run: bool) -> str | None:
    arm = ARMS[cell.arm]
    mem, walltime = resources_for(cell, cold=cold)
    env = {
        **os.environ,
        "RESULTS_ROOT": str(arm.results_root),
        "PARTITION": "all_usr_prod",
        "ACCOUNT": "intesasanpaolo_phd",
        "GRES": "gpu:1",
        "CPUS": str(arm.cpus),
        "MEM": mem,
        "TIME": walltime,
        "ENTRYPOINT": arm.entrypoint,
    }
    cmd = [str(REPO / "scripts/slurm/submit_text_rebase.sh"), cell.name, arm.base_config]
    cmd += overrides_for(cell, save_artifacts=save_artifacts)
    tag = f"[{mem} {walltime}{' COLD' if cold else ''}]"
    if dry_run:
        print(f"  {tag} {' '.join(cmd)}")
        return None
    result = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  FAILED {cell.name}\n{result.stdout}{result.stderr}", file=sys.stderr)
        return None
    job_id = next((ln.split()[-1] for ln in result.stdout.splitlines() if ln.startswith("submitted job ")), "?")
    print(f"  {tag} {cell.name} -> job {job_id}")
    return job_id


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

_TOKEN = {"completed": "done✅", "running": "running🔄", "failed": "failed❌", "missing": "missing⬜"}
_PRIORITY = ("running", "failed", "missing", "completed")


def _field(strategy: str | None, fit: str) -> str:
    """``strategy`` plus its fit label, which only block_ridge has."""
    return f"{strategy} {fit}" if fit else strategy


def _rollup_token(states: list[str]) -> str:
    primary = next((s for s in _PRIORITY if s in states), "missing")
    token = _TOKEN[primary]
    if len(set(states)) > 1:
        counts = " ".join(f"{states.count(s)}{_TOKEN[s][-1]}" for s in _PRIORITY if s in states)
        token = f"{token} ({counts})"
    return token


def cmd_status(args: argparse.Namespace) -> int:
    live = active_job_ids()
    cells = [
        c for c in all_cells()
        if (args.group is None or c.group == args.group)
        and (args.method is None or ARMS[c.arm].method == args.method)
    ]
    if not cells:
        print(f"no cells match (group={args.group}, method={args.method})", file=sys.stderr)
        return 2
    # A filtered report belongs with the runs it describes: when every selected cell
    # shares one results root, write there instead of the steer root, so a method kept
    # deliberately separate has its own grid file next to its own results.
    state_root = {ARMS[c.arm].results_root for c in cells}
    state_root = state_root.pop() if len(state_root) == 1 else RESULTS_ROOT
    records = []
    for cell in cells:
        state, job_id = state_of(cell, live)
        arm = ARMS[cell.arm]
        records.append({
            "experiment": cell.name, "group": cell.group, "arm": cell.arm,
            "flavour": arm.flavour, "feature_regime": arm.regime, "task": cell.task,
            "method": arm.method, "stage_2_strategy": cell.strategy,
            "few_shot": cell.few_shot, "seed": cell.seed,
            "block_ridge_mode": cell.block_ridge_mode if cell.strategy == "block_ridge" else None,
            "rho": cell.rho if cell.strategy == "block_ridge" else None,
            "fit": cell.fit_label,
            "state": state, "slurm_job_id": job_id,
        })

    lines: list[str] = []
    if args.per_job:
        for cell, r in zip(cells, records, strict=True):
            lines.append(
                f"{cell.task} {cell.descriptor()} "
                f"{cell.support_tag}{cell.few_shot} seed{cell.seed} {_TOKEN[r['state']]}"
            )
    elif args.rollup:
        # One line per configuration: support size and seed both collapse into braces,
        # so a line stands for every run that differs only in those two.
        seen: dict[tuple, list[tuple[Cell, dict]]] = {}
        for cell, r in zip(cells, records, strict=True):
            seen.setdefault(cell.compact_key, []).append((cell, r))
        for group in seen.values():
            head = group[0][0]
            shots = ",".join(str(v) for v in sorted({c.few_shot for c, _ in group}))
            seeds = ",".join(str(v) for v in sorted({c.seed for c, _ in group}))
            lines.append(
                f"{head.task} {head.descriptor(underscore_flavour=True)} "
                f"{head.support_tag}{{{shots}}} seed{{{seeds}}} "
                f"{_rollup_token([r['state'] for _, r in group])}"
            )
    else:
        seen = {}
        for cell, r in zip(cells, records, strict=True):
            seen.setdefault(cell.rollup_key, []).append((cell, r))
        for group in seen.values():
            head = group[0][0]
            group.sort(key=lambda pair: pair[0].seed)
            seeds = ",".join(str(c.seed) for c, _ in group)
            lines.append(
                f"{head.task} {head.descriptor()} {head.support_tag}{head.few_shot} seeds {seeds} "
                f"{_rollup_token([r['state'] for _, r in group])}"
            )

    tally = {s: sum(1 for r in records if r["state"] == s) for s in _PRIORITY}
    summary = "  ".join(f"{_TOKEN[s]} {n}" for s, n in tally.items() if n)
    selected_groups = {c.group for c in cells}
    by_group = {
        g.name: {s: sum(1 for r in records if r["group"] == g.name and r["state"] == s) for s in _PRIORITY}
        for g in GROUPS
        if g.name in selected_groups
    }

    print("\n".join(lines))
    print(f"\n{len(records)} runs:  {summary}")

    # Drift check: a task with completed runs must read as a warm cache. If it does
    # not, cache_is_warm's copy of _model_tag/_cache_split_dir has fallen out of step
    # with the real one, and every submission would silently ask for cold resources.
    for arm_key, arm in ((k, a) for k, a in ARMS.items() if any(c.arm == k for c in cells)):
        done = {r["task"] for r in records if r["arm"] == arm_key and r["state"] == "completed"}
        stale = sorted(t for t in done if not cache_is_warm(arm, t))
        if stale:
            print(f"WARNING: {arm_key} has completed runs for {', '.join(stale)} but reads as a cold "
                  f"cache -- check cache_is_warm against steer._cache_split_dir.", file=sys.stderr)
    print("\ngroup                 done  run  fail  miss   (launch order)")
    for name, counts in by_group.items():
        print(f"  {name:<20} {counts['completed']:>4} {counts['running']:>4} "
              f"{counts['failed']:>5} {counts['missing']:>5}")

    if not args.no_write:
        state_root.mkdir(parents=True, exist_ok=True)
        # `_`-prefixed so collect_t5enc_results.py's SKIP_PREFIX ignores them.
        (state_root / "_grid_state.json").write_text(
            json.dumps({"total": len(records), "tally": tally, "by_group": by_group, "runs": records}, indent=2) + "\n"
        )
        name = "_grid_status_rollup.txt" if args.rollup else "_grid_state.txt"
        header = f"{len(records)} runs in {len(lines)} lines" if args.rollup else f"{len(records)} runs"
        (state_root / name).write_text("\n".join(lines) + f"\n\n{header}:  {summary}\n")
        print(f"\nwrote {state_root}/_grid_state.json and {name}")
    return 0


def cmd_check_overrides(args: argparse.Namespace) -> int:
    """Replay overrides_for against what every finished run actually recorded.

    A driver refactor is easy to get subtly wrong: rename a field, reorder a list,
    and the *next* submission silently disagrees with everything already on disk.
    meta.json stores the exact override list each run was submitted with, so
    regenerating it and diffing is a one-command proof that nothing moved.

    Known baseline: 36 cells differ, all in nl-gr-fs300 / nl-gm-fs300. Those were
    submitted by hand before grid.py existed and leaned on the base config's
    defaults, so they carry a shorter list with the same meaning.
    """
    live = active_job_ids()
    drifted: dict[str, int] = {}
    checked = 0
    for cell in all_cells():
        meta_path = cell.directory / "meta.json"
        if not meta_path.is_file():
            continue
        checked += 1
        recorded = _meta(cell).get("overrides", [])
        current = overrides_for(cell, save_artifacts=False)
        if recorded != current:
            drifted[cell.group] = drifted.get(cell.group, 0) + 1
            if args.verbose:
                print(f"DRIFT {cell.name}\n  recorded: {recorded}\n  current:  {current}")
    del live
    print(f"compared {checked} submitted cells against their recorded overrides")
    if not drifted:
        print("  no drift")
        return 0
    for group, count in sorted(drifted.items()):
        print(f"  {group:<22} {count} differ")
    print("  (re-run with --verbose to see each one)")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    group = next((g for g in GROUPS if g.name == args.group), None)
    if group is None:
        print(f"error: unknown group {args.group!r}. Known: {', '.join(g.name for g in GROUPS)}", file=sys.stderr)
        return 2

    live = active_job_ids()
    wanted = {"missing", "failed"} if args.retry_failed else {"missing"}
    pending = [
        c for c in all_cells()
        if c.group == group.name
        and (not args.task or c.task in args.task)
        and state_of(c, live)[0] in wanted
    ]
    if not pending:
        print(f"group {group.name}: nothing to submit (all cells accounted for).")
        return 0

    arm = ARMS[group.arm]
    cold = {task for task in {c.task for c in pending} if not cache_is_warm(arm, task)}
    if cold and not args.force_cold:
        stampede = [t for t in cold if sum(1 for c in pending if c.task == t) > 1]
        if stampede:
            print(
                f"error: {arm.key} has a cold feature cache for {', '.join(sorted(stampede))}, and this group\n"
                f"       would submit several runs per task against it -- they would each recompute the same\n"
                f"       features and race on the same directory. Fill it one run per task first:\n"
                f"         python scripts/slurm/grid.py submit --group ntk-warm\n"
                f"       or pass --force-cold if you really mean it.",
                file=sys.stderr,
            )
            return 1

    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"group {group.name}: submitting {len(pending)} of {group.size} cells"
          f"{' (dry run)' if args.dry_run else ''}")
    for cell in pending:
        submit(cell, cold=cell.task in cold, save_artifacts=args.save_artifacts, dry_run=args.dry_run)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="Report every cell's state and write the state files.")
    s.add_argument("--group", help="Restrict to one submission group.")
    s.add_argument("--per-job", action="store_true", help="One line per run instead of per cell.")
    s.add_argument("--rollup", action="store_true", help="One line per configuration, collapsing few_shot and seed into braces.")
    s.add_argument("--method", choices=sorted({a.method for a in ARMS.values()}), help="Restrict to one rebase method; the state files are then written under that method's results root.")
    s.add_argument("--no-write", action="store_true", help="Print only; do not write _grid_state.*")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("check-overrides", help="Diff every finished run's recorded overrides against what grid.py generates now.")
    s.add_argument("--verbose", action="store_true", help="Print each differing cell.")
    s.set_defaults(func=cmd_check_overrides)

    s = sub.add_parser("submit", help="Submit the cells of one group that are still missing.")
    s.add_argument("--group", required=True)
    s.add_argument("--limit", type=int, help="Submit at most this many.")
    s.add_argument("--task", action="append", choices=TASKS, help="Restrict to these tasks (repeatable). Useful for filling one cold cache at a time.")
    s.add_argument("--retry-failed", action="store_true", help="Also resubmit cells whose directory exists but never produced a summary.")
    s.add_argument("--save-artifacts", action="store_true", help="Keep block_ridge stage-2 coefficients (~218 MB a run).")
    s.add_argument("--force-cold", action="store_true", help="Allow a bulk submit against a cold feature cache.")
    s.add_argument("--dry-run", action="store_true", help="Print the submit commands and stop.")
    s.set_defaults(func=cmd_submit)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass, field


def average_scores(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class PerTaskAlphaTracker:
    """Tracks two independent per-task alpha-optimized accuracy streams.

    The "primary" stream is the result being reported (e.g. the rebased/transported
    task vector). The "secondary" stream is the normalizer (e.g. the untransported
    baseline). Each stream maintains its own best alpha, best accuracy, active flag
    and bad-step counter, and early-stops independently of the other.

    Backwards-compatibility aliases are exposed as read-only properties so callers
    that used the old single-stream API keep working:

      - best_alpha          -> best_primary_alpha
      - best_rebase_acc     -> best_primary_acc
      - active              -> primary_active (and active_indices() still exists)
      - bad_steps           -> primary_bad_steps
      - best_baseline_acc   -> best_secondary_acc (NOTE: this was previously the
                              baseline acc *at the primary's best alpha*; it now
                              tracks the baseline's *own* peak. Callers relying on
                              the old semantics must read secondary acc at the
                              primary's best alpha explicitly.)
    """

    task_names: list[str]
    initial_alpha: float = 0.0
    patience: int = 0

    # Primary stream
    best_primary_alpha: list[float] = field(init=False)
    best_primary_acc: list[float] = field(init=False)
    primary_active: list[bool] = field(init=False)
    primary_bad_steps: list[int] = field(init=False)

    # Secondary stream (independent best alpha for the normalizer)
    best_secondary_alpha: list[float] = field(init=False)
    best_secondary_acc: list[float] = field(init=False)
    secondary_active: list[bool] = field(init=False)
    secondary_bad_steps: list[int] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.task_names)
        a0 = float(self.initial_alpha)
        self.best_primary_alpha = [a0] * n
        self.best_primary_acc = [float("-inf")] * n
        self.primary_active = [True] * n
        self.primary_bad_steps = [0] * n
        self.best_secondary_alpha = [a0] * n
        self.best_secondary_acc = [float("-inf")] * n
        self.secondary_active = [True] * n
        self.secondary_bad_steps = [0] * n

    # ---------- primary stream ----------
    def primary_active_indices(self) -> list[int]:
        return [i for i, is_active in enumerate(self.primary_active) if is_active]

    # ---------- secondary stream ----------
    def secondary_active_indices(self) -> list[int]:
        return [i for i, is_active in enumerate(self.secondary_active) if is_active]

    def eval_active_indices(self) -> list[int]:
        """Union of primary- and secondary-active task indices, sorted ascending."""
        return sorted(set(self.primary_active_indices()) | set(self.secondary_active_indices()))

    def update(
        self,
        *,
        alpha: float,
        indices: list[int],
        primary_accs: list[float],
        secondary_accs: list[float],
    ) -> tuple[list[int], list[int]]:
        """Update both streams at one alpha step.

        Pass ``primary_accs[i] = float('-inf')`` for indices where the primary
        stream has already early-stopped and was not evaluated this step (legacy/
        no-op since those indices have ``primary_active[i] == False``). Same for
        ``secondary_accs``.

        Returns ``(stopped_primary, stopped_secondary)`` — task indices that
        early-stopped for the first time at this alpha step.
        """
        if len(indices) != len(primary_accs) or len(indices) != len(secondary_accs):
            raise ValueError("indices, primary_accs, and secondary_accs must have the same length.")

        stopped_primary: list[int] = []
        stopped_secondary: list[int] = []
        eps = 1e-12

        for idx, primary_acc, secondary_acc in zip(indices, primary_accs, secondary_accs, strict=True):
            primary_acc = float(primary_acc)
            secondary_acc = float(secondary_acc)

            # ---- primary stream ----
            if self.primary_active[idx]:
                best_primary = float(self.best_primary_acc[idx])
                if primary_acc > best_primary + eps:
                    self.best_primary_alpha[idx] = float(alpha)
                    self.best_primary_acc[idx] = primary_acc
                    self.primary_bad_steps[idx] = 0
                elif primary_acc + eps >= best_primary:
                    self.primary_bad_steps[idx] = 0
                else:
                    self.primary_bad_steps[idx] += 1
                    if self.primary_bad_steps[idx] > self.patience:
                        self.primary_active[idx] = False
                        stopped_primary.append(idx)

            # ---- secondary stream ----
            if self.secondary_active[idx]:
                best_secondary = float(self.best_secondary_acc[idx])
                if secondary_acc > best_secondary + eps:
                    self.best_secondary_alpha[idx] = float(alpha)
                    self.best_secondary_acc[idx] = secondary_acc
                    self.secondary_bad_steps[idx] = 0
                elif secondary_acc + eps >= best_secondary:
                    self.secondary_bad_steps[idx] = 0
                else:
                    self.secondary_bad_steps[idx] += 1
                    if self.secondary_bad_steps[idx] > self.patience:
                        self.secondary_active[idx] = False
                        stopped_secondary.append(idx)

        return stopped_primary, stopped_secondary

    def best_avg(self) -> float:
        vals = [v for v in self.best_primary_acc if v != float("-inf")]
        return average_scores(vals)

    # ---------- backwards-compat aliases ----------
    @property
    def best_alpha(self) -> list[float]:
        return self.best_primary_alpha

    @property
    def best_rebase_acc(self) -> list[float]:
        return self.best_primary_acc

    @property
    def active(self) -> list[bool]:
        return self.primary_active

    @property
    def bad_steps(self) -> list[int]:
        return self.primary_bad_steps

    @property
    def best_baseline_acc(self) -> list[float]:
        return self.best_secondary_acc

    def active_indices(self) -> list[int]:
        return self.primary_active_indices()
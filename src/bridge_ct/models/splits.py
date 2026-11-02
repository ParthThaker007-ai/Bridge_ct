"""Chronological splits with label-horizon purging.

A snapshot at cutoff ``c`` has labels that are only *known* once time ``c + horizon`` has passed. Training on
such a snapshot while evaluating at an earlier moment would leak the future. With ``purge=True`` a training
cutoff is kept only if ``c + horizon <= first cutoff of the phase being evaluated``.
"""
from __future__ import annotations

from typing import Dict, List


def make_splits(cutoffs: List[int], horizon: int, val_first: int, test_first: int, purge: bool = True) -> Dict[str, List[int]]:
    """Return ``train`` (fit for validation), ``val``, ``refit`` (fit for test = train + val) and ``test`` cutoffs.

    Raises ``ValueError`` when a phase would be empty, which tells you to widen the data window or shrink the
    horizon rather than silently evaluating on nothing.
    """
    cutoffs = sorted(cutoffs)
    val = [c for c in cutoffs if val_first <= c < test_first]
    test = [c for c in cutoffs if c >= test_first]

    def known(c: int, phase_start: int) -> bool:
        return (c + horizon <= phase_start) if purge else True

    train = [c for c in cutoffs if c < val_first and known(c, val_first)]
    refit = [c for c in cutoffs if c < test_first and known(c, test_first)]
    for name, part in (("train", train), ("val", val), ("refit", refit), ("test", test)):
        if not part:
            raise ValueError(f"split '{name}' is empty (cutoffs={cutoffs}, horizon={horizon}, "
                             f"val_first={val_first}, test_first={test_first}, purge={purge})")
    return {"train": train, "val": val, "refit": refit, "test": test}

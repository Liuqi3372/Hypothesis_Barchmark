from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceCoverage:
    ref_id: str
    units: frozenset[str]
    confidence_sum: float = 0.0


def exact_minimum_cover(
    required_units: list[str],
    references: list[ReferenceCoverage],
) -> tuple[list[str], set[str]]:
    """Return a minimum-cardinality reference cover using exact bitmask DP.

    Ties are resolved by greater summed mapping confidence and then stable input
    order. The second return value contains units no candidate can cover.
    """
    unit_order = list(dict.fromkeys(required_units))
    if not unit_order:
        return [], set()
    bit = {unit: 1 << index for index, unit in enumerate(unit_order)}
    target = (1 << len(unit_order)) - 1
    available_mask = 0
    candidates: list[tuple[ReferenceCoverage, int]] = []
    for reference in references:
        mask = 0
        for unit in reference.units:
            mask |= bit.get(unit, 0)
        if mask:
            candidates.append((reference, mask))
            available_mask |= mask
    uncovered = {
        unit for unit, unit_bit in bit.items() if not available_mask & unit_bit
    }
    achievable = target & available_mask
    # mask -> (selected reference indices, summed confidence)
    dp: dict[int, tuple[tuple[int, ...], float]] = {0: ((), 0.0)}
    for index, (reference, ref_mask) in enumerate(candidates):
        snapshot = list(dp.items())
        for mask, (chosen, score) in snapshot:
            new_mask = mask | ref_mask
            proposal = (chosen + (index,), score + reference.confidence_sum)
            current = dp.get(new_mask)
            if current is None or (len(proposal[0]), -proposal[1]) < (
                len(current[0]), -current[1]
            ):
                dp[new_mask] = proposal
    selected_indices, _ = dp.get(achievable, ((), 0.0))
    return [candidates[index][0].ref_id for index in selected_indices], uncovered


def removal_test(
    required_units: list[str],
    selected: list[ReferenceCoverage],
) -> list[dict]:
    """Prove inclusion-minimality by removing each selected reference once."""
    required = set(required_units)
    baseline_covered = set().union(*(reference.units for reference in selected)) if selected else set()
    baseline_missing = required - baseline_covered
    results = []
    for removed in selected:
        covered = set().union(*(
            reference.units for reference in selected if reference.ref_id != removed.ref_id
        )) if len(selected) > 1 else set()
        missing = required - covered
        newly_uncovered = sorted(missing - baseline_missing)
        results.append({
            "removed_ref_id": removed.ref_id,
            "still_complete": not missing,
            "newly_uncovered_unit_ids": newly_uncovered,
            "indispensable": bool(newly_uncovered),
        })
    return results

"""Observed action traces and small, explicit repair decisions.

Traces contain sensor observations and skill outcomes. Simulator truth is deliberately
excluded, so the same diagnostics can be used by the controller at deployment.
"""

from __future__ import annotations

from pai.cognition.contracts import GoalRelation, StateBelief


def classify_failure(stage: str, *, stalled: bool = False, notes: str = "",
                     lost_relations=(), changed=()) -> str:
    """Give a failed skill or placement a stable, actionable cause label."""
    if "dropped" in notes or "grasp failed" in notes or "could not lift" in notes:
        return "grasp_lost"
    if stage == "insert":
        return "insertion_stalled" if stalled else "insertion_unverified"
    if stage == "turn_held" and stalled:
        return "bolt_unchanged"
    if any(p == "on" for p, *_ in lost_relations):
        return "support_relation_lost"
    if stage in ("lowered", "opened", "released"):
        return "unstable_placement"
    if stalled and not changed:
        return "contact_stall"
    return "goal_unmet"


def stack_clearance_repairs(goal: GoalRelation, state: StateBelief) -> list[GoalRelation]:
    """Clear a subject from the top down before rebuilding an earlier stack level.

    The temporary bowl destination is already an executable relation in the arm
    vocabulary. Traversal uses only observed relations and protects against cycles.
    """
    if goal.predicate != "on":
        return []
    children: dict[str, list[str]] = {}
    for predicate, subject, target in state.relations:
        if predicate == "on" and target is not None:
            children.setdefault(target, []).append(subject)
    seen: set[str] = set()
    order: list[str] = []

    def visit(subject: str) -> None:
        if subject in seen:
            return
        seen.add(subject)
        for child in sorted(children.get(subject, [])):
            visit(child)
            if child != goal.object and child not in order:
                order.append(child)

    visit(goal.subject)
    return [GoalRelation("in", subject, "bowl") for subject in order]

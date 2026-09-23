"""Human-readable episode reports: the agent's own account of what happened, next to the truth.

This is the seed of the "agent explains itself" demo. Every claim in the report comes from the
agent's beliefs (subgoals reached, surprise, inferred cause); the ground truth from the event log
is printed beside it, so the explanation can be checked.
"""

from __future__ import annotations

from pai.memory.episodic import EpisodeRecord

CAUSE_TEXT = {
    "none": "nothing unusual happened",
    "push": "something pushed my arm",
    "heavier_object": "the object is heavier than I expected",
    "unknown": "something happened that I cannot explain yet",
}


def episode_report(rec: EpisodeRecord, dt: float = 0.1) -> str:
    lines = [f"Episode {rec.episode}: goal {rec.goal}: {'SUCCESS' if rec.success else 'FAILED'} "
             f"after {rec.steps} planning steps ({rec.steps * dt:.1f} s)."]
    if rec.fallbacks:
        lines.append(f"  Setbacks: I lost the object {rec.fallbacks} time(s) and went back to re-grasp it.")
    if rec.subgoal_times:
        steps = ", ".join(f"{name} at {t * dt:.1f} s" for name, t in rec.subgoal_times.items())
        lines.append(f"  Progress: {steps}.")
    peaks = [i for i, s in enumerate(rec.surprise) if s > rec.surprise_threshold]
    if peaks:
        top = max(peaks, key=lambda i: rec.surprise[i])
        lines.append(f"  Surprise: my predictions failed at {len(peaks)} steps, most at {top * dt:.1f} s "
                     f"(surprise {rec.surprise[top]:.0f}; spikes start at {rec.surprise_threshold:.0f}).")
    else:
        lines.append("  Surprise: my predictions held throughout.")
    cause = rec.inferred_cause or "none"
    if rec.cause_posterior:
        p = rec.cause_posterior[cause]
        detail = rec.cause_params.get(cause, {}) if rec.cause_params else {}
        extra = f" ({', '.join(f'{k}={v}' for k, v in detail.items())})" if detail else ""
        lines.append(f"  My explanation: {CAUSE_TEXT[cause]}, probability {p:.2f}{extra}.")
    else:
        lines.append(f"  My explanation: {CAUSE_TEXT['none']}.")
    truth = [f"{d['disturbance']} at {d['t'] * 0.02:.1f} s" for d in rec.true_disturbances]
    lines.append(f"  Ground truth: {', '.join(truth) if truth else 'no disturbance'}.")
    return "\n".join(lines)

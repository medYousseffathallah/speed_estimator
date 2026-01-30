Roadmap: Speed Estimation Pipeline Improvements

1. Dot-based speed + turning refactor (High impact)

- [ ] Fix zero-angle turning by validating dot spacing and thresholds
- [ ] Refactor compute_turning_metrics to accept dots only
- [ ] Remove legacy vector-based turning logic in estimator
- [ ] Ensure speed and turning both use the same dot buffer
- [ ] Enforce jitter mitigation via dot gating only
- [ ] Align overlay to display applied vs detected turns
- [ ] Audit thresholds and document recommended ranges
- [ ] Delete dead code, unused imports, and unused parameters
- [ ] Add debug-only logs for dots and turning decisions

2. Evaluation criteria (High impact)

- [ ] Turning angles appear only when cars actually turn
- [ ] Straight driving yields stable 0°
- [ ] Speed does not spike during jitter
- [ ] Code is readable and auditable by a human engineer

3. Constraints (Non-negotiable)

- [ ] No per-frame vector turning
- [ ] No EMA on angles or curvature
- [ ] No mixed dot + frame logic
- [ ] No silent fallback to old behavior

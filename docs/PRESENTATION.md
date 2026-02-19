# Presentation Structure

## Time allocation

- Introduction: **2-3 min**
- Propensity Model 1 (Churn): **~4 min**
- Propensity Model 2 (Redemption): **~4 min**
- Propensity Model 3 (Lifecycle Continuation): **~4 min**
- Buffer/close: **0.5-1 min**

Total: **~14.5-15 min**

## Slide-by-slide structure

### Slide 1 - Title and challenge objective (0:00-0:45)

- Team and project title
- Business objective: actionable propensity models for Pampers loyalty activation
- Why this matters (retention, reward usage, lifecycle transition)

### Slide 2 - Data foundation and method (0:45-2:30)

- Data assets used (users, access, scans, missions, redemptions, ISTAT)
- Canonical key architecture (`idSSO` + join bridge)
- Leakage-safe approach (rolling reference dates, future-window labels)

### Slide 3 - Model 1: Churn problem framing (2:30-3:15)

- Target: `churn_30_to_60`
- Population: users currently 30-59 days inactive
- Campaign goal: prevent progression into persistent inactivity

### Slide 4 - Model 1: Features, output, and actions (3:15-4:30)

- Main signals: recency/frequency, reward behavior, lifecycle, channel eligibility
- Output: `churn_30_to_60_prob`, risk decile, recommended campaign
- Activation logic: retention prioritization + holdout measurement

### Slide 5 - Model 1: Expected impact and KPI tracking (4:30-6:30)

- KPI set: reactivation rate, 30->60 progression reduction, incremental scans
- Monitoring: drift, calibration by lifecycle bucket, fatigue checks

### Slide 6 - Model 2: Redemption problem framing (6:30-7:15)

- Target: `will_redeem_30d`
- Population: points-engaged active users (`totalPoints > 0`, recent activity <=90d)
- Campaign goal: increase reward activation and loyalty

### Slide 7 - Model 2: Features, output, and campaign use (7:15-8:30)

- Main signals: points balance/velocity, redemption recency/history, engagement
- Output: `redeem_30d_prob`, activation segment, points-gap proxy
- Use cases: activation barriers vs loyalty reinforcement

### Slide 8 - Model 2: Measurement and guardrails (8:30-10:30)

- KPI set: redemption uplift, precision-at-k, incremental value
- Guardrails: eligibility gating, budget by decile, holdout testing

### Slide 9 - Model 3: Lifecycle continuation framing (10:30-11:15)

- Target: `will_scan_60d`
- Population: users age >=24 months with recent scans
- Goal: identify retain-vs-transition opportunity near diaper graduation

### Slide 10 - Model 3: Features, segments, and actions (11:15-12:30)

- Main signals: scan momentum, lifecycle stage, value interaction, regional context
- Output: `lifecycle_continuation_60d_prob`, lifecycle stage, transition action
- Segments: high-potential transition, at-risk transition, early transition prep

### Slide 11 - Model 3: Business application and cross-sell boundary (12:30-14:30)

- Transition strategy by segment
- Cross-sell claims constrained to observed category evidence
- KPI set: continuation uplift, transition segment response, ROI by segment

### Slide 12 - Final synthesis and next steps (14:30-15:00)

- How the 3 scores work together in one activation system
- Pilot plan: first campaign wave + monitoring cadence
- Optional Q&A if time remains

## Speaker notes (recommended)

- Keep each model section to exactly 4 minutes: 45 sec framing, 75 sec approach/output, 120 sec impact/KPIs.
- Use one visual per model: problem -> signals -> score -> action.
- Prioritize business decisions and campaign actions over algorithm details.

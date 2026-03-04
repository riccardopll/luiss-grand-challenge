# Intermediate Presentation Structure

## Purpose

This deck is aligned to `eda.ipynb` and covers only the intermediate phase:

- EDA evidence and data understanding
- Problem framing and proposed propensity solution
- Operational path to final ML delivery

Out of scope for this deck: final trained model performance, final calibrated thresholds, and final KPI/ROI values.

## Time allocation

- Title, scope, and objective: **1.5-2 min**
- EDA evidence and diagnostics: **7-8 min**
- Proposed solution framing: **4-4.5 min**
- Next steps and close: **1-1.5 min**

Total: **~14.5-15 min**

## Slide-by-slide structure

### Slide 1 - Title and challenge objective (0:00-0:45)

- Team and project title
- Business objective: actionable propensity system for Pampers loyalty activation
- Why this matters (retention, reward usage, lifecycle transition)

### Slide 2 - Intermediate scope and roadmap (0:45-1:30)

- What this presentation covers now: EDA + problem framing + proposed solution
- What is deferred to final delivery: trained model results and final business values
- Roadmap of sections

### Slide 3 - Dataset size by table (1:30-2:30)

- Data assets and volume concentration across sources
- Join-planning implications for feature engineering
- Why aggregation discipline and leakage-safe logic matter

### Slide 4 - Monthly active users trend (2:30-3:30)

- Engagement baseline over time
- Non-stationarity and seasonality implications
- Why temporal validation is mandatory

### Slide 5 - Monthly event intensity (3:30-4:30)

- Events per active user by behavior type
- Intensity variation independent from active-user volume
- Signal-family motivation for model features

### Slide 6 - Total points distribution (4:30-5:30)

- Distribution shape, concentration, and heavy-tail behavior
- Non-null scope and null-vs-zero interpretation
- Segmentation and robust-threshold implications

### Slide 7 - Lifecycle stage mix + regional user footprint (5:30-7:00)

- Audience composition by child-age buckets
- Regional concentration and socioeconomic context
- Why lifecycle and region should stay explicit in targeting logic

### Slide 8 - Monthly label dynamics (7:00-8:30)

- Volume and base-rate differences across targets
- Why each target is a distinct operational problem
- Calibration and monitoring implications

### Slide 9 - Proposed solution: three complementary propensity models (8:30-10:30)

- `churn_30_to_60`: prevent persistent inactivity progression
- `will_redeem_30d`: activate reward usage among eligible users
- `will_scan_60d`: guide retain-vs-transition lifecycle actions
- Why three models (different populations, behaviors, and base rates)

### Slide 10 - EDA-to-campaign summary map (10:30-12:30)

- Sankey view: EDA signals -> model decisions -> campaign levers
- Conceptual linkage only (intermediate phase)
- Holdout and governance as cross-model operating requirements

### Slide 11 - Next steps toward final presentation (12:30-14:30)

- Lock feature/label contracts and leakage checks
- Train/validate/calibrate model stack with temporal splits
- Define threshold policy by budget and channel constraints
- Final deck will report validated model and KPI outcomes

### Slide 12 - Close / Q&A (14:30-15:00)

- Recap: evidence, proposed approach, and execution path
- Open questions and feedback for final-phase refinement

## Speaker notes (recommended)

- Keep narrative anchored to `eda.ipynb` section names for consistency.
- Emphasize that all links in the summary map are EDA-grounded and directional, not final feature importances.
- Prioritize decision logic and campaign operating model over algorithmic detail at this stage.

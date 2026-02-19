# Propensity Model 2: Redemption (next 30 days)

## 1. Problem definition

Predict whether a user will redeem points in the 30 days after a given reference date.

- Prediction target: `will_redeem_30d`
- Business goal: maximize reward activation and loyalty with budget-aware targeting.

## 2. Population and scoring strategy

## 2.1 Scoring unit

- One row per (`idSSO`, `reference_date`)

## 2.2 Primary population (campaign-facing)

Include users with:

1. `totalPoints > 0`
2. Recent activity recency <=90 days (`last_activity` from login/scan/mission)

Rationale:

- This removes structurally inactive users from activation priority.
- It aligns model output with operationally reachable reward activation cohorts.

## 2.3 Optional benchmark population

- Broad-universe benchmark can be trained/scored on all users for comparison.
- Campaign execution should continue to use the primary population.

## 3. Label definition

`will_redeem_30d = 1` if at least one redemption occurs in `(reference_date, reference_date + 30 days]`.

`will_redeem_30d = 0` otherwise.

Source:

- `premi_reduced.datarichiestapremio`

Leakage rule:

- Features only from data `<= reference_date`.

## 4. Canonical source mapping and schema fixes

## 4.1 User key mapping

- Master key: `idSSO`
- `premi.userid` and `codici.userId` are mapped via `rewusers.userId -> rewusers.idSSO`.
- `accessi.idsso` joins directly to `idSSO`.
- `missioni.idsso` joins directly to `idSSO`.

## 4.2 Accessi schema correction

- User identifier is `idsso` (not `Unnamed: 0`).
- Platform values are `app` and `sito`.

## 5. Feature set

## 5.1 Points and reward behavior

- `totalPoints` (raw)
- `is_points_user`
- `num_past_redemptions`
- `days_since_last_redemption`
- `avg_points_per_redemption`
- `points_redeemed_lifetime`
- `points_earned_lifetime_proxy = totalPoints + points_redeemed_lifetime`

Null policy:

- Do not globally impute `totalPoints` to zero in canonical training data.
- Use explicit missingness signal via `is_points_user`.

## 5.2 Activity and momentum

- `scan_frequency_30d`, `scan_frequency_60d`, `days_since_last_scan`
- `login_frequency_30d`, `login_frequency_60d`, `days_since_last_login`
- `app_vs_sito_ratio`
- `mission_completion_rate`, `num_missions_completed`

## 5.3 Profile and lifecycle context

- `tenure_days`
- `ETA_MM_BambinoTODAY`
- lifecycle bucket features
- consent/channel eligibility flags

## 5.4 Regional socioeconomic context

- `regional_median_income_eur`
- `share_families_with_children_pct` (direct from ISTAT `AVTYPE_HYESCHI` + `HSC_N`)

## 6. Data-grounded baselines and class behavior

Observed behavior indicates strong imbalance and seasonal variation.

- Across monthly snapshots, broad-universe prevalence is low (around low single digits on most months).
- A spike period exists (notably around 2025-01 snapshots), so calibration must be time-aware.

Operational implication:

- Use rolling temporal validation and monthly calibration checks.

## 7. Modeling approach

1. Baseline: Logistic Regression
2. Champion candidates: LightGBM / XGBoost
3. Class imbalance: class weights and threshold optimization on precision-at-k / uplift objective
4. Validation: rolling time splits only

## 8. Campaign activation logic (complete)

## 8.1 Primary use case: Reward Activation

Target users with high activation upside, for example:

- `totalPoints` above campaign threshold
- no/low historical redemptions
- low-to-mid predicted redemption probability (barrier-removal cohort)

Action examples:

- points gap reminders
- limited-time multiplier
- shipping/contribution reduction where applicable

## 8.2 Secondary use case: Loyalty Reinforcement

Target already engaged users with high redemption propensity and strong engagement.

Action examples:

- premium reward previews
- mission bundles tied to reward tiers

## 8.3 Guardrails

- Require channel eligibility before outbound communication.
- Preserve holdout groups for causal measurement.
- Enforce budget caps by score decile.

## 9. Output contract

Each scored row must include:

- `idSSO`
- `reference_date`
- `redeem_30d_prob`
- `activation_segment`
- `points_gap_proxy`

Example `activation_segment` values:

- `activation_barrier`
- `ready_to_redeem`
- `loyalty_reinforcement`
- `not_campaign_eligible`

## 10. Monitoring and acceptance checks

### 10.1 Monitoring

- Monthly calibration and drift by acquisition cohort and lifecycle stage.
- Precision-at-k stability by month.
- Campaign conversion uplift vs holdout.

### 10.2 Acceptance criteria

1. Canonical key mapping implemented exactly (`idSSO` contract).
2. Accessi schema used correctly (`idsso`, `app`/`sito`).
3. Family context feature uses direct ISTAT indicator (`AVTYPE_HYESCHI` + `HSC_N`).
4. No incomplete sections or truncated campaign logic.
5. Output schema is complete and operationally actionable.

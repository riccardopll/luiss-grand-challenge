# Propensity Model 1: Churn (30 -> 60 day inactivity)

Scope note:

- Intermediate presentation: this model is introduced conceptually (problem, population, action logic).
- Final presentation: training results, calibrated thresholds, and KPI impact estimates are reported with validated outputs.

## 1. Problem definition

Predict whether a user currently in the 30-59 day inactivity window will stay inactive for the next 30 days.

- Business question: who is progressing toward persistent inactivity?
- Prediction target: `churn_30_to_60`

## 2. Population and scoring unit

### 2.1 Scoring unit

- One row per (`idSSO`, `reference_date`)

### 2.2 Eligible population

At `reference_date`, include users who:

1. Have at least one historical activity event (login, scan, or mission) on or before `reference_date`.
2. Have `days_since_last_activity` in `[30, 59]`.

Where:

- `last_activity = max(last_login, last_scan, last_mission)`
- Activity is built from:
  - `accessi_reduced.created_at` (via `idsso`)
  - `codici_reduced.created_at` (via `userId -> rewusers bridge -> idSSO`)
  - `missioni_reduced.created_at` (via `idsso`)

### 2.3 Explicit exclusions

- Users with no activity history up to `reference_date`.
- Rows where required key mapping to `idSSO` fails (should be zero for mapped event tables).

## 3. Label definition

`churn_30_to_60 = 1` if the user has **no** activity in `(reference_date, reference_date + 30 days]`.

`churn_30_to_60 = 0` otherwise.

Leakage rule:

- Features use only events `<= reference_date`.
- Label uses only events in the future window.

### 3.1 CRM interpretation layer

This model predicts **overall risk of persistent inactivity** only.

- The model output is `churn_30_to_60_prob`.
- The distinction between `preventable_churn` and `physiological_transition` is made later by the CRM decision engine, not by the churn model itself.

The CRM decision layer should derive that interpretation from:

- lifecycle position (`ETA_MM_BambinoTODAY`, `child_age_bucket`, `is_near_graduation`)
- recent activity momentum
- reward/value engagement (`totalPoints`, redemption history)
- tenure and channel eligibility

## 4. Feature set

## 4.1 Recency and frequency momentum

- `days_since_last_login`
- `days_since_last_scan`
- `days_since_last_mission`
- `days_since_last_activity`
- `login_count_30d`, `login_count_60d`
- `scan_count_30d`, `scan_count_60d`
- `missions_completed_30d`, `missions_completed_60d`
- `scan_velocity_60d` (difference between recent 30d vs previous 30d scan counts)

## 4.2 Reward and value behavior

- `has_ever_redeemed`
- `days_since_last_redeem`
- `redeem_count`
- `avg_points_spent_per_redeem`
- `totalPoints` (raw)
- `is_points_user`

## 4.3 Lifecycle and profile context

- `ETA_MM_BambinoTODAY`
- `child_age_bucket` (pregnancy, 0-11, 12-23, 24-29, 30-35)
- `is_near_graduation` (>=30)
- `months_to_transition_proxy = 36 - ETA_MM_BambinoTODAY`
- `tenure_days`
- `platform`

## 4.4 Channel eligibility and region context

- `channel_eligible_email` from `privacy_marketing_email` + `double_opt_in`
- `channel_eligible_phone_push` from `privacy_marketing_telefono`
- `regional_median_income_eur`
- `share_families_with_children_pct`

## 5. Data-grounded operating baselines

From rolling monthly snapshots (2024-04-01 to 2025-08-01):

- Average eligible users per snapshot: ~342
- Mean positive rate (`churn_30_to_60 = 1`): ~64.06%

These are baseline diagnostics, not fixed business targets.

## 6. Modeling and thresholding

### 6.1 Model stack

1. Baseline: Logistic Regression
2. Champion candidates: LightGBM / XGBoost

### 6.2 Class imbalance and calibration

- Use class weights for all primary models.
- Tune thresholds on precision-recall tradeoff by campaign budget.
- Validate calibration by decile and age bucket.

### 6.3 Temporal validation

- Train/validate only with time-based splits on `reference_date`.
- No random split as primary method.

## 7. Output contract

Each scored row must include:

- `idSSO`
- `reference_date`
- `churn_30_to_60_prob`
- `risk_decile`

### 7.1 CRM decision-layer outputs

The CRM decision engine can consume the scored output and produce:

- `churn_type`
- `recommended_campaign`
- `recommended_channel`
- `priority`

Example `churn_type` values:

- `preventable_churn`
- `physiological_transition`
- `mixed_or_uncertain`

Recommended campaign values can include:

- `preventable_churn_core`
- `reward_threshold_nudge`
- `low_roi_transition_guardrail`
- `insufficient_channel_eligibility`

## 8. Activation playbook

1. Score eligible 30-59 day inactive users daily/weekly.
2. Rank by `churn_30_to_60_prob` descending.
3. Use the CRM decision layer to classify high-risk users into `preventable_churn` vs `physiological_transition` vs `mixed_or_uncertain`.
4. Apply eligibility filters (consent/channel).
5. Allocate budget primarily to preventable churn cohorts; use guardrails for likely physiological transition.
6. Reserve holdout group for uplift measurement.

Primary campaign KPIs:

- Reactivation rate at 7/14/30 days.
- Reduction in 30->60 day progression to persistent inactivity.
- Incremental scans and points earned vs holdout.

## 9. Monitoring and acceptance checks

### 9.1 Monitoring

- Score drift and feature drift (monthly).
- Calibration drift by lifecycle bucket.
- Stability of CRM `churn_type` assignment by lifecycle bucket.
- Intervention fatigue by repeated exposure.

### 9.2 Acceptance criteria

1. Every documented feature maps to existing source columns or defined engineered transforms.
2. Key mapping follows canonical bridge (`userId -> idSSO`) with documented match rates.
3. No label leakage.
4. Predictive output schema is complete and campaign-ready.
5. CRM decision-layer outputs are explicitly defined for preventable-vs-physiological churn segmentation.

# Propensity Model 3: Lifecycle Continuation (next 60-day scanning)

## 1. Problem definition

This is the canonical third model: **Lifecycle Continuation**.

Predict whether users in late lifecycle will continue scanning in the next 60 days.

- Prediction target: `will_scan_60d`
- Business objective: distinguish users worth retention/cross-sell investment during diaper-transition stages.

## 2. Population and scoring unit

## 2.1 Scoring unit

- One row per (`idSSO`, `reference_date`)

## 2.2 Eligible population

At `reference_date`, include users who satisfy:

1. `ETA_MM_BambinoTODAY >= 24`
2. Recent scan activity (`days_since_last_scan <= 90`)

Rationale:

- Focuses model on behavior continuation rather than dormant-user reactivation.
- Keeps target operationally relevant for transition programs.

## 2.3 Data range caveat

- Current extract has max observed child age = **35** months.
- `36+` lifecycle stage remains conceptual/future and is treated as out-of-sample for current training KPIs.

## 3. Label definition

`will_scan_60d = 1` if user has at least one scan in `(reference_date, reference_date + 60 days]`.

`will_scan_60d = 0` otherwise.

Source:

- `codici_reduced.created_at` (mapped to `idSSO` via `rewusers.userId -> rewusers.idSSO`)

Leakage control:

- Features strictly from data `<= reference_date`.

## 4. Lifecycle staging

Canonical lifecycle buckets:

- `pregnancy` (`ETA_MM_BambinoTODAY < 0`)
- `0-11`
- `12-23`
- `24-29`
- `30-35`
- `36+` (future/out-of-sample in current extract)

Lifecycle features to include:

- `ETA_MM_BambinoTODAY`
- `months_to_transition_proxy = 36 - ETA_MM_BambinoTODAY`
- `is_near_graduation` (>=30)

## 5. Feature set

## 5.1 Engagement momentum

- `scan_count_30d`, `scan_count_60d`, `scan_count_90d`
- `days_since_last_scan`
- `scan_velocity_60d` and `scan_velocity_90d`
- `login_count_30d`, `days_since_last_login`
- `mission_activity_60d`

## 5.2 Value interaction

- `totalPoints` (raw)
- `is_points_user`
- `points_earned_90d`
- `has_ever_redeemed`, `days_since_last_redeem`
- `redeem_count`

## 5.3 Tenure, platform, and channel

- `tenure_days`
- `platform`
- `channel_eligible_email`
- `channel_eligible_phone_push`

## 5.4 Regional context

- `regional_median_income_eur`
- `share_families_with_children_pct`

## 6. Data-grounded baseline diagnostics

For a representative late-period snapshot (2025-08-01), with the canonical population:

- Eligible users (`age >=24` and recent scan <=90d): 677
- Positive rate (`will_scan_60d = 1`): ~80.65%

By age bucket in the same snapshot:

- `24-29`: higher continuation rate
- `30-35`: lower continuation rate than `24-29`, indicating transition risk increase

These are diagnostic anchors, not fixed targets.

## 7. Segment outputs and actions

Required lifecycle segment logic:

1. `high_potential_transition`
   - Age 30-35, high continuation probability
   - Action: premium missions/cross-category upsell
2. `at_risk_transition`
   - Age 30-35, low continuation probability
   - Action: retention nudges and value-focused incentives
3. `early_transition_prep`
   - Age 24-29 with medium/high continuation
   - Action: education + gradual transition communication

## 8. Cross-sell claim boundary

Cross-sell recommendations must stay within available evidence:

- Use observed product/category signals from scan data (`EAN`, `infoCode`) and product catalog mapping.
- Do not claim unsupported category intent outside covered signals.

## 9. Modeling and validation

1. Baseline: Logistic Regression
2. Champion candidates: LightGBM / XGBoost
3. Validation: rolling temporal splits with fixed 60-day forward windows
4. Calibration checks by lifecycle bucket

## 10. Output contract

Each scored row must include:

- `idSSO`
- `reference_date`
- `lifecycle_continuation_60d_prob`
- `lifecycle_stage`
- `transition_action`

## 11. Monitoring and acceptance checks

### 11.1 Monitoring

- Drift in continuation probability by lifecycle stage.
- Calibration by age bucket (`24-29` vs `30-35`).
- Campaign response by transition segment.

### 11.2 Acceptance criteria

1. Population filter is exactly `age >=24` and `days_since_last_scan <= 90`.
2. Label window is exactly 60 days, leakage-safe.
3. `36+` stage is explicitly marked out-of-sample for current data.
4. Segment outputs are present and campaign-ready.

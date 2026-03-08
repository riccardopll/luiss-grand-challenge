# Fater Grand Challenge 2025

**Goal:** Build a unified full propensity model for the Fater app and define the CRM decision engine that activates it through targeted campaigns.

## 0. Delivery phasing

The project is delivered in two phases:

1. **Intermediate phase (current):** EDA, problem framing, and proposed solution design.
   - Primary artifacts:
     - `eda.ipynb` (full-context exploratory notebook)
     - `docs/PRESENTATION.md` (intermediate deck structure)
2. **Final phase:** full model training/validation, calibrated operating thresholds, campaign design, and KPI/ROI reporting.
   - Primary artifacts:
     - model outputs per propensity
     - evaluation and calibration report
     - final campaign operating recommendations and audience rules

## 1. Objective and business framing

Build a unified propensity system for Fater loyalty users that supports campaign decisions with two complementary scores.
This should be presented as one integrated solution with:

- a **full propensity model** as the predictive core
- a **CRM decision engine** as the campaign activation layer

The two scores below are the predictive inputs used by that decision engine, not unrelated models:

1. `churn_30_to_60_prob`: risk that a user currently 30-59 days inactive will remain inactive for the next 30 days.
2. `redeem_30d_prob`: likelihood of redeeming points in the next 30 days.

### 1.1 Naming: CRM decision engine vs. full propensity model

- We call the **predictive component** a **full propensity model** because it estimates future behavior probabilities for each user at a given `reference_date`.
- We call the **business solution** a **CRM decision engine** because CRM needs more than prediction: it needs rules for **who to contact, when, in which channel, with which action, and under which eligibility or budget constraints**.
- The difference is:
  - the **full propensity model** answers: "What is likely to happen?"
  - the **CRM decision engine** answers: "What should CRM do given that prediction?"
- Therefore, we do **not** name the whole solution only as a full propensity model, because that would describe the scoring layer but not the operational campaign logic.
- In this project, the correct framing is: **the full propensity model is the analytical core; the CRM decision engine is the activation layer built on top of it**.

### 1.2 Challenge Requirements

- Build a unified propensity system
- Segment users and identify behavioral patterns
- Propose targeted marketing campaigns by segment
- Deliverables (phased):
  - Intermediate: EDA narrative + solution framing presentation
  - Final: model results, Python pipelines, and campaign proposal with validated metrics

### 1.3 Business interpretation

- A propensity model is the probability of a specific future behavior for a user at a given `reference_date`.
- A full propensity model can include multiple complementary propensities, as in this project, when each score corresponds to a different CRM decision moment.
- A CRM decision engine consumes those propensity scores together with business rules, constraints, and channel logic to convert predictions into campaign actions.
- In this project, `churn_type` is a **CRM decision-layer output**, not a separate predictive target: the CRM engine uses `churn_30_to_60_prob` plus lifecycle and engagement context to distinguish likely **physiological/natural churn** from **preventable churn**.
- The final deliverable is not the scores alone; it is one or more campaign designs that use those scores to decide who to target, why, with which action, in which channel, and how to measure impact.
- User overlap across scores is expected. The same user can be simultaneously points-engaged and at risk of churn; lifecycle remains a supporting context variable inside the CRM decision layer.

### 1.4 Simplified operating design

To keep the submission clear and not over-engineered:

- Keep **two predictive scores** because they support the two clearest CRM decisions in scope: churn prevention and reward activation.
- Keep **one simple CRM decision engine** on top of those scores rather than adding extra predictive models for churn type, lifecycle continuation, or next-best product.
- Treat `churn_type` (`preventable_churn` vs `physiological_transition`) as a **decision-layer classification**, not as an additional model output.
- Keep lifecycle as a **supporting feature and business-rule input**, not as a third predictive target, so the submission stays easier to explain and validate.

### 1.5 CRM decision engine example

Example inputs to the CRM decision engine:

- Predictive inputs:
  - `churn_30_to_60_prob = 0.81`
  - `redeem_30d_prob = 0.18`
- Non-predictive inputs:
  - `ETA_MM_BambinoTODAY = 32`
  - `is_near_graduation = 1`
  - `totalPoints = 420`
- `channel_eligible_push = 1`
- `days_since_last_contact = 12`

Example decision logic:

- If `churn_30_to_60_prob >= 0.75`
- and `is_near_graduation = 1`
- and `channel_eligible_push = 1`
- then classify the user as more likely `physiological_transition`
- and assign `recommended_action = transition_guardrail`
- otherwise, if `churn_30_to_60_prob >= 0.75` and the user is not near graduation, classify as `preventable_churn`
- otherwise, if `redeem_30d_prob` is low and the user has points, assign a reward-activation action

Example CRM decision output:

- `user_id = 12345`
- `churn_type = physiological_transition`
- `recommended_action = transition_guardrail`
- `recommended_channel = push`
- `priority = medium`
- `reason = high churn risk, late lifecycle, push eligible`

## 2. Data inventory and observed quality constraints

### 2.1 Core tables

| Table                             |    Rows | Notes                                            |
| --------------------------------- | ------: | ------------------------------------------------ |
| `rewusers_reduced.csv`            |   6,205 | User-level profile, points, app metadata         |
| `utenti_reduced.csv`              |   6,205 | Demographics, child age, region, privacy/consent |
| `accessi_reduced.csv`             | 547,174 | Login events (`app`/`sito`)                      |
| `codici_reduced.csv`              | 162,045 | Product scan events and scan points              |
| `missioni_reduced.csv`            |  45,447 | Mission events and statuses                      |
| `premi_reduced.csv`               |   2,759 | Reward redemptions                               |
| `Anagrafica_prodotti_digital.csv` |   6,078 | Product catalog metadata                         |

### 2.2 Coverage and data quality facts

- Behavioral coverage by user:
  - Access events: **85.98%**
  - Scan events: **57.79%**
  - Mission events: **58.02%**
  - Redemption events: **28.80%**
- `rewusers.lastActivity` is fully null and must not be used for behavior recency.
- `rewusers.totalPoints` is null for **52.41%** of users; null is a meaningful signal.
- For the **Total Points Distribution** graph, use only non-null `totalPoints` users (`n=2,953`) and do not impute null to zero; `941/3,252` null-balance users (`28.9%`) still show reward activity (scan or redeem), so null is not equivalent to known zero.
- `rewusers.userId` and `rewusers.idSSO` are a 1:1 bridge.
- Data horizon:
  - Latest activity date: **2025-10-06**
  - Latest redemption date: **2025-10-03**
  - Latest ISTAT income year: **2023**
  - Latest ISTAT family composition year: **2024**
- Child age facts:
  - Observed max `ETA_MM_BambinoTODAY`: **35** months
  - Pregnancy values exist (`<0` months)

## 3. Canonical key architecture and join map

Master key is `idSSO`.

### 3.1 Key contract

- Direct `idSSO` sources:
  - `utenti_reduced.idSSO`
  - `accessi_reduced.idsso`
  - `missioni_reduced.idsso`
  - `rewusers_reduced.idSSO`
- Bridge sources:
  - `codici_reduced.userId` -> map via `rewusers_reduced.userId -> rewusers_reduced.idSSO`
  - `premi_reduced.userid` -> map via `rewusers_reduced.userId -> rewusers_reduced.idSSO`

### 3.2 Textual join diagram

1. Start from `utenti_reduced` as the user universe (`idSSO`).
2. Join `rewusers_reduced` on `idSSO`.
3. Aggregate `accessi_reduced` by `idsso` and join to `idSSO`.
4. Aggregate `missioni_reduced` by `idsso` and join to `idSSO`.
5. Map `codici_reduced.userId` to `idSSO` through `rewusers` bridge; aggregate and join.
6. Map `premi_reduced.userid` to `idSSO` through `rewusers` bridge; aggregate and join.

## 4. Shared feature backbone

All three models use a shared, leakage-safe feature layer built as of `reference_date`.

### 4.1 Feature families

- Recency/frequency:
  - `days_since_last_login`, `days_since_last_scan`, `days_since_last_mission`, `days_since_last_activity`
  - counts in trailing 30/60/90 day windows
- Points and value interaction:
  - `totalPoints` (raw), `is_points_user`, points earned windows, redemption history and recency
- Lifecycle:
  - `ETA_MM_BambinoTODAY`, lifecycle buckets, `is_near_graduation`
- Engagement breadth:
  - mission completion behavior, app/site usage mix
- Channel eligibility:
  - email and phone/push eligibility from privacy flags
- Regional socioeconomic context:
  - income and family-with-children indicators from ISTAT

### 4.2 Null handling policy

- Keep raw `totalPoints` as-is.
- Add explicit `is_points_user = 1` if `totalPoints` is not null and >0, else 0.
- Optional imputed numeric variant can be used for linear baselines only; canonical feature set preserves raw missingness.

## 5. ISTAT enrichment rules and parser safety

### 5.1 Parser safety

ISTAT files contain ragged rows due to note fields and embedded commas/quotes.

Canonical extraction rule:

- Select stable leading columns and required dimensions only.
- Ignore `NOTE_*` and trailing overflow fields.
- Required columns for enrichment:
  - `FREQ`, `REF_AREA`, `Territorio`, `DATA_TYPE`, `TIME_PERIOD`, `Osservazione`

### 5.2 Canonical enrichment fields

- Income feature: `regional_median_income_eur`
  - Source: `Regioni e tipo di comune (IT1,32_292_DF_DCCV_REDNETFAMFONTERED_9,1.0).csv`
  - Filter: `DATA_TYPE = REDD_MEDIANO_FAM`, `IMPUTED_RENTS = 1`, `FAM_MAIN_INCOME_SOURCE = 9`, latest year (2023)
- Family feature: `share_families_with_children_pct`
  - Source: `Tipologie familiari - regioni e tipo comune (IT1,82_87_DF_DCCV_AVQ_FAMIGLIE_12,1.0).csv`
  - Direct indicator: `DATA_TYPE = AVTYPE_HYESCHI`, `MEASURE = HSC_N`, latest year (2024)

### 5.3 Region mapping rules

- Normalize case, accents, punctuation, and spacing for region names.
- Special mappings:
  - `VALLE D'AOSTA` <-> `Valle d'Aosta / Vallee d'Aoste`
  - `TRENTINO-ALTO ADIGE` <- aggregate from Bolzano and Trento in income source
- Expected enrichment feasibility: **99.53%** matched users for both key regional features (unmatched users have null `Regione`).

## 6. Unified training strategy

### 6.1 Snapshot entity

Canonical training entity across models:

- One row per (`idSSO`, `reference_date`)
- Features computed using events `<= reference_date`
- Labels computed strictly in forward windows `(reference_date, reference_date + horizon]`

### 6.2 Time split and leakage controls

- Rolling monthly `reference_date` schedule: **2024-04-01 to 2025-08-01**
- Use temporal splits only (no random split as primary evaluation).
- No future events in feature construction.
- No proxy columns that directly encode future outcomes.

### 6.3 Modeling family

- Baseline: Logistic Regression
- Champion candidates: LightGBM / XGBoost
- Optional benchmark: Random Forest
- Class imbalance: class weights + threshold optimization on business objective

## 7. Campaign operating model across the 2 propensities

1. **Churn prevention**: prioritize users with high `churn_30_to_60_prob`, then let the CRM decision layer classify likely preventable churn versus likely physiological transition before activation.
2. **Reward activation**: among points-engaged users, prioritize low `redeem_30d_prob` users with high activation upside.

Combined execution principles:

- Apply channel eligibility gates before activation.
- Reserve holdout groups for all campaign families.
- Use score deciles for targeting policy and budget control.

## 8. Delivery checklist and challenge artifacts

### 8.1 Intermediate phase artifacts (EDA + framing)

1. `eda.ipynb` containing:
   - EDA charts and diagnostics
   - explicit 2-model framing
   - EDA-to-campaign summary map
   - next-step handoff section
2. Intermediate presentation structure in `docs/PRESENTATION.md`, aligned to notebook sections.

### 8.2 Final phase artifacts (modeling + activation)

Deliverables to produce from this specification:

1. Reproducible feature/label pipeline with documented joins and filters.
2. Two scored outputs with canonical predictive schemas:
   - Churn: (`idSSO`, `reference_date`, `churn_30_to_60_prob`, `risk_decile`)
   - Redemption: (`idSSO`, `reference_date`, `redeem_30d_prob`, `points_gap_proxy`)
3. CRM decision-engine outputs with activation fields, for example:
   - Churn decision: (`idSSO`, `reference_date`, `churn_type`, `recommended_campaign`, `recommended_channel`, `priority`)
   - Redemption decision: (`idSSO`, `reference_date`, `activation_segment`, `recommended_action`, `recommended_channel`, `priority`)
4. Evaluation report with temporal validation and calibration checks.
5. Campaign proposal with segmentation logic, KPI definitions, and holdout methodology.

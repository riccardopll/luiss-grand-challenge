# Canonical Propensity Program Specification

## 1. Objective and business framing

Build a unified propensity system for Pampers loyalty users that supports campaign decisions with three complementary scores:

1. `churn_30_to_60_prob`: risk that a user currently 30-59 days inactive will remain inactive for the next 30 days.
2. `redeem_30d_prob`: likelihood of redeeming points in the next 30 days.
3. `lifecycle_continuation_60d_prob`: likelihood that a user in late lifecycle will continue scanning in the next 60 days.

Model 3 is canonicalized as **Lifecycle Continuation** (a lifecycle-scoped re-engagement model).

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

## 7. Campaign operating model across the 3 propensities

1. **Churn prevention**: prioritize users with high `churn_30_to_60_prob` and campaign eligibility.
2. **Reward activation**: among points-engaged users, prioritize low `redeem_30d_prob` users with high activation upside.
3. **Lifecycle transition**: use `lifecycle_continuation_60d_prob` with lifecycle stage to separate retain-vs-transition actions.

Combined execution principles:

- Apply channel eligibility gates before activation.
- Reserve holdout groups for all campaign families.
- Use score deciles for targeting policy and budget control.

## 8. Delivery checklist and challenge artifacts

Deliverables to produce from this specification:

1. Reproducible feature/label pipeline with documented joins and filters.
2. Three scored outputs with canonical schemas:
   - Churn: (`idSSO`, `reference_date`, `churn_30_to_60_prob`, `risk_decile`, `recommended_campaign`)
   - Redemption: (`idSSO`, `reference_date`, `redeem_30d_prob`, `activation_segment`, `points_gap_proxy`)
   - Lifecycle: (`idSSO`, `reference_date`, `lifecycle_continuation_60d_prob`, `lifecycle_stage`, `transition_action`)
3. Evaluation report with temporal validation and calibration checks.
4. Campaign proposal with segmentation logic, KPI definitions, and holdout methodology.

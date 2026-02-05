# Fater Grand Challenge 2025 - Propensity Modeling Plan

**Goal:** Build actionable propensity models and targeted campaigns for the Pampers app using behavioral + profile data.

## 1. Challenge Requirements

- Build a full propensity model or focus on churn prediction
- Segment users and identify behavioral patterns
- Propose targeted marketing campaigns by segment
- Deliverables: model results, presentation, Python scripts, campaigns proposal

## 2. Dataset Overview

| Dataset             | Records | What It Contains                          |
| ------------------- | ------- | ----------------------------------------- |
| rewusers_reduced    | 6,205   | User profiles, points, app engagement     |
| utenti_reduced      | 6,205   | Demographics, child age, privacy consents |
| accessi_reduced     | 547,174 | App/web login events                      |
| codici_reduced      | 162,045 | Product code scans (points earned)        |
| missioni_reduced    | 45,447  | Mission completions                       |
| premi_reduced       | 2,759   | Reward redemptions                        |
| Anagrafica_prodotti | 6,078   | Product catalog                           |

**External ISTAT data available (all downloaded from https://esploradati.istat.it/databrowser):**

- **Household income (territorial)**: `Regioni e tipo di comune (IT1,32_292_DF_DCCV_REDNETFAMFONTERED_9,1.0).csv` (median family income by territory and year).
- **Family composition (territorial)**: `Tipologie familiari - regioni e tipo comune (IT1,82_87_DF_DCCV_AVQ_FAMIGLIE_12,1.0).csv` (shares/counts of family types, household size, households with/without children).
- **Household spending by family type (national)**: `Tipologia familiare (IT1,31_740_DF_DCCV_SPEMEFAM_COICOP_2018_4,1.0).csv`.
- **Spending by COICOP category (national)**: `Voce di spesa (Coicop 2018) (IT1,31_740_DF_DCCV_SPEMEFAM_COICOP_2018_8,1.0).csv`.
- **Spending habit changes since 2018 (national)**: `Cambiamenti nelle abitudini di spesa (dal 2018) (IT1,31_740_DF_DCCV_SPEMEFAM_COICOP_2018_1,1.0).csv`.
- **Expenditure quintiles (national)**: `Regioni e tipo di comune (IT1,31_740_DF_DCCV_SPEMEFAM_COICOP_2018_11,1.0).csv`.

## 3. Modeling Strategy (Three Propensity Models)

| Model         | Target Variable                   | Business Use         |
| ------------- | --------------------------------- | -------------------- |
| Churn         | Inactive within 60 days           | Retention campaigns  |
| Redemption    | Will redeem points                | Reward activation    |
| Re-engagement | Will scan product in next 30 days | Purchase stimulation |

**Algorithms:**

- Logistic Regression (baseline, interpretable)
- Random Forest (robust, feature importance)
- LightGBM/XGBoost (primary model for tabular performance)

**Interpretability:** SHAP for tree models; lift charts for business impact.

**Imbalance handling:** class weights + PR curve threshold tuning; SMOTE only if necessary.

## 4. Feature Families

- **RFM:** recency of login/scan, frequency of events, total points
- **Lifecycle:** child age, months until diaper graduation (36 - age months)
- **Engagement:** app vs web ratio, mission completion rate
- **Points:** velocity, distance to reward threshold, earned vs redeemed ratio
- **Regional socioeconomic (ISTAT):** median family income (territory-level)
- **Regional family composition (ISTAT):** share of households with children, single-person households, large-family share, average household size
- **Consumption context (ISTAT, national):** COICOP spending shares, spending-habit shifts since 2018 (use as time covariates or presentation context)

## 5. External Data Enrichment

- Join `Regione` (from `utenti_reduced.csv`) to ISTAT territorial datasets; map region → macro-area where needed
- Use most recent year (or align by year of user activity if building time-based features)
- National-only tables (COICOP, spending habits, quintiles) are not geographic; use as global time covariates or presentation context
- Some files include note fields with embedded commas; load only core columns (e.g., `FREQ`, `REF_AREA`, `Territorio`, `DATA_TYPE`, `TIME_PERIOD`, `Osservazione`, plus dimensions) and ignore `NOTE_*` columns
- Current ISTAT set does not include municipality/province-level demographics (e.g., birth rates, child population); geographic features are limited to region/macro-area
- All ISTAT files used here come from https://esploradati.istat.it/databrowser

## 6. Targeted Campaign Proposals

| Campaign              | Trigger                              | Channel        | Incentive / Message                      |
| --------------------- | ------------------------------------ | -------------- | ---------------------------------------- |
| Churn Prevention      | 30+ days inactive + high churn score | Push + Email   | Double points; remind of rewards         |
| Reward Activation     | Points > 3,000, no redemption        | In-app + Email | "You are X points away" + limited reward |
| Loyalty Reinforcement | Top 20% engagement, low churn risk   | Email          | VIP missions / early access              |
| Lifecycle Transition  | Child age > 30 months                | Email + In-app | Cross-sell other Fater brands            |

## 7. Out-of-the-Box Differentiators

- **Lifecycle-aware churn**: distinguish preventable vs natural churn
- **Regional opportunity map**: median income x share of families with children (regional ISTAT)
- **Mission optimization**: identify missions that drive downstream scans
- **Diaper runway**: months until graduation as a churn risk feature

## 8. Presentation Outline

1. Context and objective
2. Data overview and key stats
3. Problem definitions (3 models)
4. Features and modeling approach
5. Results and interpretability
6. Campaigns and business actions
7. Conclusions and next steps

## 9. Improvements to the Proposed Approach

- **Clarify churn label**: pick a primary rule (e.g., 60-day inactivity) and run sensitivity checks (45/90 days).
- **Prevent leakage**: build all features using data strictly before the prediction window.
- **Time-based validation**: use temporal splits instead of random to mimic deployment.
- **Champion model + baseline**: prioritize LightGBM/XGBoost + logistic baseline; neural net only if it improves metrics.
- **Campaign KPIs**: define success metrics (reactivation rate, uplift, redemption rate).
- **External data scope**: start with regional income + family composition, then add national spending context only if it improves validation metrics.
- **Location matching quality**: standardize region names/codes, track match rate, and document fallbacks.

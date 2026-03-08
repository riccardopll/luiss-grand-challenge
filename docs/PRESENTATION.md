# Intermediate Presentation Structure

## Purpose

This deck is the intermediate business presentation aligned to `eda.ipynb`.
Its job is to show that the data supports a clear CRM decision logic and that the final deliverable should be a campaign portfolio powered by propensity scores.

What this deck covers:

- EDA evidence and data understanding
- Business problem framing
- Why the project should be presented as one CRM engine with three scores
- How those scores will translate into campaign families

What this deck does not cover yet:

- Final trained model performance
- Final calibrated thresholds
- Final KPI or ROI values
- Final campaign budget sizing

## Core storyline

The presentation should follow one business narrative from start to finish:

1. Fater needs a smarter CRM decision system, not generic batch campaigns.
2. The data is rich enough to describe engagement, value, lifecycle stage, and regional context.
3. The user base is heterogeneous, so one targeting rule would be too crude.
4. The three target behaviors in scope behave differently in size, timing, and base rate.
5. Therefore, the right solution is one CRM engine with three propensity scores.
6. The final business output will be one or more campaigns built from those scores, not the scores alone.

## Graph placement

All existing EDA graphs should be used in the main deck in this order:

1. Slide 3: `images/01-dataset-size-by-table.svg`
2. Slide 4: `images/02-monthly-active-users-trend.svg`
3. Slide 5: `images/03-monthly-event-intensity.svg`
4. Slide 6: `images/04-total-points-distribution.svg`
5. Slide 7: `images/05-lifecycle-stage-mix.svg`
6. Slide 8: `images/06-regional-user-footprint-1.svg`
7. Slide 8: `images/07-regional-user-footprint-2.svg`
8. Slide 9: `images/08-monthly-label-dynamics.svg`

Slides 10 and 11 should use custom presentation visuals rather than notebook export images:

- Slide 10: a simple three-card CRM engine schema
- Slide 11: an EDA-to-campaign operating map or Sankey

## Time allocation

- Title, scope, and objective: **1.5-2 min**
- EDA evidence and diagnostics: **8-8.5 min**
- Proposed CRM logic and campaign implications: **4-4.5 min**
- Next steps and close: **1-1.5 min**

Total: **~14.5-15 min**

## Slide-by-slide structure

### Slide 1 - Title and business objective (0:00-0:45)

- Visual: none or clean title slide
- Goal: state the business problem before showing any data

Discourse:
This project is about improving Fater CRM decision-making. The business need is not just to describe users, but to decide who should receive which campaign action and when. The final deliverable is therefore a campaign framework supported by a CRM engine, not a standalone analytics exercise.

### Slide 2 - Scope of this presentation (0:45-1:30)

- Visual: none or simple roadmap
- Goal: prevent scope confusion early

Discourse:
This is an intermediate presentation. We are presenting the evidence base, the behavioral framing, and the operating logic that will support campaign design. We are not yet presenting final model performance, final thresholds, or final ROI. Those belong to the final phase once training, calibration, and campaign simulation are complete.

### Slide 3 - Dataset size by table (1:30-2:20)

- Visual: `images/01-dataset-size-by-table.svg`
- Goal: establish data breadth and integration complexity

Discourse:
We start with the data foundation because the credibility of the later business logic depends on it. The project combines user profile data, app access, scans, missions, redemptions, and product metadata. This matters because the CRM engine will rely on multiple behavior families rather than a single engagement metric, and the table mix already suggests why leakage-safe aggregation and careful user-key mapping are central.

### Slide 4 - Monthly active users trend (2:20-3:10)

- Visual: `images/02-monthly-active-users-trend.svg`
- Goal: show the baseline level and stability of engagement

Discourse:
This slide shows the broad engagement baseline over time. The key business point is that user activity is not static, so campaign expectations and model validation cannot be based on one global average. Temporal variation means we need time-aware modeling and it also means that CRM planning must respect seasonality and shifts in the active base.

### Slide 5 - Monthly event intensity (3:10-4:00)

- Visual: `images/03-monthly-event-intensity.svg`
- Goal: show that users interact in different ways, not just at different volumes

Discourse:
Looking only at active-user counts would hide the richness of the behavioral signal. This heatmap shows that access, scans, missions, and redemptions have different intensities over time. That is important because the future campaigns will not be triggered by one generic notion of engagement; they will use specific behavior types to infer churn risk, reward activation potential, and lifecycle continuation.

### Slide 6 - Total points distribution (4:00-5:00)

- Visual: `images/04-total-points-distribution.svg`
- Goal: introduce value concentration and the importance of reward logic

Discourse:
Points are a direct business-value signal because they connect behavior to loyalty activation. This distribution shows concentration and heavy-tail behavior, which means the reward system cannot be treated as uniform across the base. It also reinforces an important data-quality rule: null `totalPoints` is not the same as zero, so any reward campaign logic must treat missingness carefully rather than collapsing users into one low-value bucket.

### Slide 7 - Lifecycle stage mix (5:00-5:50)

- Visual: `images/05-lifecycle-stage-mix.svg`
- Goal: show that lifecycle is a structural business dimension, not a minor feature

Discourse:
The Fater user base is not one homogeneous audience. Lifecycle stage strongly affects relevance, expected behavior, and likely response to campaigns. This slide makes the business case for keeping lifecycle explicit in the CRM engine because the same message will not make sense for pregnancy, early infancy, and late-stage families approaching product transition.

### Slide 8 - Regional footprint and socioeconomic context (5:50-7:00)

- Visuals:
  - `images/06-regional-user-footprint-1.svg`
  - `images/07-regional-user-footprint-2.svg`
- Goal: show that geography is not only descriptive but operational context

Discourse:
Region should not be treated as decorative segmentation. The user base is geographically concentrated, and the EDA also links that concentration to socioeconomic context from ISTAT enrichment. This matters because campaign response, value perception, and channel strategy can differ by context, so region belongs in the decision layer as a supporting signal rather than being ignored until execution time.

### Slide 9 - Monthly label dynamics (7:00-8:30)

- Visual: `images/08-monthly-label-dynamics.svg`
- Goal: provide the strongest quantitative reason for three scores instead of one

Discourse:
This is the most important bridge slide in the whole presentation. It shows that the three target behaviors do not operate in the same regime: their populations, base rates, and time patterns are materially different. That means one generic propensity score would be misleading. The project needs one CRM engine with multiple score outputs because churn prevention, redemption activation, and lifecycle continuation are different decision problems even if they share the same data backbone.

### Slide 10 - Proposed solution: one CRM engine, three propensity scores (8:30-10:15)

- Visual: custom three-card schema, no notebook image
- Goal: convert the EDA into a business operating model

Discourse:
At this point the story should become simple. We are not proposing three disconnected models. We are proposing one CRM decision engine that answers three business questions: who is at risk of persistent inactivity, who is most relevant for reward activation, and who should receive retention-versus-transition actions in late lifecycle. The common foundation is the same user snapshot, but each score exists because the decision, the eligible population, and the campaign lever are different.

Recommended visual structure:

- Card 1: `churn_30_to_60` -> prevent persistent inactivity
- Card 2: `will_redeem_30d` -> activate reward usage
- Card 3: `will_scan_60d` -> guide retain-vs-transition action

### Slide 11 - EDA-to-campaign operating map (10:15-12:30)

- Visual: custom Sankey or left-to-right operating map, no notebook image
- Goal: make the final deliverable explicit for the business audience

Discourse:
This slide should answer the practical question, “So what do we do with this?” The EDA feeds the propensity engine, the engine produces score-based audiences, and those audiences map to campaign families. For example, churn scores support save campaigns, redemption scores support reward activation campaigns, and lifecycle continuation scores support transition or retention journeys. This is also the right place to state that the final output is not only scoring, but campaign design with target audience, trigger logic, channel, action, and holdout measurement.

Recommended operating flow:

- EDA signals -> shared snapshot features
- Shared features -> three propensity scores
- Scores + consent/channel rules -> campaign audiences
- Campaign audiences -> message, incentive, and measurement plan

### Slide 12 - Next steps and close (12:30-15:00)

- Visual: none or simple phase timeline
- Goal: close the scope clearly and leave the audience with the right expectation

Discourse:
We have established that the data supports a campaign-oriented CRM engine and that three score outputs are justified by the behavior in the data. The next phase is to operationalize that logic: finalize feature and label contracts, train and calibrate the models, translate scores into campaign rules, and validate the business value with holdouts and KPI measurement. The final presentation will therefore move from evidence and framing to validated campaign recommendations.

## Presentation checks

Before building slides, keep these logic checks in mind:

- Start with business objective before technical detail.
- Explain the scope on Slide 2 so no one expects final ROI too early.
- Use Slides 3 to 9 only to build evidence; do not start talking about model architecture too early.
- Use Slide 9 as the main proof that one score would be wrong.
- Use Slide 10 to simplify the architecture into business language.
- Use Slide 11 to make the final deliverable explicit: campaigns, not just scores.
- Avoid claiming validated impact in this intermediate deck.

## Speaker notes

- Keep the core framing explicit throughout: one CRM engine, three scores, multiple campaign families.
- Do not describe the three scores as three unrelated workstreams.
- When discussing graphs, always link them back to a business decision, not only to a modeling idea.
- Treat Slide 9 as the turning point from EDA evidence to solution logic.
- Treat Slide 11 as the answer to the business audience's most likely question: what action will this enable?

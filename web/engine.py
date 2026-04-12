from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd


CHURN_THRESHOLD = 0.5
ENGAGEMENT_THRESHOLD = 0.5
POINTS_CLOSE_THRESHOLD = 600
NEW_USER_TENURE_DAYS = 90
ACTIVITY_GRAPH_WEEKS = 26
ACTIVITY_INTENSITY_STEPS = 4
EVENT_TYPE_LABELS = {
    "access": "Access",
    "scan": "Scan",
    "mission": "Mission",
    "redeem": "Redeem",
}
NOTIFICATION_TEMPLATES = {
    "S1": {
        "email": {
            "headline": "Share Pampers Club with a friend who needs it now",
            "body": (
                "You are in {lifecycle_stage}. {product_category_sentence}"
                " {points_snapshot_sentence} This is a good moment to share"
                " Pampers Club with a pregnant friend or a new parent and turn"
                " your experience into a referral reward. {tenure_sentence}"
            ),
        },
        "push": {
            "headline": "Know a new parent who could use this?",
            "body": (
                "{points_balance_push_snippet} Invite a pregnant friend or a new"
                " parent to Pampers Club and share what works for your family"
                " stage."
            ),
        },
    },
    "S2": {
        "email": {
            "headline": "Come back now and earn double points",
            "body": (
                "{recency_sentence} {points_snapshot_sentence} Restart with"
                " {product_action_reference} while double points are live."
                " {redemption_sentence}"
            ),
        },
        "push": {
            "headline": "Double points are live now",
            "body": (
                "{last_scan_reference} Restart with {product_action_reference}"
                " and earn faster while the rescue boost is still active."
            ),
        },
    },
    "S3": {
        "email": {
            "headline": "{reward_email_headline}",
            "body": "{points_snapshot_sentence} {reward_next_step_sentence} {redemption_sentence}",
        },
        "push": {
            "headline": "{reward_push_headline}",
            "body": "{reward_push_sentence} {points_balance_push_snippet} {redemption_push_sentence}",
        },
    },
    "S4": {
        "email": {
            "headline": "{preferred_incentive_title} to earn faster",
            "body": (
                "{mission_profile_sentence} {points_snapshot_sentence}"
                " {mission_sentence} {acceleration_sentence}"
            ),
        },
        "push": {
            "headline": "{preferred_incentive_title}",
            "body": "{mission_profile_sentence} {push_acceleration_sentence}",
        },
    },
    "S5": {
        "email": {
            "headline": "A simple way to get more value from the app",
            "body": (
                "{educational_intro} {tenure_reference}"
                " {product_category_sentence} {points_snapshot_sentence}"
                " {restart_sentence} {redemption_sentence}"
            ),
        },
        "push": {
            "headline": "Start again with one simple scan",
            "body": "{last_scan_reference} {push_restart_sentence} {points_balance_push_snippet}",
        },
    },
}


@dataclass
class CRMOutput:
    segment_id: str
    segment_name: str
    campaign: str
    action: str
    decision_flow: "DecisionFlow"
    marketing_brief: dict[str, dict[str, str]]
    notifications: "NotificationSet"


@dataclass
class NotificationProposal:
    headline: str
    body: str


@dataclass
class NotificationSet:
    email: NotificationProposal
    push: NotificationProposal


@dataclass
class DecisionStep:
    key: str
    label: str
    result_label: str
    evaluated: bool
    decision: str | None


@dataclass
class DecisionFlow:
    steps: list[DecisionStep]
    matched_rule_name: str
    matched_segment_id: str


@dataclass
class ActivityDay:
    date: str
    count: int
    level: int
    in_range: bool
    tooltip: str


@dataclass
class ActivityMonthLabel:
    label: str
    column: int


@dataclass
class ActivityEventTotal:
    key: str
    label: str
    count: int


@dataclass
class ActivityGraph:
    range_label: str
    total_events: int
    active_days: int
    longest_streak: int
    weeks: list[list[ActivityDay]]
    month_labels: list[ActivityMonthLabel]
    event_totals: list[ActivityEventTotal]


@dataclass
class UserPayload:
    user_id: str
    profile: dict[str, object]
    crm_output: CRMOutput
    activity: ActivityGraph


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, type(pd.NA))) else False


def _clean_text(value: object, fallback: str = "Unknown") -> str:
    if _is_missing(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _clean_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _clean_int(value: object) -> int | None:
    numeric = _clean_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _score_state(score: float | None, threshold: float) -> str:
    if score is None:
        return "Not scored"
    return "High" if score >= threshold else "Low"


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _format_days(value: int | None) -> str:
    if value is None:
        return "Unknown"
    day_label = "day" if value == 1 else "days"
    return f"{value} {day_label}"


def _format_points(value: float | None) -> str:
    if value is None:
        return "Balance unavailable"
    return f"{int(round(value)):,} points"


def _truthy_flag(value: object) -> bool:
    return _clean_int(value) == 1


class CRMDecisionEngine:
    def __init__(self, *, project_root: Path, artifacts_dir: Path, rules_path: Path) -> None:
        self.project_root = project_root
        self.artifacts_dir = artifacts_dir
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self._load_data()

    def _load_data(self) -> None:
        training_artifacts = cast(
            dict[str, Any],
            pd.read_pickle(self.artifacts_dir / "eda_training_artifacts.pkl"),
        )

        user_base = training_artifacts["user_base"].copy()
        user_base["idSSO"] = user_base["idSSO"].astype(str)
        profile_columns = ["idSSO", "Regione"]
        user_profile = user_base[profile_columns].drop_duplicates(
            subset="idSSO", keep="last")

        snapshots = training_artifacts["snapshot_features"].copy()
        snapshots["reference_date"] = pd.to_datetime(
            snapshots["reference_date"])
        latest_reference = snapshots["reference_date"].max()
        latest_snapshot = snapshots[snapshots["reference_date"]
                                    == latest_reference].copy()
        latest_snapshot["idSSO"] = latest_snapshot["idSSO"].astype(str)
        reward_levels = self._load_reward_levels(latest_reference)
        cheapest_available_threshold = (
            float(reward_levels[0])
            if reward_levels
            else _clean_float(latest_snapshot["reward_threshold_points"].min())
        )
        latest_snapshot["cheapest_available_threshold"] = cheapest_available_threshold
        if cheapest_available_threshold is None:
            latest_snapshot["can_redeem_now"] = 0
        else:
            latest_snapshot["can_redeem_now"] = (
                latest_snapshot["totalPoints"].notna()
                & (latest_snapshot["totalPoints"] >= cheapest_available_threshold)
            ).astype(int)
        latest_snapshot["next_reward_threshold"] = latest_snapshot[
            "totalPoints"
        ].apply(lambda value: self._next_reward_threshold(value, reward_levels))
        latest_snapshot["next_reward_gap"] = (
            latest_snapshot["next_reward_threshold"] -
            latest_snapshot["totalPoints"]
        )
        self.latest_reference_timestamp = latest_reference.normalize()

        churn_scores = pd.read_csv(
            self.artifacts_dir / "final" / "churn_scores_current.csv")
        churn_scores["idSSO"] = churn_scores["idSSO"].astype(str)

        engagement_scores = pd.read_csv(
            self.artifacts_dir / "final" / "engagement_scores_current.csv")
        engagement_scores["idSSO"] = engagement_scores["idSSO"].astype(str)

        product_categories = self._build_product_categories(training_artifacts)

        merged = latest_snapshot.merge(
            user_profile, on="idSSO", how="left"
        ).merge(
            churn_scores[["idSSO", "churn_30_to_60_prob"]],
            on="idSSO",
            how="left",
        ).merge(
            engagement_scores[["idSSO", "re_engage_30d_prob"]],
            on="idSSO",
            how="left",
        ).merge(
            product_categories,
            on="idSSO",
            how="left",
        )

        merged["physiological_churn"] = merged["is_near_graduation"].fillna(
            0).astype(int)
        merged["high_churn"] = (
            merged["churn_30_to_60_prob"].fillna(-1) >= CHURN_THRESHOLD
        ).astype(int)
        merged["high_engagement"] = (
            merged["re_engage_30d_prob"].fillna(-1) >= ENGAGEMENT_THRESHOLD
        ).astype(int)
        merged["near_next_reward"] = (
            merged["next_reward_gap"].fillna(POINTS_CLOSE_THRESHOLD + 1) <=
            POINTS_CLOSE_THRESHOLD
        ).astype(int)
        merged["points_close"] = (
            (merged["can_redeem_now"] == 1) | (merged["near_next_reward"] == 1)
        ).astype(int)
        merged["mission_engagement"] = merged.apply(
            self._mission_engagement_value, axis=1)
        merged["preferred_incentive_title"] = merged["mission_engagement"].map(
            {
                "Mission-active": "Bonus mission",
                "Mission-curious": "Bonus mission",
                "Scan-led": "Double points scan",
            }
        ).fillna("Double points scan")
        merged["preferred_incentive_lower"] = merged[
            "preferred_incentive_title"
        ].str.lower()
        merged["education_clause"] = merged["tenure_days"].apply(
            self._education_clause)

        activity_events = training_artifacts["events_long"].copy()
        activity_events["idSSO"] = activity_events["idSSO"].astype(str)
        activity_events["event_day"] = pd.to_datetime(
            activity_events["event_date"], errors="coerce"
        ).dt.normalize()
        activity_events = activity_events.dropna(
            subset=["idSSO", "event_day"]).copy()
        activity_events["event_type"] = activity_events["event_type"].astype(
            str)
        activity_events = activity_events[
            activity_events["event_day"] <= self.latest_reference_timestamp
        ]
        self.activity_daily = (
            activity_events.groupby(
                ["idSSO", "event_day", "event_type"], as_index=False)
            .size()
            .rename(columns={"size": "event_count"})
        )

        self.latest_reference_date = latest_reference.strftime("%Y-%m-%d")
        self.user_frame = merged
        self.segment_lookup = self._build_segment_lookup(merged)
        self.sample_user_ids = self._build_sample_user_ids(merged)

    def _load_rules(self) -> list[dict[str, str]]:
        rules: list[dict[str, str]] = []
        for raw_line in self.rules_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts: dict[str, str] = {}
            for part in line.split("|"):
                key, value = part.split("=", 1)
                parts[key] = value
            rules.append(parts)
        return rules

    def lookup(self, user_id: str) -> UserPayload | None:
        match = self.user_frame[self.user_frame["idSSO"] == user_id]
        if match.empty:
            return None

        row = match.iloc[0]
        return UserPayload(
            user_id=user_id,
            profile=self._build_profile(row),
            crm_output=self._build_crm_output(row),
            activity=self._build_activity_graph(user_id),
        )

    def _build_profile(self, row: pd.Series) -> dict[str, object]:
        churn_score = _clean_float(row["churn_30_to_60_prob"])
        engagement_score = _clean_float(row["re_engage_30d_prob"])
        return {
            "region": _clean_text(row["Regione"]),
            "days_since_last_activity": _clean_int(row["days_since_last_activity"]),
            "days_since_last_scan": _clean_int(row["days_since_last_scan"]),
            "churn_score": churn_score,
            "churn_score_label": _score_state(churn_score, CHURN_THRESHOLD),
            "re_engage30d": engagement_score,
            "re_engage_label": _score_state(
                engagement_score, ENGAGEMENT_THRESHOLD),
            "push_consent": _truthy_flag(row["channel_eligible_phone_push"]),
            "email_consent": _truthy_flag(row["channel_eligible_email"]),
        }

    def _build_crm_output(self, row: pd.Series) -> CRMOutput:
        rule = self._match_rule(row)
        template_context = self._template_context(row)
        marketing_brief = self._build_marketing_brief(row)
        return CRMOutput(
            segment_id=rule["segment_id"],
            segment_name=rule["segment_name"],
            campaign=self._render_template(rule["campaign"], template_context),
            action=self._render_template(rule["action"], template_context),
            decision_flow=self._build_decision_flow(row, rule),
            marketing_brief=marketing_brief,
            notifications=self._build_notifications(
                row, rule, template_context, marketing_brief
            ),
        )

    def _match_rule(self, row: pd.Series) -> dict[str, str]:
        variables = {
            "physiological_churn": _clean_int(row["physiological_churn"]) or 0,
            "high_churn": _clean_int(row["high_churn"]) or 0,
            "high_engagement": _clean_int(row["high_engagement"]) or 0,
            "points_close": _clean_int(row["points_close"]) or 0,
        }
        for rule in self.rules:
            if bool(pd.eval(rule["condition"], local_dict=variables, engine="python")):
                return rule
        raise ValueError("No CRM rule matched the current user.")

    def _template_context(self, row: pd.Series) -> dict[str, str]:
        return _SafeFormatDict(
            {
                "preferred_incentive_title": _clean_text(
                    row["preferred_incentive_title"]),
                "preferred_incentive_lower": _clean_text(
                    row["preferred_incentive_lower"]),
                "education_clause": _clean_text(
                    row["education_clause"], fallback=""),
            }
        )

    def _build_decision_flow(
        self,
        row: pd.Series,
        rule: dict[str, str],
    ) -> DecisionFlow:
        physiological = _truthy_flag(row["physiological_churn"])
        high_churn = _truthy_flag(row["high_churn"])
        points_close = _truthy_flag(row["points_close"])
        high_engagement = _truthy_flag(row["high_engagement"])

        steps = [
            DecisionStep(
                key="physiological_churn",
                label="Lifecycle transition?",
                result_label="Yes" if physiological else "No",
                evaluated=True,
                decision="yes" if physiological else "no",
            ),
            DecisionStep(
                key="high_churn",
                label="High churn score?",
                result_label="High" if high_churn else "Low",
                evaluated=not physiological,
                decision=(
                    "yes" if high_churn else "no") if not physiological else None,
            ),
            DecisionStep(
                key="points_close",
                label="Near reward threshold?",
                result_label="Close" if points_close else "Far",
                evaluated=not physiological and not high_churn,
                decision=(
                    "yes" if points_close else "no"
                ) if not physiological and not high_churn else None,
            ),
            DecisionStep(
                key="high_engagement",
                label="High re-engage propensity?",
                result_label="High" if high_engagement else "Low",
                evaluated=not physiological and not high_churn and not points_close,
                decision=(
                    "yes" if high_engagement else "no"
                ) if not physiological and not high_churn and not points_close else None,
            ),
        ]

        return DecisionFlow(
            steps=steps,
            matched_rule_name=rule["rule_name"],
            matched_segment_id=rule["segment_id"],
        )

    def _build_marketing_brief(self, row: pd.Series) -> dict[str, dict[str, str]]:
        can_redeem_now = _truthy_flag(row["can_redeem_now"])
        current_points = _clean_float(row["totalPoints"])
        next_reward_gap = _clean_float(row["next_reward_gap"])
        next_reward_threshold = _clean_float(row["next_reward_threshold"])
        last_scan_days = _clean_int(row["days_since_last_scan"])
        has_redeemed = _truthy_flag(row["has_ever_redeemed"])
        tenure_days = _clean_int(row["tenure_days"])
        return {
            "physiological_churn": {
                "label": "Physiological churn",
                "value": "Yes" if _truthy_flag(row["physiological_churn"]) else "No",
                "guidance": "When this is yes, lifecycle transition overrides the rest of the CRM decision tree.",
            },
            "product_category": {
                "label": "Top product categories",
                "value": _clean_text(
                    row["product_category"], fallback="No scan history yet"),
                "guidance": "Use these categories to anchor the reward framing or examples in the product groups the user scans most often.",
            },
            "can_redeem_now": {
                "label": "Can redeem now",
                "value": _yes_no(can_redeem_now),
                "guidance": (
                    "Treat this as optionality, not intent. When it is yes,"
                    " avoid assuming the user wants the cheapest reward now."
                ),
            },
            "current_point_balance": {
                "label": "Current point balance",
                "value": _format_points(current_points),
                "guidance": (
                    "Use the live balance as the baseline for any reward or"
                    " acceleration message."
                ),
            },
            "next_reward": {
                "label": "Next reward threashold",
                "value": self._format_next_reward_pill(
                    next_reward_threshold, next_reward_gap
                ),
                "guidance": (
                    "Use this combined view to anchor both the remaining"
                    " distance and the next reward tier in one message."
                ),
            },
            "mission_engagement": {
                "label": "Mission engagement",
                "value": _clean_text(row["mission_engagement"]),
                "guidance": "Use this to choose between a mission-led push and a scan-led incentive.",
            },
            "last_scan_days_ago": {
                "label": "Last scan recency",
                "value": _format_days(last_scan_days),
                "guidance": self._last_scan_guidance(last_scan_days),
            },
            "has_ever_redeemed": {
                "label": "Has ever redeemed",
                "value": _yes_no(has_redeemed),
                "guidance": "First-time redeemers need a short explanation of what happens after they request a prize.",
            },
            "account_tenure_days": {
                "label": "Account tenure",
                "value": _format_days(tenure_days),
                "guidance": self._tenure_guidance(tenure_days),
            },
            "lifecycle": {
                "label": "Lifecycle",
                "value": f"{_clean_text(row['child_age_bucket'])}{f' ({_clean_int(row['ETA_MM_BambinoTODAY'])} months)' if _clean_int(row['ETA_MM_BambinoTODAY']) is not None else ''}",
                "guidance": "Use lifecycle context to keep the message relevant to the customer's current family stage.",
            },
        }

    def _build_notifications(
        self,
        row: pd.Series,
        rule: dict[str, str],
        template_context: dict[str, str],
        marketing_brief: dict[str, dict[str, str]],
    ) -> NotificationSet:
        templates = NOTIFICATION_TEMPLATES.get(
            rule["segment_id"], NOTIFICATION_TEMPLATES["S5"]
        )
        context = self._notification_context(
            row, rule, template_context, marketing_brief
        )
        return NotificationSet(
            email=NotificationProposal(
                headline=self._render_copy_template(
                    templates["email"]["headline"], context
                ),
                body=self._render_copy_template(
                    templates["email"]["body"], context),
            ),
            push=NotificationProposal(
                headline=self._render_copy_template(
                    templates["push"]["headline"], context
                ),
                body=self._render_copy_template(
                    templates["push"]["body"], context),
            ),
        )

    def _notification_context(
        self,
        row: pd.Series,
        rule: dict[str, str],
        template_context: dict[str, str],
        marketing_brief: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        can_redeem_now = _truthy_flag(row["can_redeem_now"])
        next_reward_gap = _clean_float(row["next_reward_gap"])
        last_scan_days = _clean_int(row["days_since_last_scan"])
        has_redeemed = _truthy_flag(row["has_ever_redeemed"])
        tenure_days = _clean_int(row["tenure_days"])
        lifecycle_stage = self._brief_value(marketing_brief, "lifecycle")
        product_category_list = self._brief_value(
            marketing_brief, "product_category")
        current_point_balance = self._brief_value(
            marketing_brief, "current_point_balance"
        )
        next_reward_summary = self._brief_value(marketing_brief, "next_reward")
        mission_engagement_label = self._brief_value(
            marketing_brief, "mission_engagement"
        )
        lifecycle_reference = self._lifecycle_reference(lifecycle_stage)

        return _SafeFormatDict(
            {
                **template_context,
                "segment_name": rule["segment_name"],
                "lifecycle_stage": lifecycle_stage,
                "lifecycle_reference": lifecycle_reference,
                "product_category_list": product_category_list,
                "primary_product_category": self._primary_product_category(
                    product_category_list
                ),
                "current_point_balance": current_point_balance,
                "next_reward_summary": next_reward_summary,
                "mission_engagement_label": mission_engagement_label,
                "recency_sentence": self._notification_recency_sentence(
                    last_scan_days
                ),
                "redemption_sentence": self._notification_redemption_sentence(
                    has_redeemed
                ),
                "redemption_push_sentence": self._notification_redemption_push_sentence(
                    has_redeemed
                ),
                "tenure_sentence": self._notification_tenure_sentence(tenure_days),
                "reward_email_headline": self._reward_email_headline(
                    can_redeem_now, next_reward_gap
                ),
                "reward_push_headline": self._reward_push_headline(
                    can_redeem_now, next_reward_gap
                ),
                "reward_status_sentence": self._reward_status_sentence(
                    can_redeem_now, next_reward_gap
                ),
                "reward_next_step_sentence": self._reward_next_step_sentence(
                    can_redeem_now, next_reward_gap
                ),
                "reward_push_sentence": self._reward_push_sentence(
                    can_redeem_now, next_reward_gap
                ),
                "reward_distance_sentence": self._reward_distance_sentence(
                    can_redeem_now, next_reward_gap
                ),
                "mission_sentence": self._mission_sentence(row),
                "educational_intro": self._educational_intro(
                    lifecycle_reference, last_scan_days
                ),
                "product_category_sentence": self._product_category_sentence(
                    product_category_list
                ),
                "product_action_reference": self._product_action_reference(
                    product_category_list
                ),
                "points_snapshot_sentence": self._points_snapshot_sentence(
                    current_point_balance, next_reward_summary
                ),
                "points_balance_push_snippet": self._points_balance_push_snippet(
                    current_point_balance
                ),
                "last_scan_reference": self._last_scan_reference(last_scan_days),
                "tenure_reference": self._tenure_reference(tenure_days),
                "mission_profile_sentence": self._mission_profile_sentence(
                    mission_engagement_label, product_category_list
                ),
                "push_acceleration_sentence": self._push_acceleration_sentence(
                    row, can_redeem_now, next_reward_gap
                ),
                "push_restart_sentence": self._push_restart_sentence(
                    tenure_days, has_redeemed
                ),
                "acceleration_sentence": self._acceleration_sentence(
                    can_redeem_now, next_reward_gap
                ),
                "restart_sentence": self._restart_sentence(last_scan_days),
            }
        )

    def _build_segment_lookup(self, frame: pd.DataFrame) -> dict[str, str]:
        segment_map: dict[str, str] = {}
        for _, row in frame.iterrows():
            rule = self._match_rule(row)
            segment_map[str(row["idSSO"])] = rule["segment_id"]
        return segment_map

    def _build_activity_graph(self, user_id: str) -> ActivityGraph:
        end_date = self.latest_reference_timestamp
        latest_week_start = end_date - pd.Timedelta(days=end_date.weekday())
        start_date = latest_week_start - \
            pd.Timedelta(weeks=ACTIVITY_GRAPH_WEEKS - 1)
        grid_end = latest_week_start + pd.Timedelta(days=6)

        user_events = self.activity_daily[self.activity_daily["idSSO"] == user_id]
        window_events = user_events[
            (user_events["event_day"] >= start_date)
            & (user_events["event_day"] <= end_date)
        ].copy()

        daily_totals = (
            window_events.groupby("event_day")[
                "event_count"].sum().astype(int).to_dict()
        )
        event_totals = (
            window_events.groupby("event_type")[
                "event_count"].sum().astype(int).to_dict()
        )

        breakdown_lookup: dict[pd.Timestamp, dict[str, int]] = {}
        if not window_events.empty:
            for day, day_frame in window_events.groupby("event_day"):
                breakdown_lookup[day] = {
                    event_type: int(count)
                    for event_type, count in day_frame.groupby("event_type")[
                        "event_count"
                    ].sum().items()
                }

        active_counts = [count for count in daily_totals.values() if count > 0]
        max_count = max(active_counts, default=0)
        total_events = sum(active_counts)
        active_days = len(active_counts)

        all_days = pd.date_range(start=start_date, end=grid_end, freq="D")
        weeks: list[list[ActivityDay]] = []
        current_streak = 0
        longest_streak = 0

        for week_start_idx in range(0, len(all_days), 7):
            week: list[ActivityDay] = []
            for day in all_days[week_start_idx:week_start_idx + 7]:
                in_range = day <= end_date
                count = int(daily_totals.get(day, 0)) if in_range else 0
                if in_range:
                    if count > 0:
                        current_streak += 1
                        longest_streak = max(longest_streak, current_streak)
                    else:
                        current_streak = 0
                week.append(
                    ActivityDay(
                        date=day.strftime("%Y-%m-%d"),
                        count=count,
                        level=self._activity_level(
                            count, max_count) if in_range else 0,
                        in_range=bool(in_range),
                        tooltip=self._activity_tooltip(
                            day=day,
                            count=count,
                            breakdown=breakdown_lookup.get(day, {}),
                            in_range=bool(in_range),
                        ),
                    )
                )
            weeks.append(week)

        month_labels = self._build_activity_month_labels(weeks)
        event_breakdown = [
            ActivityEventTotal(key=key, label=label,
                               count=int(event_totals.get(key, 0)))
            for key, label in EVENT_TYPE_LABELS.items()
            if int(event_totals.get(key, 0)) > 0
        ]

        return ActivityGraph(
            range_label=(
                f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
            ),
            total_events=total_events,
            active_days=active_days,
            longest_streak=longest_streak,
            weeks=weeks,
            month_labels=month_labels,
            event_totals=event_breakdown,
        )

    def _build_sample_user_ids(self, frame: pd.DataFrame) -> list[str]:
        sample_ids: list[str] = []
        sortable = frame.copy()
        sortable["segment_id"] = sortable["idSSO"].map(self.segment_lookup)
        sortable["churn_sort"] = sortable["churn_30_to_60_prob"].fillna(0.0)
        sortable["engagement_sort"] = sortable["re_engage_30d_prob"].fillna(
            0.0)
        sortable["points_sort"] = sortable["next_reward_gap"].fillna(999999.0)

        for rule in self.rules:
            segment_id = rule["segment_id"]
            subset = sortable[sortable["segment_id"] == segment_id]
            if subset.empty:
                continue
            candidate = subset.sort_values(
                ["churn_sort", "engagement_sort", "points_sort"],
                ascending=[False, False, True],
            ).iloc[0]
            sample_ids.append(str(candidate["idSSO"]))

        if len(sample_ids) >= 8:
            return sample_ids[:8]

        filler = sortable.sort_values(
            ["churn_sort", "engagement_sort", "points_sort"],
            ascending=[False, False, True],
        )["idSSO"].astype(str).tolist()
        for user_id in filler:
            if user_id not in sample_ids:
                sample_ids.append(user_id)
            if len(sample_ids) >= 8:
                break
        return sample_ids

    def _build_product_categories(self, training_artifacts: dict[str, Any]) -> pd.DataFrame:
        codici = training_artifacts["tables_clean"]["codici"].copy()
        prodotti = pd.read_csv(
            self.project_root / "data" / "Anagrafica_prodotti_digital.csv"
        )
        codici["idSSO"] = codici["userId"].map(
            training_artifacts["bridges"]["user_to_idsso"])
        codici["EAN_str"] = codici["EAN"].astype(str)
        prodotti["EANPROD_str"] = prodotti["EANPROD"].astype(
            "Int64").astype(str)
        codici = codici.merge(
            prodotti[["EANPROD_str", "SEGMENTO_DES",
                      "OCCUSO_DES", "TARGET_DES"]],
            left_on="EAN_str",
            right_on="EANPROD_str",
            how="left",
        )
        codici["scan_item"] = codici.apply(self._extract_scan_category, axis=1)
        valid_scans = codici.dropna(
            subset=["idSSO", "scan_item"]).copy()
        valid_scans = valid_scans[valid_scans["status"].eq("valid")]
        item_counts = (
            valid_scans.groupby(["idSSO", "scan_item"], as_index=False)
            .size()
            .sort_values(["idSSO", "size", "scan_item"], ascending=[True, False, True])
        )
        top_items = (
            item_counts.groupby("idSSO")["scan_item"]
            .apply(lambda values: "; ".join(values.head(3)))
            .reset_index()
            .rename(columns={"scan_item": "product_category"})
        )
        return top_items

    @classmethod
    def _extract_scan_category(cls, row: pd.Series) -> str | None:
        for column in ("SEGMENTO_DES", "OCCUSO_DES", "TARGET_DES"):
            value = _clean_text(row.get(column), fallback="")
            if value:
                return value

        if _is_missing(row.get("infoCode")):
            return None

        try:
            parsed = json.loads(str(row["infoCode"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

        for key in ("linea", "categoria"):
            value = _clean_text(parsed.get(key), fallback="")
            if value:
                return value

        return None

    @staticmethod
    def _mission_engagement_value(row: pd.Series) -> str:
        mission_count = _clean_float(row["mission_count_90d"]) or 0.0
        completed = _clean_float(row.get("missions_completed_60d")) or 0.0
        if mission_count >= 2 or completed >= 1:
            return "Mission-active"
        if mission_count > 0:
            return "Mission-curious"
        return "Scan-led"

    @staticmethod
    def _education_clause(value: object) -> str:
        tenure_days = _clean_int(value)
        if tenure_days is not None and tenure_days < NEW_USER_TENURE_DAYS:
            return (
                " Add a short educational block that explains scanning and missions,"
                " because the account is still new."
            )
        return ""

    @staticmethod
    def _render_template(template: str, context: dict[str, str]) -> str:
        return template.format_map(context)

    @staticmethod
    def _render_copy_template(template: str, context: dict[str, str]) -> str:
        return " ".join(template.format_map(context).split())

    @staticmethod
    def _lifecycle_reference(value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            return "your current family stage"
        if clean_value.startswith("pregnancy"):
            return "pregnancy"
        if "(" in clean_value:
            return clean_value.split("(", 1)[0].strip()
        return clean_value

    @staticmethod
    def _brief_value(
        marketing_brief: dict[str, dict[str, str]],
        key: str,
        fallback: str = "",
    ) -> str:
        item = marketing_brief.get(key)
        if not item:
            return fallback
        return _clean_text(item.get("value"), fallback=fallback)

    @staticmethod
    def _primary_product_category(product_category_list: str) -> str:
        if not product_category_list or product_category_list == "No scan history yet":
            return "your next scan"
        primary_category = product_category_list.split(";", 1)[0].strip()
        return primary_category or "your next scan"

    @classmethod
    def _product_category_sentence(cls, product_category_list: str) -> str:
        if not product_category_list or product_category_list == "No scan history yet":
            return (
                "Once you log one valid product, the app can tailor rewards more"
                " closely to your routine."
            )
        primary_category = cls._primary_product_category(product_category_list)
        return f"Your strongest scan habit is around {primary_category}."

    @classmethod
    def _product_action_reference(cls, product_category_list: str) -> str:
        if not product_category_list or product_category_list == "No scan history yet":
            return "one quick scan"
        primary_category = cls._primary_product_category(product_category_list)
        return f"one quick scan in {primary_category}"

    @staticmethod
    def _points_snapshot_sentence(
        current_point_balance: str, next_reward_summary: str
    ) -> str:
        current_known = bool(
            current_point_balance and current_point_balance != "Balance unavailable"
        )
        next_known = bool(
            next_reward_summary and next_reward_summary != "Balance unavailable"
        )
        if current_known and next_known:
            return (
                f"You currently have {current_point_balance}, and your next"
                f" milestone is {next_reward_summary}."
            )
        if current_known:
            return f"You currently have {current_point_balance}."
        if next_known:
            return f"Your next reward milestone is {next_reward_summary}."
        return "Your reward balance is updating right now."

    @staticmethod
    def _points_balance_push_snippet(current_point_balance: str) -> str:
        if not current_point_balance or current_point_balance == "Balance unavailable":
            return ""
        return f"You have {current_point_balance} right now."

    @staticmethod
    def _last_scan_reference(last_scan_days: int | None) -> str:
        if last_scan_days is None:
            return "A quick restart is enough to rebuild momentum."
        if last_scan_days <= 14:
            return "Your last scan was recent."
        if last_scan_days <= 45:
            return f"Your last scan was {_format_days(last_scan_days)} ago."
        return f"It has been {_format_days(last_scan_days)} since your last scan."

    @staticmethod
    def _tenure_reference(tenure_days: int | None) -> str:
        if tenure_days is None:
            return ""
        if tenure_days < NEW_USER_TENURE_DAYS:
            return f"Your account is still early at {_format_days(tenure_days)} of tenure."
        return f"You already know the program after {_format_days(tenure_days)} in the app."

    @classmethod
    def _mission_profile_sentence(
        cls, mission_engagement_label: str, product_category_list: str
    ) -> str:
        if mission_engagement_label == "Mission-active":
            return "You already respond well to mission-based engagement."
        if mission_engagement_label == "Mission-curious":
            return "You have already shown some mission interest."
        if not product_category_list or product_category_list == "No scan history yet":
            return "Your engagement is still scan-led, so a simple scan is the best re-entry point."
        primary_category = cls._primary_product_category(product_category_list)
        return f"Your engagement is still scan-led, especially around {primary_category}."

    @staticmethod
    def _notification_recency_sentence(last_scan_days: int | None) -> str:
        if last_scan_days is None:
            return "A fresh scan is enough to restart momentum."
        if last_scan_days <= 14:
            return "You were active recently, so a small nudge can turn into quick action."
        if last_scan_days <= 45:
            return "You are not far from your last scan, so this is a good moment to reconnect."
        return "It has been a while, so the easiest restart is one quick action."

    @staticmethod
    def _notification_redemption_sentence(has_redeemed: bool) -> str:
        if has_redeemed:
            return "Keep building toward your next prize."
        return "Your first reward request is simple, and the app guides each step."

    @staticmethod
    def _notification_redemption_push_sentence(has_redeemed: bool) -> str:
        if has_redeemed:
            return "Open the app and keep going."
        return "Open the app and we will guide the next step."

    @staticmethod
    def _notification_tenure_sentence(tenure_days: int | None) -> str:
        if tenure_days is None:
            return ""
        if tenure_days < NEW_USER_TENURE_DAYS:
            return (
                "A quick reminder: scans and missions are the fastest way to start"
                " collecting points."
            )
        return ""

    @staticmethod
    def _reward_email_headline(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            if next_reward_gap is None:
                return "You already have reward value available"
            return "You can redeem now or keep going for the next reward"
        if next_reward_gap is None:
            return "Your next reward is within reach"
        rounded = int(round(next_reward_gap))
        if rounded == 1:
            return "You are only 1 point from your next reward"
        return f"You are only {rounded:,} points from your next reward"

    @staticmethod
    def _reward_push_headline(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            return "You can redeem now"
        if next_reward_gap is None:
            return "Your next reward is close"
        return "Your next reward is close"

    @staticmethod
    def _reward_status_sentence(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            if next_reward_gap is None:
                return "You already have enough points for a reward."
            rounded = int(round(next_reward_gap))
            return (
                "You already have enough points for a reward, and only"
                f" {rounded:,} more points would move you to the next tier."
            )
        if next_reward_gap is None:
            return "You are already within reach of your next reward."
        rounded = int(round(next_reward_gap))
        return f"You are only {rounded:,} points away from your next reward."

    @staticmethod
    def _reward_next_step_sentence(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            return (
                "Redeem now if you want, or keep saving if you are aiming for a"
                " bigger prize."
            )
        if next_reward_gap is None:
            return "One quick action can bring the reward into view."
        return "One quick action could get you over the line."

    @staticmethod
    def _reward_push_sentence(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            return "You already have enough points for a reward."
        if next_reward_gap is None:
            return "Your next reward is within reach."
        rounded = int(round(next_reward_gap))
        return f"Only {rounded:,} points left for your next prize."

    @staticmethod
    def _reward_distance_sentence(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            return "It can also help you keep building toward a bigger reward tier."
        if next_reward_gap is None:
            return "It will also move you closer to your next reward."
        rounded = int(round(next_reward_gap))
        return f"It can also help close the remaining {rounded:,} points faster."

    @staticmethod
    def _mission_sentence(row: pd.Series) -> str:
        mission_engagement = _clean_text(row["mission_engagement"])
        preferred_incentive = _clean_text(row["preferred_incentive_title"])
        if mission_engagement == "Mission-active":
            return "Pick up your next mission and keep the streak going."
        if mission_engagement == "Mission-curious":
            return f"Use {preferred_incentive.lower()} to turn that early interest into more points."
        return "A quick scan is the simplest way to earn again."

    @staticmethod
    def _acceleration_sentence(
        can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        if can_redeem_now:
            return "Use it to keep momentum high while you build toward a bigger reward."
        if next_reward_gap is None:
            return "Use it on your next app action to turn current momentum into faster points."
        return "Use it on your next app action to move faster toward the next reward."

    @staticmethod
    def _educational_intro(
        lifecycle_reference: str, last_scan_days: int | None
    ) -> str:
        if last_scan_days is None:
            return f"It is easy to bring the app back into your routine during {lifecycle_reference}."
        if last_scan_days <= 45:
            return (
                "You are still close enough to your last activity that the app can"
                f" fit back into your routine for {lifecycle_reference}."
            )
        return (
            "If the app has been quiet for a while, this is a good moment to"
            f" rebuild value around {lifecycle_reference}."
        )

    @staticmethod
    def _push_acceleration_sentence(
        row: pd.Series, can_redeem_now: bool, next_reward_gap: float | None
    ) -> str:
        preferred_incentive = _clean_text(row["preferred_incentive_lower"])
        if can_redeem_now:
            return (
                f"Use {preferred_incentive} and keep building"
                " toward a bigger reward tier."
            )
        return (
            f"Use {preferred_incentive} on your next app action to move faster"
            " toward your next reward."
        )

    def _load_reward_levels(self, latest_reference: pd.Timestamp) -> list[float]:
        reward_catalog = pd.read_csv(
            self.project_root / "data" / "premi_reduced.csv",
            usecols=["datarichiestapremio", "puntipremio"],
            low_memory=False,
        )
        reward_catalog["datarichiestapremio"] = pd.to_datetime(
            reward_catalog["datarichiestapremio"], errors="coerce"
        )
        reward_catalog["puntipremio"] = pd.to_numeric(
            reward_catalog["puntipremio"], errors="coerce"
        )
        reward_catalog = reward_catalog.dropna(
            subset=["datarichiestapremio", "puntipremio"]
        )
        reward_catalog = reward_catalog[reward_catalog["puntipremio"] > 0]
        reward_catalog = reward_catalog[
            reward_catalog["datarichiestapremio"] <= latest_reference
        ]
        return sorted(reward_catalog["puntipremio"].astype(float).unique().tolist())

    @staticmethod
    def _next_reward_threshold(
        total_points: object, reward_levels: list[float]
    ) -> float | None:
        points_value = _clean_float(total_points)
        if points_value is None:
            return None
        for reward_level in reward_levels:
            if reward_level > points_value:
                return float(reward_level)
        return None

    @staticmethod
    def _restart_sentence(last_scan_days: int | None) -> str:
        if last_scan_days is None:
            return "Start with one simple scan and rebuild the habit without pressure."
        if last_scan_days <= 45:
            return "A small next step is enough to bring the routine back."
        return "Start with one simple scan and rebuild the habit without pressure."

    @staticmethod
    def _push_restart_sentence(tenure_days: int | None, has_redeemed: bool) -> str:
        if tenure_days is not None and tenure_days < NEW_USER_TENURE_DAYS:
            return (
                "Learn the basics fast with one scan: scans and missions turn"
                " into points right away."
            )
        if has_redeemed:
            return "Come back with one quick action and keep your points moving."
        return "Come back with one scan and we will guide you to your first reward."

    @staticmethod
    def _activity_level(count: int, max_count: int) -> int:
        if count <= 0 or max_count <= 0:
            return 0
        scaled = math.ceil((count / max_count) * ACTIVITY_INTENSITY_STEPS)
        return max(1, min(ACTIVITY_INTENSITY_STEPS, scaled))

    @staticmethod
    def _activity_tooltip(
        *,
        day: pd.Timestamp,
        count: int,
        breakdown: dict[str, int],
        in_range: bool,
    ) -> str:
        if not in_range:
            return f"{day.strftime('%b %d, %Y')}: outside the current snapshot window."
        if count == 0:
            return f"{day.strftime('%b %d, %Y')}: no recorded activity."

        parts = []
        for event_type, label in EVENT_TYPE_LABELS.items():
            event_count = breakdown.get(event_type, 0)
            if event_count > 0:
                suffix = "" if event_count == 1 else "es" if label == "Access" else "s"
                parts.append(f"{event_count} {label.lower()}{suffix}")

        details = ", ".join(parts)
        event_label = "event" if count == 1 else "events"
        return f"{day.strftime('%b %d, %Y')}: {count} {event_label} ({details})."

    @staticmethod
    def _build_activity_month_labels(
        weeks: list[list[ActivityDay]],
    ) -> list[ActivityMonthLabel]:
        month_labels: list[ActivityMonthLabel] = []
        last_label = ""
        for index, week in enumerate(weeks):
            visible_days = [day for day in week if day.in_range]
            if not visible_days:
                continue
            month_label = pd.Timestamp(visible_days[0].date).strftime("%b")
            if month_label != last_label:
                month_labels.append(ActivityMonthLabel(
                    label=month_label, column=index))
                last_label = month_label
        return month_labels

    @staticmethod
    def _last_scan_guidance(last_scan_days: int | None) -> str:
        if last_scan_days is None:
            return "Recency is missing, so keep the copy broadly welcoming."
        if last_scan_days <= 14:
            return "Recent scanners respond well to positive reinforcement and engagement language."
        if last_scan_days <= 45:
            return "Use a warm reminder that reconnects recent habit to the next reward."
        return "Re-introduce the mechanic gently because the user has been away from scanning for a while."

    @staticmethod
    def _tenure_guidance(tenure_days: int | None) -> str:
        if tenure_days is None:
            return "Tenure is unknown, so avoid assuming deep product familiarity."
        if tenure_days < NEW_USER_TENURE_DAYS:
            return "Add onboarding context because the user may still be learning how scans and missions work."
        return "The user is established enough that the message can focus on value rather than basics."

    @staticmethod
    def _format_next_reward_pill(
        next_reward_threshold: float | None,
        next_reward_gap: float | None,
    ) -> str:
        threshold_text = _format_points(next_reward_threshold)
        gap_value = _clean_float(next_reward_gap)
        if threshold_text == "Balance unavailable" or gap_value is None:
            return threshold_text
        missing_points = max(0, int(math.ceil(gap_value)))
        return f"{threshold_text} ({missing_points:,} missing)"

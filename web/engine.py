from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd


CHURN_THRESHOLD = 0.5
ENGAGEMENT_THRESHOLD = 0.5
POINTS_CLOSE_THRESHOLD = 50
NEW_USER_TENURE_DAYS = 90
ACTIVITY_GRAPH_WEEKS = 26
ACTIVITY_INTENSITY_STEPS = 4
EVENT_TYPE_LABELS = {
    "access": "Access",
    "scan": "Scan",
    "mission": "Mission",
    "redeem": "Redeem",
}


@dataclass
class CRMOutput:
    segment_id: str
    segment_name: str
    campaign: str
    action: str
    logic_lines: list[str]
    marketing_brief: dict[str, dict[str, str]]


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


def _format_score(value: float | None) -> str:
    if value is None:
        return "Not scored"
    return f"{value:.2f}"


def _format_days(value: int | None) -> str:
    if value is None:
        return "Unknown"
    day_label = "day" if value == 1 else "days"
    return f"{value} {day_label}"


def _format_points(value: float | None) -> str:
    if value is None:
        return "Unknown"
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
        latest_snapshot["points_gap"] = (
            latest_snapshot["reward_threshold_points"] -
            latest_snapshot["totalPoints"]
        ).clip(lower=0)
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
        merged["points_close"] = (
            merged["points_gap"].fillna(POINTS_CLOSE_THRESHOLD + 1) <=
            POINTS_CLOSE_THRESHOLD
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
        return CRMOutput(
            segment_id=rule["segment_id"],
            segment_name=rule["segment_name"],
            campaign=self._render_template(rule["campaign"], template_context),
            action=self._render_template(rule["action"], template_context),
            logic_lines=self._build_logic_lines(row, rule),
            marketing_brief=self._build_marketing_brief(row),
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

    def _build_logic_lines(
        self,
        row: pd.Series,
        rule: dict[str, str],
    ) -> list[str]:
        lines = [self._physiological_line(row)]
        if _truthy_flag(row["physiological_churn"]):
            lines.append(
                "Lifecycle exit short-circuits churn, engagement, and points checks."
            )
        else:
            lines.append(self._score_line(
                "Churn score",
                _clean_float(row["churn_30_to_60_prob"]),
                CHURN_THRESHOLD,
                default_note="The score is missing in the current export, so the engine falls back to low churn.",
            ))
            if _truthy_flag(row["high_churn"]):
                lines.append(self._score_line(
                    "Re-engage30d",
                    _clean_float(row["re_engage_30d_prob"]),
                    ENGAGEMENT_THRESHOLD,
                    default_note="The score is missing in the current export, so the engine falls back to low engagement.",
                ))
                lines.append(self._points_line(row))
            else:
                lines.append(self._points_line(row))
                if not _truthy_flag(row["points_close"]):
                    lines.append(self._score_line(
                        "Re-engage30d",
                        _clean_float(row["re_engage_30d_prob"]),
                        ENGAGEMENT_THRESHOLD,
                        default_note="The score is missing in the current export, so the engine falls back to low engagement.",
                    ))

        lines.append(
            f"First matching rule in priority order: `{rule['rule_name']}` -> {rule['segment_id']}."
        )
        return lines

    def _build_marketing_brief(self, row: pd.Series) -> dict[str, dict[str, str]]:
        points_gap = _clean_float(row["points_gap"])
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
            "points_gap": {
                "label": "Points gap",
                "value": _format_points(points_gap),
                "guidance": "Use the exact number in copy when the user is close to a reward.",
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
        sortable["points_sort"] = sortable["points_gap"].fillna(999999.0)

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
    def _physiological_line(row: pd.Series) -> str:
        child_age = _clean_int(row["ETA_MM_BambinoTODAY"])
        if _truthy_flag(row["physiological_churn"]):
            return (
                "Physiological churn is `Yes` because the user is already in the"
                f" lifecycle transition window ({child_age} months)."
            )
        return (
            "Physiological churn is `No`, so the engine continues to churn,"
            " engagement, and points checks."
        )

    @staticmethod
    def _score_line(
        label: str,
        score: float | None,
        threshold: float,
        *,
        default_note: str,
    ) -> str:
        if score is None:
            return f"{label} is not available. {default_note}"
        state = "High" if score >= threshold else "Low"
        return (
            f"{label} is {_format_score(score)} ({state}) against the"
            f" {threshold:.2f} split."
        )

    @staticmethod
    def _points_line(row: pd.Series) -> str:
        points_gap = _clean_float(row["points_gap"])
        if points_gap is None:
            return "Points gap is unknown, so the engine treats proximity as `Far`."
        proximity = "Close" if _truthy_flag(row["points_close"]) else "Far"
        return (
            f"Points gap is {_format_points(points_gap)}, which is `{proximity}`"
            f" relative to the {POINTS_CLOSE_THRESHOLD}-point trigger."
        )

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
            return "Recent scanners respond well to positive reinforcement and momentum language."
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

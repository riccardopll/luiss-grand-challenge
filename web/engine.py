from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass
class RuleView:
    audience: str
    matched_rule: str
    condition: str
    segment: str
    action: str
    channel: str
    priority: str
    score: float | None
    score_label: str
    eligible: bool
    summary: str
    logic_lines: list[str]


@dataclass
class SuggestedAction:
    title: str
    channel: str
    priority: str
    campaign_label: str
    summary: str
    next_steps: list[str]


@dataclass
class UserPayload:
    user_id: str
    reference_date: str
    profile: dict[str, object]
    churn: RuleView
    engagement: RuleView
    suggested_action: SuggestedAction


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, type(pd.NA))) else False


def _clean_text(value: object, fallback: str = "Unknown") -> str:
    if _is_missing(value):
        return fallback
    txt = str(value).strip()
    return txt if txt else fallback


def _clean_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _clean_int(value: object) -> int | None:
    numeric = _clean_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _priority_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value.lower(), 0)


def _score_band(probability: float | None) -> str:
    if probability is None:
        return "Not scored"
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


def _campaign_label(value: str) -> str:
    return value.replace("_", " ").strip().title()


class CRMDecisionEngine:
    def __init__(self, *, project_root: Path, artifacts_dir: Path, rules_path: Path) -> None:
        self.project_root = project_root
        self.artifacts_dir = artifacts_dir
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self._load_data()

    def _load_data(self) -> None:
        training_artifacts = pd.read_pickle(
            self.artifacts_dir / "eda_training_artifacts.pkl")

        user_base = training_artifacts["user_base"].copy()
        user_base["idSSO"] = user_base["idSSO"].astype(str)
        profile_columns = ["idSSO", "Regione", "registration_date"]
        self.user_profile = user_base[profile_columns].drop_duplicates(
            subset="idSSO", keep="last")

        snapshots = training_artifacts["snapshot_features"].copy()
        snapshots["reference_date"] = pd.to_datetime(
            snapshots["reference_date"])
        latest_reference = snapshots["reference_date"].max()
        latest_snapshot = snapshots[snapshots["reference_date"]
                                    == latest_reference].copy()
        latest_snapshot["idSSO"] = latest_snapshot["idSSO"].astype(str)
        latest_snapshot["points_gap_proxy"] = (
            latest_snapshot["reward_threshold_points"] -
            latest_snapshot["totalPoints"].fillna(0)
        ).clip(lower=0)

        self.latest_reference_date = latest_reference.strftime("%Y-%m-%d")
        self.snapshot = latest_snapshot.merge(
            self.user_profile, on="idSSO", how="left")

        churn = pd.read_csv(self.artifacts_dir / "final" /
                            "churn_engine_current.csv")
        churn["idSSO"] = churn["idSSO"].astype(str)
        self.churn = churn

        engagement = pd.read_csv(
            self.artifacts_dir / "final" / "engagement_engine_current.csv")
        engagement["idSSO"] = engagement["idSSO"].astype(str)
        self.engagement = engagement

        sample_frame = churn.merge(
            engagement,
            on="idSSO",
            how="inner",
            suffixes=("_churn", "_engagement"),
        )
        if sample_frame.empty:
            self.sample_user_ids = sorted(churn["idSSO"].head(6).tolist())
        else:
            sample_frame["priority_rank"] = sample_frame[["priority_churn", "priority_engagement"]].apply(
                lambda row: max(_priority_rank(
                    str(row.iloc[0])), _priority_rank(str(row.iloc[1]))),
                axis=1,
            )
            sample_frame = sample_frame.sort_values(
                ["priority_rank", "churn_30_to_60_prob", "re_engage_30d_prob"],
                ascending=[False, False, False],
            )
            self.sample_user_ids = sample_frame["idSSO"].head(6).tolist()

    def _load_rules(self) -> dict[tuple[str, str], dict[str, str]]:
        rules: dict[tuple[str, str], dict[str, str]] = {}
        for raw_line in self.rules_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts: dict[str, str] = {}
            for part in line.split("|"):
                key, value = part.split("=", 1)
                parts[key] = value
            rules[(parts["audience"], parts["rule_name"])] = parts
        return rules

    def lookup(self, user_id: str) -> UserPayload | None:
        snapshot_match = self.snapshot[self.snapshot["idSSO"] == user_id]
        if snapshot_match.empty:
            return None

        profile_row = snapshot_match.iloc[0]
        churn_row = self._first_row(self.churn, user_id)
        engagement_row = self._first_row(self.engagement, user_id)

        profile = self._build_profile(profile_row, churn_row, engagement_row)
        churn_view = self._build_churn_view(profile_row, churn_row)
        engagement_view = self._build_engagement_view(
            profile_row, engagement_row)
        suggested_action = self._build_suggested_action(
            profile_row, churn_view, engagement_view)

        return UserPayload(
            user_id=user_id,
            reference_date=self.latest_reference_date,
            profile=profile,
            churn=churn_view,
            engagement=engagement_view,
            suggested_action=suggested_action,
        )

    @staticmethod
    def _first_row(frame: pd.DataFrame, user_id: str) -> pd.Series | None:
        subset = frame[frame["idSSO"] == user_id]
        if subset.empty:
            return None
        return subset.iloc[0]

    def _build_profile(
        self,
        profile_row: pd.Series,
        churn_row: pd.Series | None,
        engagement_row: pd.Series | None,
    ) -> dict[str, object]:
        churn_prob = _clean_float(
            churn_row["churn_30_to_60_prob"]) if churn_row is not None else None
        engagement_prob = _clean_float(
            engagement_row["re_engage_30d_prob"]) if engagement_row is not None else None
        total_points = _clean_float(profile_row["totalPoints"])
        return {
            "region": _clean_text(profile_row["Regione"]),
            "child_age_bucket": _clean_text(profile_row["child_age_bucket"]),
            "child_age_months": _clean_int(profile_row["ETA_MM_BambinoTODAY"]),
            "days_since_last_activity": _clean_int(profile_row["days_since_last_activity"]),
            "total_points": total_points,
            "points_gap_proxy": _clean_float(profile_row["points_gap_proxy"]),
            "churn_risk_label": _score_band(churn_prob),
            "re_engage_label": _score_band(engagement_prob),
            "push_consent": bool(_clean_int(profile_row["channel_eligible_phone_push"]) == 1),
            "email_consent": bool(_clean_int(profile_row["channel_eligible_email"]) == 1),
        }

    def _build_churn_view(self, profile_row: pd.Series, churn_row: pd.Series | None) -> RuleView:
        if churn_row is None:
            return RuleView(
                audience="churn",
                matched_rule="not_scored",
                condition="days_since_last_activity must be between 30 and 59 days",
                segment="not_eligible",
                action="no_action",
                channel="none",
                priority="low",
                score=None,
                score_label="Not scored",
                eligible=False,
                summary="User is outside the 30-59 day inactivity window, so the churn engine is not active.",
                logic_lines=[
                    f"Latest inactivity is {_clean_int(
                        profile_row['days_since_last_activity'])} days.",
                    "Churn scoring only runs for users currently 30-59 days inactive.",
                ],
            )

        matched_rule = _clean_text(churn_row["matched_rule"])
        rule = self.rules[("churn", matched_rule)]
        score = _clean_float(churn_row["churn_30_to_60_prob"])
        return RuleView(
            audience="churn",
            matched_rule=matched_rule,
            condition=rule["condition"],
            segment=_clean_text(churn_row["churn_type"]),
            action=_clean_text(churn_row["recommended_campaign"]),
            channel=_clean_text(churn_row["recommended_channel"]),
            priority=_clean_text(churn_row["priority"]),
            score=score,
            score_label=_score_band(score),
            eligible=True,
            summary=self._summarize_rule("churn", matched_rule, churn_row),
            logic_lines=self._logic_lines(
                rule["condition"], profile_row, churn_row),
        )

    def _build_engagement_view(
        self,
        profile_row: pd.Series,
        engagement_row: pd.Series | None,
    ) -> RuleView:
        if engagement_row is None:
            return RuleView(
                audience="engagement",
                matched_rule="not_scored",
                condition="user needs historical activity to enter the re-engagement audience",
                segment="not_eligible",
                action="no_action",
                channel="none",
                priority="low",
                score=None,
                score_label="Not scored",
                eligible=False,
                summary="User is outside the current re-engagement audience.",
                logic_lines=[
                    f"Latest activity is {_clean_int(profile_row['days_since_last_activity'])} days ago.",
                    "Re-engagement scoring is reserved for users with observed historical activity.",
                ],
            )

        matched_rule = _clean_text(engagement_row["matched_rule"])
        rule = self.rules[("engagement", matched_rule)]
        score = _clean_float(engagement_row["re_engage_30d_prob"])
        return RuleView(
            audience="engagement",
            matched_rule=matched_rule,
            condition=rule["condition"],
            segment=_clean_text(engagement_row["engagement_segment"]),
            action=_clean_text(engagement_row["recommended_action"]),
            channel=_clean_text(engagement_row["recommended_channel"]),
            priority=_clean_text(engagement_row["priority"]),
            score=score,
            score_label=_score_band(score),
            eligible=True,
            summary=self._summarize_rule(
                "engagement", matched_rule, engagement_row),
            logic_lines=self._logic_lines(
                rule["condition"], profile_row, engagement_row),
        )

    def _build_suggested_action(
        self,
        profile_row: pd.Series,
        churn: RuleView,
        engagement: RuleView,
    ) -> SuggestedAction:
        points_gap = _clean_float(profile_row["points_gap_proxy"]) or 0.0

        if churn.eligible and engagement.eligible and _priority_rank(churn.priority) >= 2 and points_gap <= 2000:
            channel = churn.channel if churn.channel != "none" else engagement.channel
            return SuggestedAction(
                title="Reward-led reactivation",
                channel=channel,
                priority="high" if churn.priority == "high" else "medium",
                campaign_label="Reactivation With Reward Angle",
                summary="Use the churn signal to trigger outreach, but frame the message around reward momentum to encourage return.",
                next_steps=[
                    f"Contact through {
                        channel} today with a reward-focused reactivation message.",
                    "Reference the points balance or reward threshold to make the return value concrete.",
                    f"Route the user into `{churn.action}` with `{
                        engagement.action}` as the creative angle.",
                ],
            )

        chosen = churn if _priority_rank(churn.priority) >= _priority_rank(
            engagement.priority) else engagement
        actionable = chosen.eligible and chosen.channel != "none" and chosen.action not in {
            "no_action",
            "insufficient_channel_eligibility",
        }
        label = _campaign_label(chosen.action)
        summary = (
            "The top-priority engine result should drive CRM activation for this user."
            if actionable
            else "The current rules do not support an outbound campaign for this user in the latest snapshot."
        )
        return SuggestedAction(
            title=label if actionable else "No active campaign",
            channel=chosen.channel,
            priority=chosen.priority,
            campaign_label=label,
            summary=summary,
            next_steps=[
                f"Use `{chosen.matched_rule}` as the governing rule.",
                f"Preferred channel: {chosen.channel}.",
                f"Execution priority: {chosen.priority}.",
            ],
        )

    def _summarize_rule(self, audience: str, matched_rule: str, rule_row: pd.Series) -> str:
        if matched_rule.startswith("default"):
            if audience == "churn":
                return "No higher-priority churn rule matched, so the fallback retention stance applies."
            return "No tighter re-engagement trigger matched, so the default nurture action applies."
        if audience == "churn":
            return (
                f"Matched `{
                    matched_rule}` because the user is in an actionable inactivity state "
                f"with `{_clean_text(
                    rule_row['recommended_channel'])}` available."
            )
        return (
            f"Matched `{
                matched_rule}` because the user fits the current re-engagement scenario "
            f"for `{_clean_text(rule_row['recommended_channel'])}` outreach."
        )

    def _logic_lines(
        self,
        condition: str,
        profile_row: pd.Series,
        rule_row: pd.Series,
    ) -> list[str]:
        if condition == "True":
            return self._fallback_logic(profile_row, rule_row)

        lines: list[str] = []
        for clause in [part.strip() for part in condition.split("&")]:
            explanation = self._explain_clause(clause, profile_row, rule_row)
            if explanation:
                lines.append(explanation)
        return lines or self._fallback_logic(profile_row, rule_row)

    def _fallback_logic(self, profile_row: pd.Series, rule_row: pd.Series) -> list[str]:
        lines = []
        churn_prob = _clean_float(rule_row.get("churn_30_to_60_prob")) if isinstance(
            rule_row, pd.Series) else None
        engagement_prob = _clean_float(rule_row.get("re_engage_30d_prob")) if isinstance(
            rule_row, pd.Series) else None
        if churn_prob is not None:
            lines.append(f"Churn probability is {
                         churn_prob:.1%}, below the strongest save-rule thresholds.")
        if engagement_prob is not None:
            lines.append(f"Re-engagement probability is {
                         engagement_prob:.1%}, so a light nurture action is sufficient.")
        if not lines:
            lines.append(
                "No stronger rule condition was satisfied in the latest snapshot.")
        lines.append(
            f"Latest inactivity is {_clean_int(
                profile_row['days_since_last_activity'])} days with "
            f"{self._points_text(_clean_float(
                profile_row['totalPoints']))} available."
        )
        return lines

    def _explain_clause(self, clause: str, profile_row: pd.Series, rule_row: pd.Series) -> str | None:
        match = re.match(
            r"([a-zA-Z0-9_]+)\s*(>=|<=|==|>|<)\s*([a-zA-Z0-9_.]+)", clause)
        if not match:
            return clause

        field, operator, raw_value = match.groups()
        threshold = float(raw_value) if raw_value.replace(
            ".", "", 1).isdigit() else raw_value
        source_row = rule_row if field in rule_row.index else profile_row
        current_value = source_row.get(field)

        if field == "churn_30_to_60_prob":
            current = _clean_float(current_value) or 0.0
            return f"Churn probability is {current:.1%}, which satisfies `{field} {operator} {float(threshold):.2f}`."
        if field == "re_engage_30d_prob":
            current = _clean_float(current_value) or 0.0
            return f"Re-engagement probability is {current:.1%}, which satisfies `{field} {operator} {float(threshold):.2f}`."
        if field == "points_gap_proxy":
            current = _clean_float(current_value) or 0.0
            return f"Points gap is {current:,.0f}, matching `{field} {operator} {int(float(threshold))}`."
        if field == "is_near_graduation":
            state = "close to lifecycle transition" if _clean_int(
                current_value) == 1 else "not close to lifecycle transition"
            return f"Lifecycle status is {state}."
        if field == "channel_eligible_phone_push":
            state = "Push is allowed by consent." if _clean_int(
                current_value) == 1 else "Push is not allowed by consent."
            return state
        if field == "channel_eligible_email":
            state = "Email is allowed by consent." if _clean_int(
                current_value) == 1 else "Email is not allowed by consent."
            return state
        if field == "is_points_user":
            state = "User has an observable points balance." if _clean_int(
                current_value) == 1 else "User has no observable points balance."
            return state
        if field == "has_ever_redeemed":
            state = "User has redeemed at least once before." if _clean_int(
                current_value) == 1 else "User has not redeemed before."
            return state
        return f"{field} is `{current_value}` and satisfies `{clause}`."

    @staticmethod
    def _points_text(points: float | None) -> str:
        if points is None:
            return "an unknown points balance"
        return f"{points:,.0f} points"

from __future__ import annotations

import os
import textwrap
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import streamlit as st


APP_TITLE = "REI Survey Intelligence Dashboard"
PURPOSE = "Helping Rofiyat NGOs understand community needs and make better decisions using survey data."
EXPECTED_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Location",
    "Education Level",
    "Employment Status",
    "Main Challenge",
    "Program Interest",
    "Digital Skill Level",
    "Health Awareness Level",
    "Financial Literacy Level",
    "Satisfaction Score",
    "Recommendation",
]
SUMMARY_COLUMNS = EXPECTED_COLUMNS + ["Age Group"]
RISK_COLUMNS = SUMMARY_COLUMNS + ["Risk Score", "Risk Level", "Urgent Action", "Risk Drivers"]
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_data.csv"
CHART_COLORS = ["#1d4ed8", "#047857", "#b45309", "#be123c", "#6d28d9", "#0f766e", "#a21caf", "#475569"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    }
    [data-testid="stMetricLabel"] {
        color: #475569;
    }
    div[data-testid="stSidebar"] {
        background: #eef2f7;
    }
    h1, h2, h3 {
        color: #0f172a;
        letter-spacing: 0;
    }
    .app-kicker {
        color: #475569;
        font-size: 0.98rem;
        margin-bottom: 0.75rem;
    }
    .section-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -0.5rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in EXPECTED_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    text_columns = [column for column in EXPECTED_COLUMNS if column not in {"Age", "Satisfaction Score"}]
    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .fillna("Not provided")
        )

    cleaned["Age"] = pd.to_numeric(cleaned["Age"], errors="coerce")
    cleaned["Satisfaction Score"] = pd.to_numeric(cleaned["Satisfaction Score"], errors="coerce").clip(
        lower=1,
        upper=5,
    )
    cleaned["Age Group"] = pd.cut(
        cleaned["Age"],
        bins=[0, 17, 24, 34, 44, 54, 64, 120],
        labels=["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=True,
    )
    cleaned["Age Group"] = cleaned["Age Group"].astype("string").fillna("Not provided")
    return cleaned


def filter_options(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(value for value in df[column].dropna().astype(str).unique() if value)


def apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = df.copy()
    for column, selected_values in filters.items():
        if selected_values:
            filtered = filtered[filtered[column].isin(selected_values)]
    return filtered


def filter_summary(filters: dict[str, list[str]], all_options: dict[str, list[str]] | None = None) -> str:
    active = []
    for column, selected_values in filters.items():
        if selected_values and (not all_options or set(selected_values) != set(all_options.get(column, []))):
            active.append(f"{column}: {', '.join(selected_values)}")
    return "No filters applied" if not active else "; ".join(active)


def value_counts(df: pd.DataFrame, column: str, limit: int | None = None) -> pd.Series:
    counts = df[column].fillna("Not provided").value_counts()
    if limit:
        counts = counts.head(limit)
    return counts


def top_value(df: pd.DataFrame, column: str) -> str:
    counts = value_counts(df, column)
    if counts.empty:
        return "Not available"
    return str(counts.index[0])


def top_count_share(df: pd.DataFrame, column: str) -> tuple[str, int, float]:
    counts = value_counts(df, column)
    if counts.empty or len(df) == 0:
        return "Not available", 0, 0.0
    label = str(counts.index[0])
    count = int(counts.iloc[0])
    share = count / len(df)
    return label, count, share


def average_satisfaction(df: pd.DataFrame) -> float:
    score = df["Satisfaction Score"].dropna()
    if score.empty:
        return 0.0
    return float(score.mean())


def low_satisfaction_count(df: pd.DataFrame) -> int:
    return int((df["Satisfaction Score"] <= 2).sum())


def support_priority_location(df: pd.DataFrame) -> str:
    needs_mask = (
        df["Digital Skill Level"].str.lower().isin(["beginner", "none"])
        | df["Financial Literacy Level"].str.lower().isin(["beginner", "none"])
        | (df["Satisfaction Score"] <= 2)
    )
    priority_counts = df.loc[needs_mask, "Location"].value_counts()
    if priority_counts.empty:
        return top_value(df, "Location")
    return str(priority_counts.index[0])


def lowest_satisfaction_challenge(df: pd.DataFrame) -> str:
    challenge_scores = (
        df.dropna(subset=["Satisfaction Score"])
        .groupby("Main Challenge", dropna=False)["Satisfaction Score"]
        .mean()
        .sort_values()
    )
    if challenge_scores.empty:
        return top_value(df, "Main Challenge")
    return str(challenge_scores.index[0])


def strongest_program_location_pair(df: pd.DataFrame) -> tuple[str, str, int]:
    grouped = (
        df.groupby(["Location", "Program Interest"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return "Not available", "Not available", 0
    location, program = grouped.index[0]
    return str(program), str(location), int(grouped.iloc[0])


def score_beneficiary(row: pd.Series) -> tuple[int, str, str]:
    score = 0
    drivers: list[str] = []

    challenge_weights = {
        "unemployment": 20,
        "access to healthcare": 20,
        "mental health support": 20,
        "limited funding": 18,
        "lack of digital skills": 18,
        "financial planning": 14,
        "childcare support": 14,
        "market access": 12,
    }
    challenge = str(row["Main Challenge"]).strip().lower()
    challenge_score = challenge_weights.get(challenge, 8)
    score += challenge_score
    drivers.append(f"challenge: {row['Main Challenge']}")

    employment_status = str(row["Employment Status"]).strip().lower()
    if employment_status == "unemployed":
        score += 20
        drivers.append("unemployed")
    elif employment_status == "self-employed":
        score += 8
        drivers.append("income stability")

    digital_level = str(row["Digital Skill Level"]).strip().lower()
    if digital_level in {"none", "beginner"}:
        score += 15
        drivers.append("low digital skills")
    elif digital_level == "intermediate":
        score += 5

    financial_level = str(row["Financial Literacy Level"]).strip().lower()
    if financial_level in {"none", "beginner"}:
        score += 15
        drivers.append("low financial literacy")
    elif financial_level == "intermediate":
        score += 5

    health_level = str(row["Health Awareness Level"]).strip().lower()
    if health_level == "low":
        score += 12
        drivers.append("low health awareness")
    elif health_level == "medium":
        score += 5

    satisfaction = row["Satisfaction Score"]
    if pd.notna(satisfaction) and satisfaction <= 2:
        score += 15
        drivers.append("low satisfaction")
    elif pd.notna(satisfaction) and satisfaction == 3:
        score += 7

    education_level = str(row["Education Level"]).strip().lower()
    if education_level == "primary":
        score += 10
        drivers.append("primary education")
    elif education_level == "secondary":
        score += 5

    age = row["Age"]
    if pd.notna(age) and (age < 18 or age >= 60):
        score += 5
        drivers.append("age vulnerability")

    score = min(score, 100)
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"

    return score, level, ", ".join(dict.fromkeys(drivers))


def build_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    risk_df = df.copy()
    scored = risk_df.apply(score_beneficiary, axis=1, result_type="expand")
    risk_df["Risk Score"] = scored[0].astype(int)
    risk_df["Risk Level"] = scored[1]
    risk_df["Risk Drivers"] = scored[2]
    risk_df["Urgent Action"] = risk_df.apply(urgent_action, axis=1)
    return risk_df.sort_values(["Risk Score", "Satisfaction Score"], ascending=[False, True])


def urgent_action(row: pd.Series) -> str:
    if row["Risk Score"] >= 70:
        if "health" in str(row["Risk Drivers"]).lower():
            return "Immediate welfare check and health referral"
        if "unemployed" in str(row["Risk Drivers"]).lower():
            return "Priority enrollment and livelihood follow-up"
        return "Immediate case review"
    if row["Risk Score"] >= 40:
        return "Schedule targeted follow-up"
    return "Monitor through routine program feedback"


def most_affected_group(risk_df: pd.DataFrame) -> tuple[str, str, float, int]:
    candidates = []
    for column in ["Location", "Education Level", "Gender", "Age Group"]:
        grouped = (
            risk_df.groupby(column, dropna=False)
            .agg(avg_risk=("Risk Score", "mean"), respondents=("Name", "size"))
            .reset_index()
        )
        if grouped.empty:
            continue
        grouped = grouped[grouped["respondents"] >= max(1, min(2, len(risk_df)))]
        if grouped.empty:
            continue
        grouped["group_type"] = column
        candidates.append(grouped)

    if not candidates:
        return "Not available", "Group", 0.0, 0

    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.sort_values(["avg_risk", "respondents"], ascending=[False, False])
    top_group = combined.iloc[0]
    return (
        str(top_group[top_group["group_type"]]),
        str(top_group["group_type"]),
        float(top_group["avg_risk"]),
        int(top_group["respondents"]),
    )


def build_key_findings(df: pd.DataFrame, risk_df: pd.DataFrame) -> list[dict[str, str]]:
    challenge, challenge_count, challenge_share = top_count_share(df, "Main Challenge")
    program, program_count, program_share = top_count_share(df, "Program Interest")
    group, group_type, avg_risk, group_count = most_affected_group(risk_df)

    return [
        {
            "label": "Most common challenge",
            "value": challenge,
            "detail": f"{challenge_count} respondents selected this ({challenge_share:.0%} of the current view).",
        },
        {
            "label": "Most demanded program",
            "value": program,
            "detail": f"{program_count} respondents requested this ({program_share:.0%} of the current view).",
        },
        {
            "label": "Most affected group",
            "value": f"{group} ({group_type})",
            "detail": f"{group_count} respondents, average vulnerability score {avg_risk:.0f}/100.",
        },
    ]


def build_interpretation(df: pd.DataFrame, risk_df: pd.DataFrame) -> list[str]:
    challenge = top_value(df, "Main Challenge")
    program = top_value(df, "Program Interest")
    location = top_value(df, "Location")
    group, group_type, avg_risk, _ = most_affected_group(risk_df)
    high_risk_count = int((risk_df["Risk Level"] == "High").sum())
    moderate_risk_count = int((risk_df["Risk Level"] == "Moderate").sum())
    avg_score = average_satisfaction(df)

    return [
        f"The data shows that {challenge} is the clearest pressure point, while {program} is the strongest program demand.",
        f"Demand is concentrated around {location}, so location-based planning will help the NGO deploy resources more efficiently.",
        f"The most affected segment is {group} under {group_type}, with an average vulnerability score of {avg_risk:.0f}/100.",
        f"There are {high_risk_count} high-risk and {moderate_risk_count} moderate-risk respondents in this filtered view.",
        f"Program satisfaction averages {avg_score:.1f}/5, so recommendations should balance new program delivery with follow-up support.",
    ]


def build_ai_recommendations(df: pd.DataFrame, risk_df: pd.DataFrame) -> list[dict[str, str]]:
    program, location, pair_count = strongest_program_location_pair(df)
    challenge = top_value(df, "Main Challenge")
    support_location = support_priority_location(df)
    avg_score = average_satisfaction(df)
    high_risk_count = int((risk_df["Risk Level"] == "High").sum())
    digital_gap_share = risk_df["Digital Skill Level"].str.lower().isin(["none", "beginner"]).mean()
    financial_gap_share = risk_df["Financial Literacy Level"].str.lower().isin(["none", "beginner"]).mean()
    low_health_share = (risk_df["Health Awareness Level"].str.lower() == "low").mean()

    recommendations = [
        {
            "priority": "High",
            "action": f"Prioritize {program} in {location}.",
            "reason": f"This is the strongest program-location pattern with {pair_count} matching responses.",
        },
        {
            "priority": "High" if high_risk_count else "Medium",
            "action": f"Run targeted follow-up in {support_location}.",
            "reason": f"{high_risk_count} respondents are currently high risk based on skills, needs, and satisfaction signals.",
        },
        {
            "priority": "Medium",
            "action": f"Design the next intervention around {challenge}.",
            "reason": "This is the most common barrier in the current filtered dataset.",
        },
    ]

    if digital_gap_share >= 0.35:
        recommendations.append(
            {
                "priority": "High",
                "action": "Add beginner-friendly digital skills training.",
                "reason": f"{digital_gap_share:.0%} of respondents report beginner or no digital skills.",
            }
        )

    if financial_gap_share >= 0.35:
        recommendations.append(
            {
                "priority": "High",
                "action": "Include practical financial literacy coaching.",
                "reason": f"{financial_gap_share:.0%} of respondents report beginner or no financial literacy.",
            }
        )

    if low_health_share >= 0.25:
        recommendations.append(
            {
                "priority": "Medium",
                "action": "Pair program delivery with health awareness outreach.",
                "reason": f"{low_health_share:.0%} of respondents report low health awareness.",
            }
        )

    if avg_score < 3.5:
        recommendations.append(
            {
                "priority": "High",
                "action": "Interview low-satisfaction respondents before scaling the next activity.",
                "reason": f"The average satisfaction score is {avg_score:.1f}/5.",
            }
        )

    return recommendations[:6]


def build_insights(df: pd.DataFrame) -> list[str]:
    program, program_count, program_share = top_count_share(df, "Program Interest")
    challenge, challenge_count, challenge_share = top_count_share(df, "Main Challenge")
    location, location_count, location_share = top_count_share(df, "Location")
    education, education_count, education_share = top_count_share(df, "Education Level")
    avg_score = average_satisfaction(df)
    low_scores = low_satisfaction_count(df)

    return [
        f"Most respondents need {program}: {program_count} of {len(df)} respondents selected it ({program_share:.0%}).",
        f"The most common challenge is {challenge}, reported by {challenge_count} respondents ({challenge_share:.0%}).",
        f"{location} has the highest response demand with {location_count} respondents ({location_share:.0%}).",
        f"{education} is the largest education group in this view, representing {education_count} respondents ({education_share:.0%}).",
        f"Average satisfaction is {avg_score:.1f}/5, with {low_scores} respondents scoring 2 or below.",
    ]


def build_recommendations(df: pd.DataFrame) -> list[str]:
    program, location, pair_count = strongest_program_location_pair(df)
    support_location = support_priority_location(df)
    challenge = lowest_satisfaction_challenge(df)
    top_challenge = top_value(df, "Main Challenge")
    low_scores = low_satisfaction_count(df)

    recommendations = [
        f"Prioritize {program} in {location}; this is the strongest program-location demand signal ({pair_count} responses).",
        f"Run targeted support in {support_location}, especially for respondents with beginner digital or financial skills.",
        f"Design the next intervention around {top_challenge}, while investigating why {challenge} is linked to lower satisfaction.",
    ]
    if low_scores:
        recommendations.append(f"Follow up with the {low_scores} respondents who gave low satisfaction scores before the next program cycle.")
    else:
        recommendations.append("Maintain the current satisfaction level by collecting short feedback after each program session.")
    return recommendations


def build_report(
    df: pd.DataFrame,
    key_findings: list[dict[str, str]],
    interpretation: list[str],
    recommendations: list[dict[str, str]],
    risk_df: pd.DataFrame,
    filters: dict[str, list[str]],
    filter_choices: dict[str, list[str]],
    source_label: str,
    total_rows: int,
) -> str:
    total = len(df)
    location = top_value(df, "Location")
    challenge = top_value(df, "Main Challenge")
    program = top_value(df, "Program Interest")
    avg_score = average_satisfaction(df)
    high_risk_count = int((risk_df["Risk Level"] == "High").sum())
    moderate_risk_count = int((risk_df["Risk Level"] == "Moderate").sum())

    lines = [
        APP_TITLE,
        "",
        PURPOSE,
        "",
        "Data scope",
        f"- Source: {source_label}",
        f"- Responses analyzed: {total} of {total_rows}",
        f"- Active filters: {filter_summary(filters, filter_choices)}",
        "",
        "Executive summary",
        f"- Total respondents in current view: {total}",
        f"- Most common location: {location}",
        f"- Most selected challenge: {challenge}",
        f"- Most requested program: {program}",
        f"- Average satisfaction score: {avg_score:.1f}/5",
        f"- High-risk respondents: {high_risk_count}",
        f"- Moderate-risk respondents: {moderate_risk_count}",
        "",
        "Key findings",
    ]
    lines.extend([f"- {finding['label']}: {finding['value']} - {finding['detail']}" for finding in key_findings])
    lines.extend(["", "Interpretation"])
    lines.extend([f"- {item}" for item in interpretation])
    lines.extend(["", "Recommendations"])
    lines.extend(
        [
            f"- [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}"
            for recommendation in recommendations
        ]
    )
    return "\n".join(lines)


def add_wrapped_text(ax, text: str, x: float, y: float, width: int, **kwargs) -> float:
    lines = textwrap.wrap(text, width=width) or [""]
    line_height = kwargs.pop("line_height", 0.026)
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, **kwargs)
        y -= line_height
    return y


def build_pdf_report(
    df: pd.DataFrame,
    key_findings: list[dict[str, str]],
    interpretation: list[str],
    recommendations: list[dict[str, str]],
    risk_df: pd.DataFrame,
    filters: dict[str, list[str]],
    filter_choices: dict[str, list[str]],
    source_label: str,
    total_rows: int,
) -> bytes:
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=120)
        ax.axis("off")
        fig.patch.set_facecolor("white")

        y = 0.96
        ax.text(0.06, y, APP_TITLE, fontsize=18, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.035
        y = add_wrapped_text(ax, PURPOSE, 0.06, y, 92, fontsize=9.5, color="#475569")
        y -= 0.02

        summary_lines = [
            f"Source: {source_label}",
            f"Responses analyzed: {len(df)} of {total_rows}",
            f"Active filters: {filter_summary(filters, filter_choices)}",
            f"Average satisfaction: {average_satisfaction(df):.1f}/5",
            f"High-risk respondents: {int((risk_df['Risk Level'] == 'High').sum())}",
            f"Moderate-risk respondents: {int((risk_df['Risk Level'] == 'Moderate').sum())}",
        ]
        ax.text(0.06, y, "Donor summary", fontsize=13, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.03
        for line in summary_lines:
            y = add_wrapped_text(ax, f"- {line}", 0.07, y, 92, fontsize=9.5, color="#1f2937")
        y -= 0.015

        ax.text(0.06, y, "Key findings", fontsize=13, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.03
        for finding in key_findings:
            line = f"{finding['label']}: {finding['value']} - {finding['detail']}"
            y = add_wrapped_text(ax, f"- {line}", 0.07, y, 90, fontsize=9.5, color="#1f2937")
        y -= 0.015

        ax.text(0.06, y, "Interpretation", fontsize=13, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.03
        for item in interpretation:
            y = add_wrapped_text(ax, f"- {item}", 0.07, y, 90, fontsize=9.5, color="#1f2937")
        y -= 0.015

        ax.text(0.06, y, "Recommendations", fontsize=13, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.03
        for recommendation in recommendations:
            line = f"[{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}"
            y = add_wrapped_text(ax, f"- {line}", 0.07, y, 90, fontsize=9.5, color="#1f2937")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=120)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        y = 0.96
        ax.text(0.06, y, "Risk scoring detail", fontsize=16, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        y -= 0.04
        y = add_wrapped_text(
            ax,
            "Scores combine unemployment, core challenge, low skills, low health awareness, low satisfaction, education, and age vulnerability. Higher scores indicate stronger need for follow-up.",
            0.06,
            y,
            92,
            fontsize=9.5,
            color="#475569",
        )
        y -= 0.025

        top_risk = risk_df[
            ["Name", "Location", "Program Interest", "Risk Score", "Risk Level", "Urgent Action", "Risk Drivers"]
        ].head(12)
        for _, row in top_risk.iterrows():
            line = (
                f"{row['Name']} | {row['Location']} | {row['Program Interest']} | "
                f"{row['Risk Score']}/100 {row['Risk Level']} | {row['Urgent Action']} | {row['Risk Drivers']}"
            )
            y = add_wrapped_text(ax, f"- {line}", 0.07, y, 98, fontsize=8.8, color="#1f2937", line_height=0.023)
            y -= 0.008

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    output.seek(0)
    return output.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned Responses", index=False)
    return output.getvalue()


def draw_bar_chart(
    counts: pd.Series,
    title: str,
    x_label: str = "Responses",
    horizontal: bool = False,
) -> None:
    if counts.empty:
        st.info("No responses available for this chart yet.")
        return

    chart_data = counts.sort_values(ascending=True if horizontal else False)
    fig_height = max(2.35, min(4.3, len(chart_data) * 0.38 + 0.9))
    fig, ax = plt.subplots(figsize=(7.2, fig_height), dpi=135)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    colors = [CHART_COLORS[index % len(CHART_COLORS)] for index in range(len(chart_data))]

    if horizontal:
        bars = ax.barh(chart_data.index.astype(str), chart_data.values, color=colors, height=0.58)
        ax.set_xlabel(x_label)
        ax.set_ylabel("")
        ax.bar_label(bars, padding=4, fontsize=8, color="#334155")
        ax.margins(x=0.14)
    else:
        bars = ax.bar(chart_data.index.astype(str), chart_data.values, color=colors, width=0.62)
        ax.set_ylabel(x_label)
        ax.set_xlabel("")
        ax.bar_label(bars, padding=3, fontsize=8, color="#334155")
        ax.tick_params(axis="x", labelrotation=24)
        ax.margins(y=0.18)

    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=8, color="#0f172a")
    ax.grid(axis="x" if horizontal else "y", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5, colors="#334155")
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def draw_donut_chart(counts: pd.Series, title: str) -> None:
    if counts.empty:
        st.info("No responses available for this chart yet.")
        return

    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=135)
    fig.patch.set_facecolor("#ffffff")
    colors = [CHART_COLORS[index % len(CHART_COLORS)] for index in range(len(counts))]
    ax.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white", "width": 0.45},
        textprops={"fontsize": 8.5, "color": "#334155"},
    )
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=8, color="#0f172a")
    ax.axis("equal")
    fig.tight_layout(pad=0.7)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


with st.sidebar:
    st.header("Survey Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.caption("Use exports from Google Forms, KoboToolbox, or Excel.")

    show_raw_data = st.toggle("Show data table", value=True)
    chart_limit = st.slider("Chart categories", min_value=5, max_value=15, value=8)


try:
    source_label = "Uploaded file" if uploaded_file else "Sample dataset"
    raw_df = load_uploaded_data(uploaded_file) if uploaded_file else load_sample_data()
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()


missing_columns = [column for column in EXPECTED_COLUMNS if column not in raw_df.columns]
df = clean_data(raw_df)


with st.sidebar:
    st.header("Filters")
    filter_choices = {
        "Gender": filter_options(df, "Gender"),
        "Location": filter_options(df, "Location"),
        "Program Interest": filter_options(df, "Program Interest"),
        "Education Level": filter_options(df, "Education Level"),
    }
    filters = {
        "Gender": st.multiselect("Gender", options=filter_choices["Gender"], default=filter_choices["Gender"]),
        "Location": st.multiselect("Location", options=filter_choices["Location"], default=filter_choices["Location"]),
        "Program Interest": st.multiselect(
            "Program interest",
            options=filter_choices["Program Interest"],
            default=filter_choices["Program Interest"],
        ),
        "Education Level": st.multiselect(
            "Education level",
            options=filter_choices["Education Level"],
            default=filter_choices["Education Level"],
        ),
    }


filtered_df = apply_filters(df, filters)

if filtered_df.empty:
    st.title(APP_TITLE)
    st.caption(PURPOSE)
    st.warning("No responses match the selected filters. Clear one or more filters to continue.")
    st.stop()


risk_df = build_risk_scores(filtered_df)
key_findings = build_key_findings(filtered_df, risk_df)
interpretation = build_interpretation(filtered_df, risk_df)
recommendations = build_ai_recommendations(filtered_df, risk_df)
report_text = build_report(
    filtered_df,
    key_findings,
    interpretation,
    recommendations,
    risk_df,
    filters,
    filter_choices,
    source_label,
    len(df),
)
report_pdf = build_pdf_report(
    filtered_df,
    key_findings,
    interpretation,
    recommendations,
    risk_df,
    filters,
    filter_choices,
    source_label,
    len(df),
)


with st.sidebar:
    st.header("Downloads")
    st.download_button(
        "Download cleaned CSV",
        data=dataframe_to_csv_bytes(filtered_df[SUMMARY_COLUMNS]),
        file_name="cleaned_survey_responses.csv",
        mime="text/csv",
        width="stretch",
    )
    st.download_button(
        "Download report",
        data=report_text,
        file_name="rei_survey_intelligence_report.txt",
        mime="text/plain",
        width="stretch",
    )
    st.download_button(
        "Download PDF report",
        data=report_pdf,
        file_name="rei_survey_donor_report.pdf",
        mime="application/pdf",
        width="stretch",
    )


st.title(APP_TITLE)
st.markdown(f"<div class='app-kicker'>{PURPOSE}</div>", unsafe_allow_html=True)

if missing_columns:
    st.warning(
        "The uploaded file is missing expected columns: "
        + ", ".join(missing_columns)
        + ". Missing fields were added as 'Not provided' so the dashboard can still run."
    )

st.caption(f"Analyzing {len(filtered_df):,} of {len(df):,} responses")
st.caption(f"Data source: {source_label} | {filter_summary(filters, filter_choices)}")

overview_tab, demographics_tab, needs_tab, insights_tab = st.tabs(
    ["Overview", "Demographics", "Needs Assessment", "Insights"]
)


with overview_tab:
    total_responses = len(filtered_df)
    common_location = top_value(filtered_df, "Location")
    common_challenge = top_value(filtered_df, "Main Challenge")
    avg_score = average_satisfaction(filtered_df)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4, gap="small")
    metric_1.metric("Filtered responses", f"{total_responses:,}", f"of {len(df):,}")
    metric_2.metric("Top location", common_location)
    metric_3.metric("Top challenge", common_challenge)
    metric_4.metric("Avg. satisfaction", f"{avg_score:.1f}/5")

    st.subheader("Response Snapshot")
    st.markdown(
        "<div class='section-note'>A compact view of where responses are coming from and how participants rated the program.</div>",
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns([1.15, 1.15, 0.9], gap="small")
    with left:
        draw_bar_chart(value_counts(filtered_df, "Location", chart_limit), "Responses by Location", horizontal=True)
    with middle:
        draw_bar_chart(value_counts(filtered_df, "Program Interest", chart_limit), "Program Demand", horizontal=True)
    with right:
        satisfaction_counts = (
            filtered_df["Satisfaction Score"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        draw_bar_chart(satisfaction_counts, "Satisfaction Scores")

    if show_raw_data:
        st.subheader("Cleaned Survey Responses")
        st.dataframe(filtered_df[SUMMARY_COLUMNS], width="stretch", hide_index=True)


with demographics_tab:
    st.subheader("Beneficiary and Participant Profile")
    st.markdown(
        "<div class='section-note'>Use these views to understand who the NGO is reaching and who may be missing.</div>",
        unsafe_allow_html=True,
    )
    col_1, col_2 = st.columns(2, gap="small")
    with col_1:
        draw_donut_chart(value_counts(filtered_df, "Gender"), "Gender Distribution")
        draw_bar_chart(value_counts(filtered_df, "Location", chart_limit), "Location Breakdown", horizontal=True)
    with col_2:
        draw_bar_chart(value_counts(filtered_df, "Age Group"), "Age Groups")
        draw_bar_chart(value_counts(filtered_df, "Education Level", chart_limit), "Education Level", horizontal=True)


with needs_tab:
    st.subheader("Needs Assessment")
    st.markdown(
        "<div class='section-note'>Compare stated challenges with the practical support areas people selected.</div>",
        unsafe_allow_html=True,
    )
    col_1, col_2, col_3 = st.columns(3, gap="small")
    with col_1:
        draw_bar_chart(value_counts(filtered_df, "Main Challenge", chart_limit), "Main Challenges", horizontal=True)
    with col_2:
        draw_bar_chart(value_counts(filtered_df, "Program Interest", chart_limit), "Program Interests", horizontal=True)
    with col_3:
        draw_bar_chart(value_counts(filtered_df, "Digital Skill Level"), "Digital Skill Level", horizontal=True)

    lower_1, lower_2 = st.columns(2, gap="small")
    with lower_1:
        draw_bar_chart(value_counts(filtered_df, "Financial Literacy Level"), "Financial Literacy", horizontal=True)
    with lower_2:
        draw_bar_chart(value_counts(filtered_df, "Health Awareness Level"), "Health Awareness", horizontal=True)


with insights_tab:
    st.subheader("Key Findings")
    finding_cols = st.columns(3, gap="small")
    for index, finding in enumerate(key_findings):
        with finding_cols[index]:
            st.metric(finding["label"], finding["value"])
            st.caption(finding["detail"])

    st.subheader("Interpretation")
    interpretation_cols = st.columns(2, gap="small")
    for index, item in enumerate(interpretation):
        with interpretation_cols[index % 2]:
            st.info(item)

    st.subheader("Recommendations")
    for recommendation in recommendations[:3]:
        st.success(f"{recommendation['priority']} priority: {recommendation['action']} {recommendation['reason']}")

    st.subheader("Action Focus")
    focus_1, focus_2, focus_3 = st.columns(3, gap="small")
    focus_1.metric("Support location", support_priority_location(filtered_df))
    focus_2.metric("Program priority", top_value(filtered_df, "Program Interest"))
    focus_3.metric("Challenge focus", lowest_satisfaction_challenge(filtered_df))

    st.subheader("PDF Report Download")
    st.markdown(
        "<div class='section-note'>A donor-ready PDF summary with key findings, interpretation, recommendations, and vulnerable beneficiary detail.</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download donor PDF report",
        data=report_pdf,
        file_name="rei_survey_donor_report.pdf",
        mime="application/pdf",
        width="stretch",
    )

    st.subheader("Vulnerability Scoring System")
    st.markdown(
        "<div class='section-note'>Identifies beneficiaries who may need urgent help. Scores combine challenge severity, employment, skills, health awareness, satisfaction, education, and age vulnerability.</div>",
        unsafe_allow_html=True,
    )
    high_risk = int((risk_df["Risk Level"] == "High").sum())
    moderate_risk = int((risk_df["Risk Level"] == "Moderate").sum())
    low_risk = int((risk_df["Risk Level"] == "Low").sum())
    risk_1, risk_2, risk_3 = st.columns(3, gap="small")
    risk_1.metric("High risk", high_risk)
    risk_2.metric("Moderate risk", moderate_risk)
    risk_3.metric("Low risk", low_risk)

    urgent_help_df = risk_df[risk_df["Risk Level"] == "High"]
    if urgent_help_df.empty:
        st.success("No beneficiaries are currently flagged for urgent help in this filtered view.")
    else:
        st.warning(f"{len(urgent_help_df)} beneficiaries are flagged for urgent help.")
        st.dataframe(
            urgent_help_df[["Name", "Location", "Main Challenge", "Risk Score", "Urgent Action"]].head(10),
            width="stretch",
            hide_index=True,
        )

    st.dataframe(
        risk_df[
            ["Name", "Location", "Program Interest", "Risk Score", "Risk Level", "Urgent Action", "Risk Drivers"]
        ].head(15),
        width="stretch",
        hide_index=True,
    )

    st.subheader("AI-style Recommendation Engine")
    st.markdown(
        "<div class='section-note'>Rule-based decision logic ranks actions from the strongest demand, vulnerability, satisfaction, and skill-gap signals.</div>",
        unsafe_allow_html=True,
    )
    recommendation_table = pd.DataFrame(
        [
            {
                "Priority": recommendation["priority"],
                "Recommended Action": recommendation["action"],
                "Decision Logic": recommendation["reason"],
            }
            for recommendation in recommendations
        ]
    )
    st.dataframe(recommendation_table, width="stretch", hide_index=True)

    st.subheader("Downloads")
    download_1, download_2, download_3, download_4 = st.columns(4, gap="small")
    download_1.download_button(
        "Download text report",
        data=report_text,
        file_name="rei_survey_intelligence_report.txt",
        mime="text/plain",
        width="stretch",
    )
    download_2.download_button(
        "Download cleaned CSV",
        data=dataframe_to_csv_bytes(filtered_df[SUMMARY_COLUMNS]),
        file_name="cleaned_survey_responses.csv",
        mime="text/csv",
        width="stretch",
    )
    download_3.download_button(
        "Download cleaned Excel",
        data=dataframe_to_excel_bytes(filtered_df[SUMMARY_COLUMNS]),
        file_name="cleaned_survey_responses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    download_4.download_button(
        "Download risk CSV",
        data=dataframe_to_csv_bytes(risk_df[RISK_COLUMNS]),
        file_name="beneficiary_risk_scores.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Preview generated report", expanded=True):
        st.text(report_text)

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

import boto3
import pandas as pd
from strands import Agent, tool

CANONICAL_COLUMNS = {
    "incidentyear": "Incident.year",
    "incidentdate": "Incident.date",
    "incidenthourofday": "Incident.hour.of.day",
    "victiminjury": "Victim.injury",
    "state": "State",
    "victimsurvivedordead": "Victim.survived.or.dead",
    "victimsex": "Victim.Sex",
    "victimage": "Victim.Age",
    "sharktype": "Shark.type",
}


def _clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@dataclass
class SharkDataStore:
    df: pd.DataFrame | None = None
    path: str | None = None
    missing_columns: list[str] | None = None


DATA_STORE = SharkDataStore()


def _normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    existing = set(df.columns)
    for col in df.columns:
        key = _clean_key(str(col))
        canonical = CANONICAL_COLUMNS.get(key)
        if canonical and canonical not in existing:
            rename_map[col] = canonical
            existing.add(canonical)

    df = df.rename(columns=rename_map)
    missing = [col for col in CANONICAL_COLUMNS.values() if col not in df.columns]
    return df, missing


def _require_data() -> pd.DataFrame:
    if DATA_STORE.df is None:
        raise ValueError("No dataset loaded. Use load_shark_attacks() with a file path first.")
    return DATA_STORE.df


def _apply_filters(
    df: pd.DataFrame,
    year: int | None = None,
    date: str | None = None,
    hour_of_day: str | None = None,
    survived_or_dead: str | None = None,
    injury: str | None = None,
    state: str | None = None,
    sex: str | None = None,
    age: str | None = None,
    shark_type: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str | int]] = []
    if year is not None:
        filters.append(("Incident.year", year))
    if date:
        filters.append(("Incident.date", date))
    if hour_of_day:
        filters.append(("Incident.hour.of.day", hour_of_day))
    if survived_or_dead:
        filters.append(("Victim.survived.or.dead", survived_or_dead))
    if injury:
        filters.append(("Victim.injury", injury))
    if state:
        filters.append(("State", state))
    if sex:
        filters.append(("Victim.Sex", sex))
    if age:
        filters.append(("Victim.Age", age))
    if shark_type:
        filters.append(("Shark.type", shark_type))

    for col, value in filters:
        if col not in df.columns:
            continue
        if isinstance(value, int):
            series = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
            df = df[series == value]
        else:
            series = df[col].astype(str)
            df = df[series.str.contains(str(value), case=False, na=False)]
    return df


def _fatal_mask(df: pd.DataFrame) -> pd.Series:
    if "Victim.survived.or.dead" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df["Victim.survived.or.dead"].astype(str)
    return series.str.contains(r"\bdead\b|fatal|deceased", case=False, na=False)


def _format_table(rows: Iterable[dict[str, Any]]) -> str:
    return json.dumps(list(rows), indent=2)


def _load_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_from_s3(s3_uri: str) -> pd.DataFrame:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")

    path = s3_uri[5:]
    if "/" not in path:
        raise ValueError("S3 URI must include bucket and key, e.g. s3://bucket/key")

    bucket, key = path.split("/", 1)
    client = boto3.client("s3")
    obj = client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    if key.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(data))
    return pd.read_excel(BytesIO(data))


@tool
def load_shark_attacks(path: str) -> dict:
    """
    Load a shark attack dataset from an Excel file.

    Args:
        path: Path to the Excel file (.xlsx, .xls) or CSV.

    Returns:
        Status plus row count and column coverage.
    """
    if path.lower().startswith("s3://"):
        df = _load_from_s3(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df, missing = _normalize_columns(df)
    DATA_STORE.df = df
    DATA_STORE.path = path
    DATA_STORE.missing_columns = missing

    return {
        "status": "success",
        "content": [
            {
                "json": {
                    "rows": int(df.shape[0]),
                    "columns": list(df.columns),
                    "missing_required_columns": missing,
                }
            }
        ],
    }


@tool
def dataset_overview() -> dict:
    """
    Summarize the loaded dataset (row count, column list, missing values).
    """
    df = _require_data()
    missing = df.isna().sum().to_dict()
    return {
        "status": "success",
        "content": [
            {
                "json": {
                    "rows": int(df.shape[0]),
                    "columns": list(df.columns),
                    "missing_values": {k: int(v) for k, v in missing.items()},
                }
            }
        ],
    }


@tool
def count_attacks(
    year: int | None = None,
    date: str | None = None,
    hour_of_day: str | None = None,
    survived_or_dead: str | None = None,
    injury: str | None = None,
    state: str | None = None,
    sex: str | None = None,
    age: str | None = None,
    shark_type: str | None = None,
) -> dict:
    """
    Count shark attacks with optional filters.

    Args:
        year: Incident year to match (exact).
        date: Filter by Incident.date (substring match).
        hour_of_day: Filter by Incident.hour.of.day (substring match).
        survived_or_dead: Filter by Victim.survived.or.dead (substring match).
        injury: Filter by Victim.injury (substring match).
        state: Filter by State (substring match).
        sex: Filter by Victim.Sex (substring match).
        age: Filter by Victim.Age (substring match).
        shark_type: Filter by Shark.type (substring match).
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        date=date,
        hour_of_day=hour_of_day,
        survived_or_dead=survived_or_dead,
        injury=injury,
        state=state,
        sex=sex,
        age=age,
        shark_type=shark_type,
    )
    return {
        "status": "success",
        "content": [{"json": {"count": int(filtered.shape[0])}}],
    }


@tool
def top_values(
    column: str,
    n: int = 5,
    year: int | None = None,
    date: str | None = None,
    hour_of_day: str | None = None,
    survived_or_dead: str | None = None,
    injury: str | None = None,
    state: str | None = None,
    sex: str | None = None,
    age: str | None = None,
    shark_type: str | None = None,
) -> dict:
    """
    Return the top N values for a column with optional filters.

    Args:
        column: Column name to rank (e.g., Shark.common.name).
        n: Number of top values to return.
        year: Incident year to filter by.
        date: Filter by Incident.date.
        hour_of_day: Filter by Incident.hour.of.day.
        survived_or_dead: Filter by Victim.survived.or.dead.
        injury: Filter by Victim.injury.
        state: Filter by State.
        sex: Filter by Victim.Sex.
        age: Filter by Victim.Age.
        shark_type: Filter by Shark.type.
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        date=date,
        hour_of_day=hour_of_day,
        survived_or_dead=survived_or_dead,
        injury=injury,
        state=state,
        sex=sex,
        age=age,
        shark_type=shark_type,
    )
    if column not in filtered.columns:
        raise ValueError(f"Column not found: {column}")

    counts = filtered[column].astype(str).value_counts(dropna=False).head(max(1, n))
    rows = [{"value": str(idx), "count": int(count)} for idx, count in counts.items()]
    return {"status": "success", "content": [{"json": {"column": column, "top": rows}}]}


@tool
def fatality_rate_by(
    column: str,
    min_count: int = 5,
    year: int | None = None,
    date: str | None = None,
    hour_of_day: str | None = None,
    state: str | None = None,
    sex: str | None = None,
    age: str | None = None,
    shark_type: str | None = None,
) -> dict:
    """
    Compute fatality rates by a column using Victim.injury text.

    Args:
        column: Column to group by (e.g., Shark.common.name).
        min_count: Minimum total incidents to include a group.
        year: Incident year to filter by.
        date: Filter by Incident.date.
        hour_of_day: Filter by Incident.hour.of.day.
        state: Filter by State.
        sex: Filter by Victim.Sex.
        age: Filter by Victim.Age.
        shark_type: Filter by Shark.type.
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        date=date,
        hour_of_day=hour_of_day,
        state=state,
        sex=sex,
        age=age,
        shark_type=shark_type,
    )
    if column not in filtered.columns:
        raise ValueError(f"Column not found: {column}")

    fatal = _fatal_mask(filtered)
    grouped = filtered.assign(_fatal=fatal).groupby(column, dropna=False)
    stats = grouped["_fatal"].agg(["sum", "count"]).reset_index()
    stats = stats.rename(columns={"sum": "fatal_count", "count": "total_count"})
    stats = stats[stats["total_count"] >= max(1, min_count)]
    stats["fatal_rate"] = (stats["fatal_count"] / stats["total_count"]).round(4)
    stats = stats.sort_values("fatal_rate", ascending=False)

    rows = [
        {
            "value": str(row[column]),
            "fatal_count": int(row["fatal_count"]),
            "total_count": int(row["total_count"]),
            "fatal_rate": float(row["fatal_rate"]),
        }
        for _, row in stats.iterrows()
    ]
    return {
        "status": "success",
        "content": [{"json": {"column": column, "groups": rows}}],
    }


SYSTEM_PROMPT = (
    "You are a data analyst for a shark-attack dataset. "
    "Use the tools to load the Excel file and compute exact counts. "
    "Do not guess. If the dataset is not loaded, ask for the file path. "
    "When asked 'why', interpret it as patterns visible in the data "
    "(e.g., higher counts by year, injury type, or shark type), "
    "and explain those patterns without claiming causation. "
    "Be concise and cite the counts you used."
)


def build_agent() -> Agent:
    return Agent(
        tools=[load_shark_attacks, dataset_overview, count_attacks, top_values, fatality_rate_by],
        system_prompt=SYSTEM_PROMPT,
        name="Shark Attack Analyst",
        description="Answer questions about shark attack records from a loaded spreadsheet.",
    )


def resolve_data_path() -> str | None:
    env_path = os.getenv("SHARK_ATTACKS_DATA")
    if env_path:
        return env_path

    config_override = os.getenv("SHARK_ATTACKS_CONFIG")
    base_dir = os.path.dirname(__file__)
    config_candidates = [
        config_override,
        os.path.join(base_dir, "config.json"),
        os.path.join(base_dir, "..", "config.json"),
    ]
    for candidate in config_candidates:
        if not candidate:
            continue
        config = _load_config(candidate)
        data_path = config.get("data_path")
        if data_path:
            return data_path
    return None


def load_data_if_available(agent: Agent) -> str | None:
    data_path = resolve_data_path()
    if not data_path:
        return None
    agent.tool.load_shark_attacks(path=data_path)
    return data_path

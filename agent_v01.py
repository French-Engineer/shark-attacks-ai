from __future__ import annotations

import argparse
import json
import os
import re
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import boto3
from strands import Agent, tool


CANONICAL_COLUMNS = {
    "incidentyear": "Incident.year",
    "victiminjury": "Victim.injury",
    "state": "State",
    "sharkcommonname": "Shark.common.name",
    "provokedunprovoked": "Provoked/unprovoked",
    "victimactivity": "Victim.activity",
}


def _clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@dataclass
class SharkDataStore:
    df: pd.DataFrame | None = None
    path: str | None = None
    missing_columns: list[str] | None = None


DATA_STORE = SharkDataStore()
CONFIG_PATH = "config.json"


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
    injury: str | None = None,
    state: str | None = None,
    shark: str | None = None,
    provoked: str | None = None,
    activity: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str | int]] = []
    if year is not None:
        filters.append(("Incident.year", year))
    if injury:
        filters.append(("Victim.injury", injury))
    if state:
        filters.append(("State", state))
    if shark:
        filters.append(("Shark.common.name", shark))
    if provoked:
        filters.append(("Provoked/unprovoked", provoked))
    if activity:
        filters.append(("Victim.activity", activity))

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
    if "Victim.injury" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df["Victim.injury"].astype(str)
    return series.str.contains(r"\bfatal\b|dead|death", case=False, na=False)


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
    injury: str | None = None,
    state: str | None = None,
    shark: str | None = None,
    provoked: str | None = None,
    activity: str | None = None,
) -> dict:
    """
    Count shark attacks with optional filters.

    Args:
        year: Incident year to match (exact).
        injury: Filter by Victim.injury (substring match).
        state: Filter by State (substring match).
        shark: Filter by Shark.common.name (substring match).
        provoked: Filter by Provoked/unprovoked (substring match).
        activity: Filter by Victim.activity (substring match).
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        injury=injury,
        state=state,
        shark=shark,
        provoked=provoked,
        activity=activity,
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
    injury: str | None = None,
    state: str | None = None,
    shark: str | None = None,
    provoked: str | None = None,
    activity: str | None = None,
) -> dict:
    """
    Return the top N values for a column with optional filters.

    Args:
        column: Column name to rank (e.g., Shark.common.name).
        n: Number of top values to return.
        year: Incident year to filter by.
        injury: Filter by Victim.injury.
        state: Filter by State.
        shark: Filter by Shark.common.name.
        provoked: Filter by Provoked/unprovoked.
        activity: Filter by Victim.activity.
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        injury=injury,
        state=state,
        shark=shark,
        provoked=provoked,
        activity=activity,
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
    state: str | None = None,
    shark: str | None = None,
    provoked: str | None = None,
    activity: str | None = None,
) -> dict:
    """
    Compute fatality rates by a column using Victim.injury text.

    Args:
        column: Column to group by (e.g., Shark.common.name).
        min_count: Minimum total incidents to include a group.
        year: Incident year to filter by.
        state: Filter by State.
        shark: Filter by Shark.common.name.
        provoked: Filter by Provoked/unprovoked.
        activity: Filter by Victim.activity.
    """
    df = _require_data()
    filtered = _apply_filters(
        df,
        year=year,
        state=state,
        shark=shark,
        provoked=provoked,
        activity=activity,
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
    "(e.g., higher counts by year, injury type, or shark species), "
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Shark attack QA agent (Strands).")
    parser.add_argument("--data", help="Path to the Excel/CSV file.")
    parser.add_argument("--s3", help="S3 URI to the Excel/CSV file, e.g. s3://bucket/key")
    parser.add_argument("--question", help="Single question to ask.")
    args = parser.parse_args()

    agent = build_agent()

    config_path = os.getenv("SHARK_ATTACKS_CONFIG", CONFIG_PATH)
    config = _load_config(config_path)
    config_data_path = config.get("data_path")
    data_path = args.s3 or args.data or os.getenv("SHARK_ATTACKS_DATA") or config_data_path
    if data_path:
        agent.tool.load_shark_attacks(path=data_path)

    if args.question:
        response = agent(args.question)
        print(response)
        return

    print("Hi ! I'm your Shark Attack Analyst. Type a question or 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        response = agent(user_input)
        print(response)


if __name__ == "__main__":
    main()

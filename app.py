from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "ESB Smart Meter Insight"
APP_TAGLINE = "Upload your ESB smart meter CSV files and get a clear electricity usage, cost, and supplier-accuracy dashboard."

DEFAULT_TARIFF = {
    "supplier_name": "",
    "plan_name": "",
    "unit_rate_cent": 30.0,
    "standing_charge_year": 250.0,
    "pso_levy_year": 0.0,
    "export_rate_cent": 0.0,
    "supplier_app_kwh": None,
}

APPLIANCE_EXAMPLES = pd.DataFrame(
    [
        ["Kettle", "0.08-0.12 kWh per boil", "Small individually, noticeable if used often"],
        ["Oven", "1-2 kWh per hour", "Often part of an evening peak"],
        ["Immersion heater", "2-3 kWh per hour", "Large steady block, often timed"],
        ["Electric shower", "1.5-3 kWh for 10-15 min", "Short, sharp, high-power use"],
        ["Tumble dryer", "2-4 kWh per cycle", "Often visible as a clear high-use block"],
        ["Dishwasher", "0.8-1.5 kWh per cycle", "May appear in late evening or overnight data"],
        ["Washing machine", "0.5-1.2 kWh per cycle", "Higher for hot washes"],
        ["Fridge/freezer", "0.5-1.5 kWh per day", "Part of the everyday baseload"],
        ["Router and standby load", "0.2-1.0 kWh per day combined", "Small but continuous"],
        ["EV charger", "7 kWh per hour at 7 kW", "Only relevant if the home has an EV"],
    ],
    columns=["Appliance", "Typical electricity use", "How it may appear"],
)


@dataclass
class ParsedUpload:
    file_name: str
    file_type: str
    rows: int
    read_types: str
    issues: list[str]
    frame: pd.DataFrame


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 3rem; }
    .hero {
        border: 1px solid #dbe5ef;
        background: linear-gradient(135deg, #f7fbff 0%, #effaf4 100%);
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 .25rem 0; font-size: 2.05rem; letter-spacing: 0; }
    .hero p { margin: 0; color: #334155; font-size: 1.02rem; }
    .metric-card {
        border: 1px solid #dbe5ef;
        border-radius: 8px;
        padding: .9rem;
        background: #ffffff;
        min-height: 112px;
    }
    .metric-card .label { color: #64748b; font-size: .82rem; margin-bottom: .2rem; }
    .metric-card .value { color: #0f172a; font-size: 1.45rem; font-weight: 700; }
    .metric-card .note { color: #475569; font-size: .82rem; margin-top: .25rem; }
    .insight-card {
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        background: #f8fafc;
        padding: .85rem 1rem;
        margin-bottom: .75rem;
    }
    .insight-card h4 { margin: 0 0 .35rem 0; }
    .status-ok { color: #166534; font-weight: 700; }
    .status-amber { color: #b45309; font-weight: 700; }
    .status-red { color: #b91c1c; font-weight: 700; }
    .muted { color: #64748b; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def euro(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"€{value:,.2f}"


def kwh(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.1f} kWh"


def duplicate_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return [str(col) for col in df.columns[df.columns.duplicated()].tolist()]


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique column names: peak_kwh, peak_kwh_2, etc."""
    if df.empty:
        return df.copy()
    out = df.copy()
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    existing = set()
    for col in map(str, out.columns):
        count = seen.get(col, 0) + 1
        seen[col] = count
        candidate = col if count == 1 else f"{col}_{count}"
        while candidate in existing:
            count += 1
            seen[col] = count
            candidate = f"{col}_{count}"
        existing.add(candidate)
        new_columns.append(candidate)
    out.columns = new_columns
    return out


def ensure_unique_columns(df: pd.DataFrame, notes: list[str] | None = None, label: str = "dataframe") -> pd.DataFrame:
    duplicates = duplicate_columns(df)
    out = make_unique_columns(df)
    if duplicates and notes is not None:
        notes.append(f"Duplicate columns in {label} were renamed safely: {', '.join(sorted(set(duplicates)))}.")
    return out


def safe_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    return make_unique_columns(df)


def collapse_duplicate_value_columns(df: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
    """For duplicated semantic value columns, sum duplicates into one canonical column."""
    if df.empty:
        return df.copy()
    out = df.copy()
    for target in target_names:
        positions = [i for i, col in enumerate(out.columns) if col == target]
        if len(positions) > 1:
            combined = out.iloc[:, positions].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            out = out.drop(columns=[target])
            out[target] = combined
    return out


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, finding: str, why: str, cause: str, check: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <h4>{title}</h4>
            <p><b>Finding:</b> {finding}</p>
            <p><b>Why it matters:</b> {why}</p>
            <p><b>Possible contributors:</b> {cause}</p>
            <p><b>What to check at home:</b> {check}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def landing_page() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### What it explains")
        st.write("How much electricity you used, when you use most, what it may cost, and whether supplier app figures look believable.")
    with c2:
        st.markdown("### How it works")
        st.write("You manually upload ESB Networks HDF CSV files. The app detects the confusing file types and builds the dashboard.")
    with c3:
        st.markdown("### Privacy")
        st.write("Version 1 does not ask for ESB login details and does not permanently store uploaded files.")


def onboarding_guide() -> None:
    st.markdown("### How to get your ESB data")
    st.markdown(
        """
        1. Go to **ESB Networks My Account**.
        2. Log in, or create an account.
        3. You may need your **MPRN** from your electricity bill.
        4. Open **My energy consumption**.
        5. Go to **Downloads**.
        6. Download the available **HDF CSV** files.
        7. Upload those CSV files here.

        The file names can look confusing. That is normal. The app looks inside each file and detects whether it is half-hour kW, half-hour kWh, daily total, day/night/peak, or export data.
        """
    )


def privacy_note() -> None:
    st.info(
        "Privacy note: this upload-based version reads files in memory for the current dashboard session. "
        "It does not ask for ESB login details and does not intentionally save uploaded files. "
        "If you export or share screenshots, you can remove personal identifiers such as MPRN and meter serial number first."
    )


def find_header_and_delimiter(text: str) -> tuple[int, str]:
    lines = text.splitlines()
    sample = "\n".join(lines[:40])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    expected = {"mprn", "meter serial number", "read value", "read type", "read date and end time"}
    for i, line in enumerate(lines[:120]):
        cells = {cell.strip().lower() for cell in line.split(delimiter)}
        if expected.issubset(cells):
            return i, delimiter
    return 0, delimiter


def read_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    header_row, delimiter = find_header_and_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, skiprows=header_row, dtype=str)
    except Exception as exc:
        return pd.DataFrame(), [f"Could not read CSV: {exc}"]

    df = df.dropna(how="all")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    required = {"read_value", "read_type", "read_date_and_end_time"}
    if not required.issubset(df.columns):
        return pd.DataFrame(), ["This does not look like an ESB HDF file with read value, read type and timestamp columns."]

    if "mprn" in df.columns:
        df = df[~df["mprn"].astype(str).str.lower().eq("mprn")].copy()
    df["read_value"] = pd.to_numeric(df["read_value"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["timestamp"] = pd.to_datetime(df["read_date_and_end_time"], dayfirst=True, errors="coerce")
    df["read_type"] = df["read_type"].astype(str).str.strip()
    bad_values = int(df["read_value"].isna().sum())
    bad_dates = int(df["timestamp"].isna().sum())
    if bad_values:
        issues.append(f"{bad_values} rows had unreadable numeric values and were ignored.")
    if bad_dates:
        issues.append(f"{bad_dates} rows had unreadable dates and were ignored.")
    df = df.dropna(subset=["read_value", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = ensure_unique_columns(df, issues, uploaded_file.name)
    return df, issues


def detect_file_type(df: pd.DataFrame, file_name: str) -> str:
    read_types = " ".join(df.get("read_type", pd.Series(dtype=str)).dropna().astype(str).str.lower().unique())
    name = file_name.lower()
    if "export" in read_types or "export" in name:
        return "export"
    if "interval" in read_types and "kwh" in read_types:
        return "interval_kwh"
    if "interval" in read_types and "kw" in read_types:
        return "interval_kw"
    if any(term in read_types for term in ["night import", "day peak", "day off-peak", "day off peak"]):
        return "daily_dnp"
    if "24 hr active import register" in read_types or ("daily" in name and "dnp" not in name):
        return "daily_register"
    if "kwh" in read_types:
        return "other_kwh"
    if "kw" in read_types:
        return "other_kw"
    return "unknown"


def parse_uploads(uploaded_files: Iterable) -> list[ParsedUpload]:
    parsed: list[ParsedUpload] = []
    for uploaded in uploaded_files:
        df, issues = read_uploaded_csv(uploaded)
        file_type = detect_file_type(df, uploaded.name) if not df.empty else "unreadable"
        read_types = "; ".join(sorted(df["read_type"].dropna().astype(str).unique())) if not df.empty else ""
        parsed.append(ParsedUpload(uploaded.name, file_type, len(df), read_types, issues, df))
    return parsed


def combine_by_type(parsed: list[ParsedUpload], file_type: str) -> pd.DataFrame:
    frames = [item.frame for item in parsed if item.file_type == file_type and not item.frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = make_unique_columns(combined)
    return combined.drop_duplicates(subset=["timestamp", "read_type", "read_value"]).sort_values("timestamp")


def register_differences(register_df: pd.DataFrame) -> pd.DataFrame:
    if register_df.empty:
        return pd.DataFrame()
    df = register_df.sort_values("timestamp").copy()
    df["usage_date"] = df["timestamp"].dt.normalize()
    df["opening_register_kwh"] = df["read_value"]
    df["closing_register_kwh"] = df["read_value"].shift(-1)
    df["usage_kwh"] = df["closing_register_kwh"] - df["opening_register_kwh"]
    return df[df["usage_kwh"].notna()][["usage_date", "opening_register_kwh", "closing_register_kwh", "usage_kwh"]]


def interval_daily_totals(interval: pd.DataFrame) -> pd.DataFrame:
    if interval.empty or "interval_kwh" not in interval.columns:
        return pd.DataFrame()
    return interval.groupby("usage_date", as_index=False).agg(
        usage_kwh=("interval_kwh", "sum"),
        interval_kwh=("interval_kwh", "sum"),
        interval_count=("interval_kwh", "count"),
        max_interval_kwh=("interval_kwh", "max"),
        max_kw=("interval_kw", "max"),
        avg_kw=("interval_kw", "mean"),
    )


def normalise_daily_usage(daily: pd.DataFrame, interval_daily: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Ensure daily has one canonical usage_kwh column before downstream analysis."""
    daily = ensure_unique_columns(daily, notes, "daily usage data")
    interval_daily = ensure_unique_columns(interval_daily, notes, "interval daily totals")
    if daily.empty and not interval_daily.empty:
        notes.append("Daily totals were unavailable, so daily usage was calculated from 30-minute interval kWh totals.")
        return interval_daily.copy()
    if daily.empty:
        return daily

    daily = daily.copy()
    candidates = [
        "usage_kwh",
        "usage_kwh_x",
        "usage_kwh_y",
        "total_dnp_kwh",
        "interval_kwh",
    ]
    for col in ["night_kwh", "day_off_peak_kwh", "peak_kwh"]:
        if col not in daily.columns:
            daily[col] = np.nan
    if {"night_kwh", "day_off_peak_kwh", "peak_kwh"}.issubset(daily.columns):
        daily["total_dnp_kwh"] = daily[["night_kwh", "day_off_peak_kwh", "peak_kwh"]].sum(axis=1, min_count=1)

    canonical = pd.Series(np.nan, index=daily.index, dtype="float64")
    source_used = None
    for col in candidates:
        if col in daily.columns:
            values = pd.to_numeric(daily[col], errors="coerce")
            if values.notna().any():
                canonical = canonical.fillna(values)
                source_used = source_used or col

    if canonical.isna().all() and not interval_daily.empty:
        fallback = daily[["usage_date"]].merge(interval_daily[["usage_date", "usage_kwh"]], on="usage_date", how="left")["usage_kwh"]
        canonical = pd.to_numeric(fallback, errors="coerce")
        source_used = "interval totals"

    daily["usage_kwh"] = canonical
    for col in ["usage_kwh_x", "usage_kwh_y"]:
        if col in daily.columns:
            daily = daily.drop(columns=col)

    if source_used == "interval totals":
        notes.append("Some daily usage values were missing and were filled from 30-minute interval totals.")
    elif source_used in {"usage_kwh_y", "total_dnp_kwh"}:
        notes.append("Daily totals were derived from day/night/peak register differences where needed.")
    elif source_used == "interval_kwh":
        notes.append("Daily register usage was unavailable for some rows, so interval totals were used where needed.")

    return daily


def tariff_bucket(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute / 60
    if h >= 23 or h < 8:
        return "Night"
    if 17 <= h < 19:
        return "Peak"
    return "Day off-peak"


def build_dataset(parsed: list[ParsedUpload]) -> dict[str, pd.DataFrame | str | float]:
    validation_messages: list[str] = []
    interval_kwh_raw = combine_by_type(parsed, "interval_kwh")
    interval_kw_raw = combine_by_type(parsed, "interval_kw")
    daily_register_raw = combine_by_type(parsed, "daily_register")
    daily_dnp_raw = combine_by_type(parsed, "daily_dnp")
    export_raw = combine_by_type(parsed, "export")

    interval = pd.DataFrame()
    if not interval_kwh_raw.empty:
        interval = interval_kwh_raw.rename(columns={"read_value": "interval_kwh"})[
            ["timestamp", "read_type", "interval_kwh"]
        ].copy()
    if not interval_kw_raw.empty:
        kw = interval_kw_raw.rename(columns={"read_value": "interval_kw"})[["timestamp", "interval_kw"]].copy()
        interval = kw if interval.empty else interval.merge(kw, on="timestamp", how="outer")
    if interval.empty and not daily_register_raw.empty:
        interval = pd.DataFrame()
    elif not interval.empty:
        if "interval_kwh" not in interval and "interval_kw" in interval:
            interval["interval_kwh"] = interval["interval_kw"] * 0.5
        if "interval_kw" not in interval and "interval_kwh" in interval:
            interval["interval_kw"] = interval["interval_kwh"] * 2
        interval["estimated_kwh_from_kw"] = interval["interval_kw"] * 0.5
        interval["kw_kwh_difference"] = interval["estimated_kwh_from_kw"] - interval["interval_kwh"]
        interval["interval_start"] = interval["timestamp"] - pd.Timedelta(minutes=30)
        interval["usage_date"] = interval["interval_start"].dt.normalize()
        interval["hour"] = interval["interval_start"].dt.hour
        interval["weekday"] = interval["interval_start"].dt.day_name()
        interval["is_weekend"] = interval["interval_start"].dt.weekday >= 5
        interval["tariff_bucket_estimate"] = interval["interval_start"].map(tariff_bucket)
    interval = ensure_unique_columns(interval, validation_messages, "interval data")

    interval_daily = interval_daily_totals(interval)
    daily = register_differences(daily_register_raw)
    if daily.empty and not daily_register_raw.empty:
        validation_messages.append("A daily file was uploaded, but daily usage could not be derived from it. The app will use interval totals if available.")
    if not interval_daily.empty:
        if daily.empty:
            daily = interval_daily.copy()
            validation_messages.append("Daily register totals were unavailable, so usage was calculated from 30-minute interval data.")
        else:
            interval_cols = ["usage_date", "interval_kwh", "interval_count", "max_interval_kwh", "max_kw", "avg_kw"]
            daily = daily.merge(interval_daily[interval_cols], on="usage_date", how="outer", suffixes=("", "_interval"))
            daily = ensure_unique_columns(daily, validation_messages, "daily plus interval merge")
    if not daily_dnp_raw.empty:
        dnp = daily_dnp_raw.pivot_table(index="timestamp", columns="read_type", values="read_value", aggfunc="first").sort_index()
        dnp_daily = (dnp.shift(-1) - dnp).dropna(how="all").reset_index().rename(columns={"timestamp": "usage_date"})
        dnp_daily["usage_date"] = pd.to_datetime(dnp_daily["usage_date"]).dt.normalize()
        rename = {}
        for col in dnp_daily.columns:
            lower = str(col).lower()
            if "night" in lower:
                rename[col] = "night_kwh"
            elif "off" in lower:
                rename[col] = "day_off_peak_kwh"
            elif "peak" in lower:
                rename[col] = "peak_kwh"
        dnp_daily = dnp_daily.rename(columns=rename)
        dnp_daily = collapse_duplicate_value_columns(dnp_daily, ["night_kwh", "day_off_peak_kwh", "peak_kwh"])
        dnp_daily = ensure_unique_columns(dnp_daily, validation_messages, "day/night/peak data")
        dnp_value_cols = [c for c in ["night_kwh", "day_off_peak_kwh", "peak_kwh"] if c in dnp_daily.columns]
        if dnp_value_cols:
            dnp_daily["total_dnp_kwh"] = dnp_daily[dnp_value_cols].sum(axis=1, min_count=1)
        if not daily.empty:
            overlapping = [c for c in dnp_daily.columns if c != "usage_date" and c in daily.columns]
            if overlapping:
                validation_messages.append(f"Overlapping day/night/peak columns were refreshed from the DNP file: {', '.join(overlapping)}.")
                daily = daily.drop(columns=overlapping)
            daily = daily.merge(dnp_daily, on="usage_date", how="left", suffixes=("", "_dnp"))
        else:
            daily = dnp_daily
        daily = ensure_unique_columns(daily, validation_messages, "daily plus DNP merge")

    export_daily = pd.DataFrame(columns=["usage_date", "export_kwh"])
    export_kwh = 0.0
    if not export_raw.empty:
        read_type_text = " ".join(export_raw["read_type"].str.lower().unique())
        if "interval" in read_type_text:
            export_tmp = export_raw.copy()
            export_tmp["usage_date"] = (export_tmp["timestamp"] - pd.Timedelta(minutes=30)).dt.normalize()
            export_daily = export_tmp.groupby("usage_date", as_index=False).agg(export_kwh=("read_value", "sum"))
            export_kwh = float(export_daily["export_kwh"].sum())
        else:
            export_register = register_differences(export_raw)
            if not export_register.empty:
                export_daily = export_register.rename(columns={"usage_kwh": "export_kwh"})[["usage_date", "export_kwh"]]
                export_kwh = float(export_daily["export_kwh"].clip(lower=0).sum())

    if daily.empty:
        return {"error": "No usable import usage data was found. Upload at least one interval kWh/kW file or daily total kWh file."}

    daily = normalise_daily_usage(daily, interval_daily, validation_messages)
    if daily.empty or "usage_kwh" not in daily.columns:
        return {
            "error": "The uploaded files were recognised, but the app could not create daily usage totals. Try adding a 30-minute kWh/kW file or daily total kWh file.",
            "messages": validation_messages,
        }
    missing_usage = int(pd.to_numeric(daily["usage_kwh"], errors="coerce").isna().sum())
    if missing_usage:
        validation_messages.append(f"{missing_usage} daily rows had no usable kWh value and were excluded from charts.")
    daily["usage_kwh"] = pd.to_numeric(daily["usage_kwh"], errors="coerce")
    daily = daily.sort_values("usage_date")
    daily = daily[daily["usage_kwh"].notna()].copy()
    if daily.empty:
        return {
            "error": "Daily usage totals could not be calculated from the uploaded files after cleaning.",
            "messages": validation_messages,
        }
    daily = daily[daily["usage_kwh"] >= 0].copy()
    if not export_daily.empty:
        if "export_kwh" in daily.columns:
            daily = daily.drop(columns=["export_kwh"])
        daily = daily.merge(export_daily, on="usage_date", how="left", suffixes=("", "_export"))
        daily = ensure_unique_columns(daily, validation_messages, "daily plus export merge")
    if "export_kwh" not in daily.columns:
        daily["export_kwh"] = 0.0
    daily["export_kwh"] = daily["export_kwh"].fillna(0).clip(lower=0)
    daily["weekday"] = daily["usage_date"].dt.day_name()
    daily["is_weekend"] = daily["usage_date"].dt.weekday >= 5
    daily["rolling_7_day_avg_kwh"] = daily["usage_kwh"].rolling(7, min_periods=3).mean()
    if "interval_count" in daily:
        daily["complete_48_intervals"] = daily["interval_count"].eq(48)

    monthly = daily.assign(month=daily["usage_date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(
        total_kwh=("usage_kwh", "sum"),
        export_kwh=("export_kwh", "sum"),
        days=("usage_kwh", "count"),
        average_daily_kwh=("usage_kwh", "mean"),
        peak_day_kwh=("usage_kwh", "max"),
    )
    monthly = ensure_unique_columns(monthly, validation_messages, "monthly summary")
    weekly = daily.assign(week_start=daily["usage_date"].dt.to_period("W-MON").apply(lambda p: p.start_time)).groupby(
        "week_start", as_index=False
    ).agg(total_kwh=("usage_kwh", "sum"), days=("usage_kwh", "count"))
    weekly = ensure_unique_columns(weekly, validation_messages, "weekly summary")

    missing = pd.DataFrame()
    quality_rows = []
    if not interval.empty:
        expected = pd.date_range(interval["timestamp"].min(), interval["timestamp"].max(), freq="30min")
        missing = pd.DataFrame({"missing_timestamp": expected.difference(pd.DatetimeIndex(interval["timestamp"].dropna()))})
        completeness = len(interval.dropna(subset=["interval_kwh"])) / len(expected) if len(expected) else np.nan
        duplicate_count = int(interval["timestamp"].duplicated().sum())
        impossible_count = int(((interval["interval_kwh"] < 0) | (interval["interval_kwh"] > 20) | (interval["interval_kw"] < 0) | (interval["interval_kw"] > 40)).sum())
        recon = float(interval["kw_kwh_difference"].abs().max()) if "kw_kwh_difference" in interval else np.nan
        quality_rows.extend(
            [
                ["Interval completeness", f"{completeness:.2%}", "OK" if completeness >= 0.99 else "Check"],
                ["Missing 30-minute intervals", str(len(missing)), "OK" if missing.empty else "Check"],
                ["Duplicate timestamps", str(duplicate_count), "OK" if duplicate_count == 0 else "Check"],
                ["Impossible interval values", str(impossible_count), "OK" if impossible_count == 0 else "Problem"],
                ["kW to kWh reconciliation", f"{recon:.6f} kWh", "OK" if pd.isna(recon) or recon < 0.001 else "Check"],
            ]
        )
        flat_runs = count_flat_runs(interval)
        quality_rows.append(["Long repeated constant-value runs", str(flat_runs), "OK" if flat_runs == 0 else "Check"])
    else:
        quality_rows.append(["Interval data", "Not uploaded", "Check"])
    if not daily.empty:
        quality_rows.append(["Daily usage days", str(len(daily)), "OK"])
    for message in validation_messages:
        quality_rows.append(["Validation note", message, "Check"])

    return {
        "interval": make_unique_columns(interval),
        "daily": make_unique_columns(daily),
        "weekly": make_unique_columns(weekly),
        "monthly": make_unique_columns(monthly),
        "quality": pd.DataFrame(quality_rows, columns=["check", "value", "status"]),
        "missing": missing,
        "export_kwh": export_kwh,
        "messages": validation_messages,
        "source": "upload",
    }


def count_flat_runs(interval: pd.DataFrame) -> int:
    values = interval.dropna(subset=["interval_kwh"]).sort_values("timestamp").copy()
    if values.empty:
        return 0
    values["run"] = (values["interval_kwh"] != values["interval_kwh"].shift()).cumsum()
    return int(sum(len(group) >= 8 for _, group in values.groupby("run")))


def demo_dataset() -> dict[str, pd.DataFrame | str | float]:
    rng = pd.date_range("2026-02-01 00:30", periods=48 * 60, freq="30min")
    hour = (rng - pd.Timedelta(minutes=30)).hour
    weekday = (rng - pd.Timedelta(minutes=30)).weekday
    base = 0.08 + np.where((hour >= 18) & (hour <= 21), 0.28, 0) + np.where((hour >= 6) & (hour <= 8), 0.12, 0)
    weekend = np.where(weekday >= 5, 0.05, 0)
    cycle = 0.05 * (np.sin(np.arange(len(rng)) / 19) + 1)
    spikes = np.zeros(len(rng))
    spikes[::317] = 1.1
    interval_kwh = np.round(base + weekend + cycle + spikes, 3)
    interval = pd.DataFrame({"timestamp": rng, "interval_kwh": interval_kwh})
    interval["interval_kw"] = interval["interval_kwh"] * 2
    interval["estimated_kwh_from_kw"] = interval["interval_kw"] * 0.5
    interval["kw_kwh_difference"] = 0.0
    interval["interval_start"] = interval["timestamp"] - pd.Timedelta(minutes=30)
    interval["usage_date"] = interval["interval_start"].dt.normalize()
    interval["hour"] = interval["interval_start"].dt.hour
    interval["weekday"] = interval["interval_start"].dt.day_name()
    interval["is_weekend"] = interval["interval_start"].dt.weekday >= 5
    interval["tariff_bucket_estimate"] = interval["interval_start"].map(tariff_bucket)
    daily = interval.groupby("usage_date", as_index=False).agg(
        usage_kwh=("interval_kwh", "sum"),
        interval_kwh=("interval_kwh", "sum"),
        interval_count=("interval_kwh", "count"),
        max_interval_kwh=("interval_kwh", "max"),
        max_kw=("interval_kw", "max"),
        avg_kw=("interval_kw", "mean"),
    )
    daily = daily[daily["interval_count"].eq(48)].copy()
    night = interval[interval["tariff_bucket_estimate"].eq("Night")].groupby("usage_date")["interval_kwh"].sum()
    peak = interval[interval["tariff_bucket_estimate"].eq("Peak")].groupby("usage_date")["interval_kwh"].sum()
    day = interval[interval["tariff_bucket_estimate"].eq("Day off-peak")].groupby("usage_date")["interval_kwh"].sum()
    daily["night_kwh"] = daily["usage_date"].map(night).fillna(0)
    daily["peak_kwh"] = daily["usage_date"].map(peak).fillna(0)
    daily["day_off_peak_kwh"] = daily["usage_date"].map(day).fillna(0)
    daily["weekday"] = daily["usage_date"].dt.day_name()
    daily["is_weekend"] = daily["usage_date"].dt.weekday >= 5
    daily["rolling_7_day_avg_kwh"] = daily["usage_kwh"].rolling(7, min_periods=3).mean()
    daily["export_kwh"] = 0.0
    monthly = daily.assign(month=daily["usage_date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(
        total_kwh=("usage_kwh", "sum"),
        export_kwh=("export_kwh", "sum"),
        days=("usage_kwh", "count"),
        average_daily_kwh=("usage_kwh", "mean"),
        peak_day_kwh=("usage_kwh", "max"),
    )
    weekly = daily.assign(week_start=daily["usage_date"].dt.to_period("W-MON").apply(lambda p: p.start_time)).groupby(
        "week_start", as_index=False
    ).agg(total_kwh=("usage_kwh", "sum"), days=("usage_kwh", "count"))
    quality = pd.DataFrame(
        [
            ["Interval completeness", "100.00%", "OK"],
            ["Missing 30-minute intervals", "0", "OK"],
            ["Duplicate timestamps", "0", "OK"],
            ["Impossible interval values", "0", "OK"],
            ["kW to kWh reconciliation", "0.000000 kWh", "OK"],
            ["Long repeated constant-value runs", "0", "OK"],
        ],
        columns=["check", "value", "status"],
    )
    return {
        "interval": interval,
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "quality": quality,
        "missing": pd.DataFrame(columns=["missing_timestamp"]),
        "export_kwh": 0.0,
        "source": "demo",
    }


def cost_for_period(kwh_used: float, days: int, tariff: dict, export_kwh: float = 0.0) -> dict[str, float]:
    usage = kwh_used * tariff["unit_rate_cent"] / 100
    standing = tariff["standing_charge_year"] * days / 365
    pso = tariff["pso_levy_year"] * days / 365
    export_credit = export_kwh * tariff["export_rate_cent"] / 100
    total = usage + standing + pso - export_credit
    annualised_kwh = kwh_used / max(days, 1) * 365
    annualised_cost = annualised_kwh * tariff["unit_rate_cent"] / 100 + tariff["standing_charge_year"] + tariff["pso_levy_year"]
    return {
        "usage_charge": usage,
        "standing_charge": standing,
        "pso_levy": pso,
        "export_credit": export_credit,
        "total": total,
        "annualised_kwh": annualised_kwh,
        "annualised_cost": annualised_cost,
    }


def add_monthly_costs(monthly: pd.DataFrame, tariff: dict) -> pd.DataFrame:
    out = make_unique_columns(monthly)
    if "export_kwh" not in out.columns:
        out["export_kwh"] = 0.0
    out["usage_charge_eur"] = out["total_kwh"] * tariff["unit_rate_cent"] / 100
    out["standing_charge_eur"] = out["days"] * tariff["standing_charge_year"] / 365
    out["pso_levy_eur"] = out["days"] * tariff["pso_levy_year"] / 365
    out["export_credit_eur"] = out["export_kwh"] * tariff["export_rate_cent"] / 100
    out["total_estimated_cost_eur"] = out["usage_charge_eur"] + out["standing_charge_eur"] + out["pso_levy_eur"] - out["export_credit_eur"]
    return out


def status_class(status: str) -> str:
    status = str(status).lower()
    if "ok" in status:
        return "status-ok"
    if "problem" in status or "fail" in status:
        return "status-red"
    return "status-amber"


def month_label(period: str) -> str:
    try:
        return pd.Period(period, freq="M").strftime("%b %Y")
    except Exception:
        return str(period)


def supplier_message(dataset: dict, tariff: dict, selected_kwh: float, selected_cost: float, start: date, end: date) -> str:
    app_kwh = tariff.get("supplier_app_kwh")
    discrepancy = ""
    if app_kwh is not None:
        ratio = selected_kwh / max(float(app_kwh), 0.01)
        discrepancy = f"\n- Supplier app reported usage for the comparable period: {float(app_kwh):.1f} kWh\n- Difference versus ESB-derived usage: about {ratio:.1f}x\n"
    return f"""Hello,

I have downloaded my ESB Networks HDF smart meter CSV files and compared them with the usage shown in my supplier app.

For the period {start:%d %b %Y} to {end:%d %b %Y}, the ESB-derived import usage is:

- ESB-derived usage: {selected_kwh:.1f} kWh
- Estimated cost using my tariff inputs: {euro(selected_cost)}{discrepancy}
- The uploaded ESB interval/daily data appears internally consistent based on the dashboard quality checks.

Please refresh or re-sync the smart meter usage data for my account and confirm that the correct MPRN is mapped to my supplier account and app profile.

Please also confirm whether billing will be based on the ESB Networks meter data rather than the app display.

Thank you."""


def render_file_detection(parsed: list[ParsedUpload]) -> None:
    st.markdown("### Uploaded files detected")
    if not parsed:
        st.write("No files uploaded yet.")
        return
    rows = [
        {
            "File": item.file_name,
            "Detected type": item.file_type.replace("_", " "),
            "Rows": item.rows,
            "Read types found": item.read_types,
            "Notes": " ".join(item.issues),
        }
        for item in parsed
    ]
    st.dataframe(make_unique_columns(pd.DataFrame(rows)), hide_index=True, width="stretch")


def render_dashboard(dataset: dict, tariff: dict, parsed: list[ParsedUpload] | None = None) -> None:
    daily = make_unique_columns(dataset["daily"])
    interval = make_unique_columns(dataset["interval"])
    monthly = add_monthly_costs(dataset["monthly"].copy(), tariff)
    monthly = make_unique_columns(monthly)
    quality = make_unique_columns(dataset["quality"])
    missing = make_unique_columns(dataset["missing"])
    export_kwh = float(dataset.get("export_kwh", 0.0) or 0.0)
    messages = [str(m) for m in dataset.get("messages", []) if str(m).strip()]

    min_day = daily["usage_date"].min().date()
    max_day = daily["usage_date"].max().date()
    default_start = min_day
    default_end = max_day

    total_kwh = float(daily["usage_kwh"].sum())
    avg_daily = float(daily["usage_kwh"].mean())
    peak_day = daily.loc[daily["usage_kwh"].idxmax()]
    quiet_day = daily.loc[daily[daily["usage_kwh"] > 0]["usage_kwh"].idxmin()]
    whole_period_cost = cost_for_period(total_kwh, len(daily), tariff, export_kwh)
    confidence = "High" if "Problem" not in set(quality["status"]) else "Needs checking"

    st.markdown("## Your dashboard")
    if dataset.get("source") == "demo":
        st.info("Demo mode is using synthetic sample data. Upload your own ESB HDF files for real results.")
    for message in messages:
        st.warning(message)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Usage in uploaded period", kwh(total_kwh), f"{min_day:%d %b %Y} to {max_day:%d %b %Y}")
    with c2:
        metric_card("Estimated cost", euro(whole_period_cost["total"]), "Usage + standing + PSO - export")
    with c3:
        metric_card("Average daily use", kwh(avg_daily), "Useful baseline")
    with c4:
        metric_card("Data confidence", confidence, "Based on quality checks")

    if tariff.get("supplier_app_kwh") is not None:
        app_kwh = float(tariff["supplier_app_kwh"])
        ratio = total_kwh / max(app_kwh, 0.01)
        if ratio > 1.2 or ratio < 0.8:
            st.warning(f"Supplier app check: uploaded ESB data shows {kwh(total_kwh)}, while the app figure entered is {kwh(app_kwh)}. That is about {ratio:.1f}x different.")
        else:
            st.success("Supplier app check: the app figure entered is broadly close to the uploaded ESB-derived usage.")

    tab_summary, tab_patterns, tab_cost, tab_quality, tab_support = st.tabs(
        ["Summary", "Patterns", "Cost", "Data quality", "Supplier support"]
    )

    with tab_summary:
        st.markdown("### Key metrics")
        row = st.columns(4)
        with row[0]:
            metric_card("Highest day", kwh(peak_day["usage_kwh"]), pd.Timestamp(peak_day["usage_date"]).strftime("%d %b %Y"))
        with row[1]:
            metric_card("Quietest day", kwh(quiet_day["usage_kwh"]), pd.Timestamp(quiet_day["usage_date"]).strftime("%d %b %Y"))
        with row[2]:
            metric_card("Annualised usage", kwh(whole_period_cost["annualised_kwh"]), "Based on uploaded period")
        with row[3]:
            metric_card("Annualised cost", euro(whole_period_cost["annualised_cost"]), "At your tariff inputs")

        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("#### Daily usage")
            st.caption("Spikes are the days worth investigating first.")
            fig = px.line(safe_plot_df(daily), x="usage_date", y="usage_kwh", markers=True, labels={"usage_date": "Date", "usage_kwh": "kWh"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Monthly totals")
            chart_monthly = safe_plot_df(monthly)
            fig = px.bar(chart_monthly, x="month", y="total_kwh", text=chart_monthly["total_kwh"].round(1), labels={"month": "Month", "total_kwh": "kWh"})
            fig.update_xaxes(ticktext=[month_label(m) for m in chart_monthly["month"]], tickvals=chart_monthly["month"])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Estimated daily cost")
        daily_cost = daily.copy()
        daily_cost["estimated_cost_eur"] = daily_cost["usage_kwh"] * tariff["unit_rate_cent"] / 100 + tariff["standing_charge_year"] / 365 + tariff["pso_levy_year"] / 365
        fig = px.bar(safe_plot_df(daily_cost), x="usage_date", y="estimated_cost_eur", labels={"usage_date": "Date", "estimated_cost_eur": "Estimated cost"})
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_patterns:
        st.markdown("### Where is my electricity going?")
        st.write("The meter shows when electricity was used. It cannot identify exact appliances, so these are clues rather than proof.")

        if not interval.empty:
            hourly = interval.dropna(subset=["interval_kwh"]).groupby("hour", as_index=False).agg(avg_kwh=("interval_kwh", "mean"))
            hourly = make_unique_columns(hourly)
            hourly["avg_kw"] = hourly["avg_kwh"] * 2
            peak_hours = hourly.nlargest(3, "avg_kw")
            baseload_kw = float(hourly.loc[hourly["hour"].between(2, 5), "avg_kw"].median()) if not hourly.empty else np.nan
            insight_card(
                "Evening or routine peak",
                f"Your strongest average hours are around {', '.join(str(int(h)) + ':00' for h in peak_hours['hour'])}.",
                "Peaks are where behavioural changes can have the biggest impact.",
                "Cooking, immersion, electric shower, laundry, dishwasher, heating controls, or general evening activity.",
                "Compare two high-use days with your household routine between the peak hours.",
            )
            insight_card(
                "Overnight baseload",
                f"Your rough overnight baseload is about {baseload_kw:.2f} kW.",
                "A continuous load runs every day, so small watts become meaningful monthly cost.",
                "Fridge/freezer, router, standby devices, pumps, chargers, dehumidifier, or timed water heating.",
                "Check timers and anything left running, warm, humming, charging, or cycling overnight.",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Hourly average usage")
                fig = px.line(safe_plot_df(hourly), x="hour", y="avg_kw", markers=True, labels={"hour": "Hour starting", "avg_kw": "Average kW"})
                fig.update_xaxes(dtick=2)
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### Weekday vs weekend")
                weekday = daily.groupby("is_weekend", as_index=False).agg(avg_kwh=("usage_kwh", "mean"))
                weekday["period"] = weekday["is_weekend"].map({False: "Weekday", True: "Weekend"})
                weekday = make_unique_columns(weekday)
                fig = px.bar(safe_plot_df(weekday), x="period", y="avg_kwh", text=weekday["avg_kwh"].round(1), labels={"period": "", "avg_kwh": "Average kWh/day"})
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Day/hour heatmap")
            heat = safe_plot_df(interval.dropna(subset=["interval_kwh"]).copy())
            heat["date_label"] = pd.to_datetime(heat["usage_date"]).dt.strftime("%d %b")
            heat_table = heat.pivot_table(index="date_label", columns="hour", values="interval_kwh", aggfunc="sum")
            fig = go.Figure(data=go.Heatmap(z=heat_table.values, x=heat_table.columns, y=heat_table.index, colorscale="YlOrRd", colorbar=dict(title="kWh")))
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Hour starting", yaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a 30-minute kWh or kW file to unlock hourly patterns and heatmaps.")

        dnp_cols = [c for c in ["night_kwh", "day_off_peak_kwh", "peak_kwh"] if c in daily.columns]
        if dnp_cols:
            st.markdown("#### Day / night / peak split")
            split = pd.DataFrame(
                {
                    "period": ["Night", "Day off-peak", "Peak"],
                    "kwh": [daily.get("night_kwh", pd.Series(dtype=float)).sum(), daily.get("day_off_peak_kwh", pd.Series(dtype=float)).sum(), daily.get("peak_kwh", pd.Series(dtype=float)).sum()],
                }
            )
            fig = px.pie(safe_plot_df(split), names="period", values="kwh", hole=0.45)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload the daily day/night/peak file to see tariff-period behaviour.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Highest usage days")
            top_days = daily.nlargest(10, "usage_kwh").sort_values("usage_kwh")
            top_days = safe_plot_df(top_days)
            fig = px.bar(top_days, x="usage_kwh", y=top_days["usage_date"].dt.strftime("%d %b"), orientation="h", labels={"usage_kwh": "kWh", "y": "Date"})
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Rolling 7-day average")
            fig = go.Figure()
            chart_daily = safe_plot_df(daily)
            fig.add_trace(go.Scatter(x=chart_daily["usage_date"], y=chart_daily["usage_kwh"], mode="lines+markers", name="Daily kWh", opacity=0.45))
            fig.add_trace(go.Scatter(x=chart_daily["usage_date"], y=chart_daily["rolling_7_day_avg_kwh"], mode="lines", name="7-day average", line=dict(width=4)))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="kWh/day")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Appliance interpretation helper")
        st.write("Possible contributors include the appliances below. Treat this as a sense-check, not a diagnosis.")
        st.dataframe(make_unique_columns(APPLIANCE_EXAMPLES), hide_index=True, width="stretch")

    with tab_cost:
        st.markdown("### Cost simulator")
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Billing period start", value=default_start, min_value=min_day, max_value=max_day)
        with c2:
            end = st.date_input("Billing period end", value=default_end, min_value=min_day, max_value=max_day)

        if start > end:
            st.warning("Start date must be before end date.")
        else:
            selected = daily[(daily["usage_date"].dt.date >= start) & (daily["usage_date"].dt.date <= end)]
            selected_kwh = float(selected["usage_kwh"].sum())
            selected_export = float(selected.get("export_kwh", pd.Series(dtype=float)).sum())
            export_for_period = st.number_input("Export kWh in this period, if known", min_value=0.0, value=selected_export, step=1.0)
            days = (end - start).days + 1
            selected_cost = cost_for_period(selected_kwh, days, tariff, export_for_period)
            cols = st.columns(5)
            with cols[0]:
                metric_card("Selected usage", kwh(selected_kwh), f"{days} days")
            with cols[1]:
                metric_card("Estimated cost", euro(selected_cost["total"]), "Usage + standing + PSO - export")
            with cols[2]:
                metric_card("Usage charge", euro(selected_cost["usage_charge"]), f"{tariff['unit_rate_cent']:.2f}c/kWh")
            with cols[3]:
                metric_card("Fixed charges", euro(selected_cost["standing_charge"] + selected_cost["pso_levy"]), "Standing + PSO")
            with cols[4]:
                metric_card("Annualised cost", euro(selected_cost["annualised_cost"]), "Based on selected period")

            breakdown = pd.DataFrame(
                [
                    ["Usage charge", selected_cost["usage_charge"]],
                    ["Standing charge", selected_cost["standing_charge"]],
                    ["PSO levy", selected_cost["pso_levy"]],
                    ["Export credit", -selected_cost["export_credit"]],
                ],
                columns=["Component", "EUR"],
            )
            fig = px.bar(safe_plot_df(breakdown), x="Component", y="EUR", text=breakdown["EUR"].map(lambda x: f"€{x:.2f}"))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="€")
            st.plotly_chart(fig, use_container_width=True)

            if tariff.get("supplier_app_kwh") is not None:
                app_kwh = float(tariff["supplier_app_kwh"])
                app_cost = cost_for_period(app_kwh, days, tariff, export_for_period)
                comparison = pd.DataFrame(
                    [
                        ["ESB-derived usage", selected_kwh, selected_cost["total"]],
                        ["Supplier app reported usage", app_kwh, app_cost["total"]],
                    ],
                    columns=["Scenario", "kWh", "Estimated total cost"],
                )
                fig = px.bar(safe_plot_df(comparison), x="Scenario", y="kWh", text=comparison["kWh"].round(1))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Monthly estimated costs")
        st.dataframe(
            make_unique_columns(monthly[["month", "total_kwh", "days", "usage_charge_eur", "standing_charge_eur", "pso_levy_eur", "total_estimated_cost_eur"]]).style.format(
                {
                    "total_kwh": "{:.1f}",
                    "usage_charge_eur": "€{:.2f}",
                    "standing_charge_eur": "€{:.2f}",
                    "pso_levy_eur": "€{:.2f}",
                    "total_estimated_cost_eur": "€{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    with tab_quality:
        st.markdown("### Data quality checks")
        with st.expander("Uploaded data diagnostics"):
            if parsed is not None:
                render_file_detection(parsed)
            duplicate_rows = []
            for label, frame in [
                ("daily", daily),
                ("interval", interval),
                ("monthly", monthly),
                ("quality", quality),
                ("missing intervals", missing),
            ]:
                duplicate_rows.append(
                    {
                        "Dataframe": label,
                        "Columns": ", ".join(map(str, frame.columns)),
                        "Duplicate columns currently present": ", ".join(duplicate_columns(frame)) or "None",
                    }
                )
            st.dataframe(make_unique_columns(pd.DataFrame(duplicate_rows)), hide_index=True, width="stretch")
        for _, row in quality.iterrows():
            st.markdown(
                f"<p><span class='{status_class(row['status'])}'>{row['status']}</span> — <b>{row['check']}</b>: {row['value']}</p>",
                unsafe_allow_html=True,
            )
        if not missing.empty:
            st.warning(f"Missing interval range: {missing['missing_timestamp'].min()} to {missing['missing_timestamp'].max()}.")
        st.markdown("#### What this means")
        if "Problem" in set(quality["status"]):
            st.error("There are data issues worth checking before relying on exact totals.")
        else:
            st.success("The uploaded meter data looks usable for homeowner-level insight.")
        st.write("A supplier app can still be stale or partially synced even when the ESB export itself looks consistent.")
        with st.expander("Assumptions"):
            st.markdown(
                """
                - ESB timestamps are treated as local Irish meter time.
                - Interval readings are treated as 30-minute values ending at the timestamp shown.
                - Daily register files are cumulative, so daily usage is calculated by subtracting one day from the next.
                - If both daily and interval files are present, daily register totals are preferred for daily/monthly totals.
                - Export credit is only applied when export data is uploaded or entered in the cost simulator.
                """
            )

    with tab_support:
        st.markdown("### Supplier support message")
        default_start = min_day
        default_end = max_day
        selected_kwh = total_kwh
        selected_cost = whole_period_cost["total"]
        message = supplier_message(dataset, tariff, selected_kwh, selected_cost, default_start, default_end)
        st.text_area("Copy and paste this into a supplier support request", value=message, height=300)

        st.markdown("### Recommended next actions")
        recommendations = pd.DataFrame(
            [
                ["1", "Compare the next supplier bill with this dashboard", "Bills can include standing charges, PSO, VAT, credits and adjustments."],
                ["2", "If the supplier app disagrees, ask for a smart meter data refresh", "The ESB export is the stronger evidence source."],
                ["3", "Check high-use days first", "One or two routines often explain most spikes."],
                ["4", "Investigate evening peaks", "Cooking, hot water, drying and showering can cluster together."],
                ["5", "Investigate overnight baseload", "Continuous loads quietly add up over a month."],
                ["6", "Download another ESB export next month", "A second month confirms whether patterns are persistent."],
            ],
            columns=["Rank", "Action", "Why it helps"],
        )
        st.dataframe(make_unique_columns(recommendations), hide_index=True, width="stretch")


def render_analytics_dashboard(dataset: dict, tariff: dict, parsed: list[ParsedUpload] | None = None) -> None:
    """Render the upload-based app as a richer, multi-tab homeowner dashboard."""
    daily = make_unique_columns(dataset["daily"].copy())
    interval = make_unique_columns(dataset["interval"].copy())
    monthly = make_unique_columns(add_monthly_costs(dataset["monthly"].copy(), tariff))
    quality = make_unique_columns(dataset["quality"].copy())
    missing = make_unique_columns(dataset["missing"].copy())
    export_kwh = float(dataset.get("export_kwh", 0.0) or 0.0)
    messages = [str(m) for m in dataset.get("messages", []) if str(m).strip()]

    min_day = daily["usage_date"].min().date()
    max_day = daily["usage_date"].max().date()
    period_days = max((max_day - min_day).days + 1, 1)
    total_kwh = float(daily["usage_kwh"].sum())
    avg_daily = float(daily["usage_kwh"].mean())
    median_daily = float(daily["usage_kwh"].median())
    peak_day = daily.loc[daily["usage_kwh"].idxmax()]
    non_zero_days = daily[daily["usage_kwh"] > 0]
    quiet_day = non_zero_days.loc[non_zero_days["usage_kwh"].idxmin()] if not non_zero_days.empty else daily.loc[daily["usage_kwh"].idxmin()]
    total_cost = cost_for_period(total_kwh, period_days, tariff, export_kwh)

    problem_count = int((quality["status"] == "Problem").sum()) if "status" in quality.columns else 0
    check_count = int((quality["status"] == "Check").sum()) if "status" in quality.columns else 0
    missing_penalty = min(20, int(len(missing) / max(period_days, 1))) if not missing.empty else 0
    confidence_score = max(0, min(100, 96 - problem_count * 28 - check_count * 8 - missing_penalty))
    confidence_label = "High" if confidence_score >= 85 else "Medium" if confidence_score >= 65 else "Needs checking"

    interval_available = not interval.empty and "interval_kwh" in interval.columns
    dnp_cols = [c for c in ["night_kwh", "day_off_peak_kwh", "peak_kwh"] if c in daily.columns]
    export_available = "export_kwh" in daily.columns and float(daily["export_kwh"].fillna(0).sum()) > 0
    estimated_reads = 0
    for item in parsed or []:
        if "read_type" in item.frame.columns:
            estimated_reads += int(item.frame["read_type"].astype(str).str.contains("estimate|estimated", case=False, na=False).sum())

    hourly = pd.DataFrame()
    peak_hours = pd.DataFrame()
    baseload_kw = np.nan
    overnight_kwh = np.nan
    if interval_available:
        hourly = interval.dropna(subset=["interval_kwh"]).groupby("hour", as_index=False).agg(avg_kwh=("interval_kwh", "mean"))
        hourly = make_unique_columns(hourly)
        if not hourly.empty:
            hourly["avg_kw"] = hourly["avg_kwh"] * 2
            peak_hours = hourly.nlargest(4, "avg_kw")
            overnight_kwh = float(interval.loc[interval["hour"].isin([23, 0, 1, 2, 3, 4, 5, 6]), "interval_kwh"].sum())
            baseload_values = hourly.loc[hourly["hour"].between(2, 5), "avg_kw"]
            baseload_kw = float(baseload_values.median()) if not baseload_values.empty else np.nan

    st.markdown("## Your electricity dashboard")
    if dataset.get("source") == "demo":
        st.info("Demo mode is using synthetic sample data. Upload your own ESB HDF files for real results.")
    for message in messages:
        st.warning(message)

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        metric_card("Total usage", kwh(total_kwh), f"{min_day:%d %b} to {max_day:%d %b}")
    with kpi_cols[1]:
        metric_card("Estimated cost", euro(total_cost["total"]), "Usage + fixed charges")
    with kpi_cols[2]:
        metric_card("Average day", kwh(avg_daily), f"Median {kwh(median_daily)}")
    with kpi_cols[3]:
        metric_card("Annualised cost", euro(total_cost["annualised_cost"]), "At tariff inputs")
    with kpi_cols[4]:
        metric_card("Confidence", f"{confidence_score}/100", confidence_label)

    tabs = st.tabs(["Overview", "Usage Patterns", "Time-of-Day Analysis", "Tariff & Cost Analysis", "Data Quality"])

    with tabs[0]:
        st.markdown("### Bottom line")
        if tariff.get("supplier_app_kwh") is not None:
            app_kwh = float(tariff["supplier_app_kwh"])
            ratio = total_kwh / max(app_kwh, 0.01)
            if ratio > 1.2 or ratio < 0.8:
                st.error(f"Supplier app discrepancy detected: ESB-derived usage is {kwh(total_kwh)} versus {kwh(app_kwh)} in the app, about {ratio:.1f}x different.")
            else:
                st.success("The supplier app figure entered is broadly consistent with the uploaded ESB data.")
        else:
            st.info("Add a supplier app kWh figure in the sidebar if you want an app-accuracy check.")

        overview_metrics = st.columns(4)
        with overview_metrics[0]:
            metric_card("Highest day", kwh(peak_day["usage_kwh"]), pd.Timestamp(peak_day["usage_date"]).strftime("%d %b %Y"))
        with overview_metrics[1]:
            metric_card("Quietest day", kwh(quiet_day["usage_kwh"]), pd.Timestamp(quiet_day["usage_date"]).strftime("%d %b %Y"))
        with overview_metrics[2]:
            metric_card("Annualised usage", kwh(total_cost["annualised_kwh"]), "Based on uploaded period")
        with overview_metrics[3]:
            metric_card("Data quality", confidence_label, f"{confidence_score}/100")

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Monthly totals")
            chart_monthly = safe_plot_df(monthly)
            fig = px.bar(chart_monthly, x="month", y="total_kwh", text=chart_monthly["total_kwh"].round(1), labels={"month": "Month", "total_kwh": "kWh"})
            fig.update_xaxes(ticktext=[month_label(m) for m in chart_monthly["month"]], tickvals=chart_monthly["month"])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Estimated monthly cost")
            fig = px.bar(chart_monthly, x="month", y="total_estimated_cost_eur", text=chart_monthly["total_estimated_cost_eur"].round(0), labels={"month": "Month", "total_estimated_cost_eur": "€"})
            fig.update_xaxes(ticktext=[month_label(m) for m in chart_monthly["month"]], tickvals=chart_monthly["month"])
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key insights")
        insight_cols = st.columns(3)
        with insight_cols[0]:
            peak_text = "Upload interval data to reveal the busiest hours."
            if not peak_hours.empty:
                peak_text = "Your strongest average hours are " + ", ".join(f"{int(h)}:00" for h in peak_hours["hour"].head(3)) + "."
            insight_card("When usage rises", peak_text, "Peak windows are where household behaviour most visibly affects the bill.", "Cooking, hot water, laundry, showering, heating controls, or several appliances together.", "Compare high-use days with what was happening during those hours.")
        with insight_cols[1]:
            baseload_text = "Interval data is needed for a baseload estimate."
            if not np.isnan(baseload_kw):
                baseload_text = f"Your rough overnight baseload is about {baseload_kw:.2f} kW."
            insight_card("Always-on electricity", baseload_text, "Small continuous loads become meaningful because they run every day.", "Fridge/freezer, router, standby devices, pumps, chargers, dehumidifier, or timed water heating.", "Check timers and anything left running, warm, humming, charging, or cycling overnight.")
        with insight_cols[2]:
            insight_card("Cost drivers", f"The uploaded period costs about {euro(total_cost['total'])} at the tariff entered.", "Final bills include both energy use and fixed daily charges.", "High kWh days drive the usage charge; standing charge and PSO continue on quiet days.", "Separate usage changes from fixed charges when checking bills.")

        st.markdown("### Recommendations")
        recommendations = pd.DataFrame(
            [
                ["1", "Compare the next supplier bill with this dashboard", "Bills can include standing charges, PSO, VAT, credits and adjustments."],
                ["2", "If the supplier app disagrees, ask for a smart meter data refresh", "The ESB export is the stronger evidence source."],
                ["3", "Investigate the highest-use days", "A small number of routines often explain most spikes."],
                ["4", "Look at evening and overnight patterns", "These often reveal cooking, hot water, standby load, or timed appliances."],
                ["5", "Download another ESB export next month", "A second month confirms whether patterns are persistent."],
            ],
            columns=["Rank", "Action", "Why it helps"],
        )
        st.dataframe(make_unique_columns(recommendations), hide_index=True, width="stretch")
        with st.expander("Copy/paste supplier support message"):
            st.text_area("Supplier message", value=supplier_message(dataset, tariff, total_kwh, total_cost["total"], min_day, max_day), height=280)

    with tabs[1]:
        st.markdown("### Usage patterns")
        st.write("Use this tab to spot high days, quiet days, and whether your normal daily usage is drifting upward or downward.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Daily electricity usage")
            fig = px.line(safe_plot_df(daily), x="usage_date", y="usage_kwh", markers=True, labels={"usage_date": "Date", "usage_kwh": "kWh/day"})
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Rolling 7-day average")
            chart_daily = safe_plot_df(daily)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_daily["usage_date"], y=chart_daily["usage_kwh"], mode="lines+markers", name="Daily kWh", opacity=0.45))
            fig.add_trace(go.Scatter(x=chart_daily["usage_date"], y=chart_daily["rolling_7_day_avg_kwh"], mode="lines", name="7-day average", line=dict(width=4)))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="kWh/day")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Highest usage days")
            top_days = safe_plot_df(daily.nlargest(10, "usage_kwh").sort_values("usage_kwh"))
            fig = px.bar(top_days, x="usage_kwh", y=top_days["usage_date"].dt.strftime("%d %b"), orientation="h", labels={"usage_kwh": "kWh", "y": "Date"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.markdown("#### Distribution of daily usage")
            fig = px.histogram(safe_plot_df(daily), x="usage_kwh", nbins=18, labels={"usage_kwh": "kWh/day"})
            fig.add_vline(x=avg_daily, line_dash="dash", line_color="#0f766e", annotation_text="Average")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Days")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Weekday vs weekend")
        weekday = daily.groupby("is_weekend", as_index=False).agg(avg_kwh=("usage_kwh", "mean"), median_kwh=("usage_kwh", "median"), days=("usage_kwh", "count"))
        weekday["period"] = weekday["is_weekend"].map({False: "Weekday", True: "Weekend"})
        weekday = make_unique_columns(weekday)
        fig = px.bar(safe_plot_df(weekday), x="period", y=["avg_kwh", "median_kwh"], barmode="group", labels={"period": "", "value": "kWh/day", "variable": "Measure"})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("### Time-of-day analysis")
        st.write("This view turns 30-minute meter readings into clues about routine. It cannot identify exact appliances, but it shows when to investigate.")
        if interval_available and not hourly.empty:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown("#### Average hourly usage profile")
                fig = px.line(safe_plot_df(hourly), x="hour", y="avg_kw", markers=True, labels={"hour": "Hour starting", "avg_kw": "Average kW"})
                fig.update_xaxes(dtick=2)
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### Peak usage windows")
                peak_table = peak_hours.copy()
                peak_table["Window"] = peak_table["hour"].map(lambda h: f"{int(h):02d}:00-{(int(h) + 1) % 24:02d}:00")
                peak_table["Average kW"] = peak_table["avg_kw"]
                st.dataframe(make_unique_columns(peak_table[["Window", "Average kW"]]).style.format({"Average kW": "{:.2f}"}), hide_index=True, width="stretch")
                if not np.isnan(baseload_kw):
                    metric_card("Baseload estimate", f"{baseload_kw:.2f} kW", "Median average load from 02:00 to 05:00")
                if not np.isnan(overnight_kwh):
                    metric_card("Overnight share", f"{overnight_kwh / max(total_kwh, 0.01) * 100:.0f}%", f"{kwh(overnight_kwh)} from 23:00-07:00")

            st.markdown("#### Day/hour heatmap")
            heat = safe_plot_df(interval.dropna(subset=["interval_kwh"]).copy())
            heat["date_label"] = pd.to_datetime(heat["usage_date"]).dt.strftime("%d %b")
            heat_table = heat.pivot_table(index="date_label", columns="hour", values="interval_kwh", aggfunc="sum")
            fig = go.Figure(data=go.Heatmap(z=heat_table.values, x=heat_table.columns, y=heat_table.index, colorscale="YlOrRd", colorbar=dict(title="kWh")))
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Hour starting", yaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Possible appliance behaviour")
            a1, a2 = st.columns(2)
            with a1:
                insight_card("Evening routines", "Sharp rises after 17:00 often point to normal household activity.", "Evening peaks are common, but expensive if several high-power appliances overlap.", "Cooking, electric shower, immersion, tumble dryer, dishwasher, washing machine, or heating controls.", "On the next high-use evening, note what runs during the top peak hours.")
            with a2:
                insight_card("Overnight usage", f"The overnight baseload estimate is {baseload_kw:.2f} kW.", "Always-on use is easy to miss because it does not feel like an event.", "Fridge/freezer, router, standby loads, pumps, dehumidifier, timed hot water, or charging.", "Try a quick evening audit of timers and always-on devices.")
            st.markdown("#### Typical appliance guide")
            st.dataframe(make_unique_columns(APPLIANCE_EXAMPLES), hide_index=True, width="stretch")
        else:
            st.info("Upload a 30-minute kWh or kW file to unlock hourly profiles, heatmaps, baseload estimates, and appliance-timing clues.")

    with tabs[3]:
        st.markdown("### Tariff and estimated cost analysis")
        st.write("This uses the tariff inputs in the sidebar. It is an estimate, because real bills can include prior balances, discounts, credits, VAT treatment, and corrections.")

        if dnp_cols:
            st.markdown("#### Day / night / peak usage split")
            split = pd.DataFrame(
                {
                    "Period": ["Night", "Day off-peak", "Peak"],
                    "kWh": [
                        float(daily.get("night_kwh", pd.Series(dtype=float)).sum()),
                        float(daily.get("day_off_peak_kwh", pd.Series(dtype=float)).sum()),
                        float(daily.get("peak_kwh", pd.Series(dtype=float)).sum()),
                    ],
                }
            )
            split["Estimated usage charge"] = split["kWh"] * tariff["unit_rate_cent"] / 100
            d1, d2 = st.columns(2)
            with d1:
                fig = px.bar(safe_plot_df(split), x="Period", y="kWh", text=split["kWh"].round(1), labels={"kWh": "kWh"})
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with d2:
                fig = px.pie(safe_plot_df(split), names="Period", values="kWh", hole=0.45)
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload the daily day/night/peak file to see tariff-period behaviour. Usage and cost estimates still work from daily or interval totals.")

        st.markdown("#### Estimated bill breakdown")
        breakdown = pd.DataFrame(
            [
                ["Usage charge", total_cost["usage_charge"]],
                ["Standing charge", total_cost["standing_charge"]],
                ["PSO levy", total_cost["pso_levy"]],
                ["Export credit", -total_cost["export_credit"]],
            ],
            columns=["Component", "EUR"],
        )
        b1, b2 = st.columns([1.1, 1])
        with b1:
            fig = px.bar(safe_plot_df(breakdown), x="Component", y="EUR", text=breakdown["EUR"].map(lambda x: f"€{x:.2f}"))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="€")
            st.plotly_chart(fig, use_container_width=True)
        with b2:
            metric_card("Usage charge", euro(total_cost["usage_charge"]), f"{tariff['unit_rate_cent']:.2f}c/kWh")
            metric_card("Fixed charges", euro(total_cost["standing_charge"] + total_cost["pso_levy"]), "Standing charge + PSO")
            metric_card("Export credit", euro(total_cost["export_credit"]), "Only when export data exists or is entered")

        st.markdown("### Tariff simulator")
        s1, s2 = st.columns(2)
        with s1:
            start = st.date_input("Billing period start", value=min_day, min_value=min_day, max_value=max_day, key="analytics_cost_start")
        with s2:
            end = st.date_input("Billing period end", value=max_day, min_value=min_day, max_value=max_day, key="analytics_cost_end")
        if start > end:
            st.warning("Start date must be before end date.")
        else:
            selected = daily[(daily["usage_date"].dt.date >= start) & (daily["usage_date"].dt.date <= end)]
            selected_kwh = float(selected["usage_kwh"].sum())
            selected_export = float(selected.get("export_kwh", pd.Series(dtype=float)).sum())
            sim_cols = st.columns(4)
            with sim_cols[0]:
                sim_unit_rate = st.number_input("Unit rate, cent/kWh", min_value=0.0, value=float(tariff["unit_rate_cent"]), step=0.1, key="analytics_sim_unit")
            with sim_cols[1]:
                sim_standing = st.number_input("Standing charge, €/year", min_value=0.0, value=float(tariff["standing_charge_year"]), step=1.0, key="analytics_sim_standing")
            with sim_cols[2]:
                sim_pso = st.number_input("PSO levy, €/year", min_value=0.0, value=float(tariff["pso_levy_year"]), step=1.0, key="analytics_sim_pso")
            with sim_cols[3]:
                export_for_period = st.number_input("Export kWh, if known", min_value=0.0, value=selected_export, step=1.0, key="analytics_sim_export")
            sim_tariff = {**tariff, "unit_rate_cent": sim_unit_rate, "standing_charge_year": sim_standing, "pso_levy_year": sim_pso}
            days = (end - start).days + 1
            selected_cost = cost_for_period(selected_kwh, days, sim_tariff, export_for_period)
            cols = st.columns(5)
            with cols[0]:
                metric_card("Selected usage", kwh(selected_kwh), f"{days} days")
            with cols[1]:
                metric_card("Estimated cost", euro(selected_cost["total"]), "Usage + standing + PSO - export")
            with cols[2]:
                metric_card("Usage charge", euro(selected_cost["usage_charge"]), f"{sim_unit_rate:.2f}c/kWh")
            with cols[3]:
                metric_card("Fixed charges", euro(selected_cost["standing_charge"] + selected_cost["pso_levy"]), "Standing + PSO")
            with cols[4]:
                metric_card("Annualised cost", euro(selected_cost["annualised_cost"]), "Based on selected period")

            st.markdown("#### What-if comparisons")
            scenario_rows = []
            for label, rate_multiplier, standing_multiplier in [
                ("Current inputs", 1.0, 1.0),
                ("Unit rate 10% lower", 0.9, 1.0),
                ("Unit rate 10% higher", 1.1, 1.0),
                ("Standing charge 10% lower", 1.0, 0.9),
                ("Standing charge 10% higher", 1.0, 1.1),
            ]:
                scenario_tariff = {**sim_tariff, "unit_rate_cent": sim_unit_rate * rate_multiplier, "standing_charge_year": sim_standing * standing_multiplier}
                scenario_cost = cost_for_period(selected_kwh, days, scenario_tariff, export_for_period)
                scenario_rows.append([label, scenario_tariff["unit_rate_cent"], scenario_tariff["standing_charge_year"], scenario_cost["total"], scenario_cost["annualised_cost"]])
            scenarios = pd.DataFrame(scenario_rows, columns=["Scenario", "Unit rate cent/kWh", "Standing charge €/year", "Selected period cost", "Annualised cost"])
            fig = px.bar(safe_plot_df(scenarios), x="Scenario", y="Selected period cost", text=scenarios["Selected period cost"].map(lambda x: f"€{x:.0f}"))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="€")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(make_unique_columns(scenarios).style.format({"Unit rate cent/kWh": "{:.2f}", "Standing charge €/year": "€{:.2f}", "Selected period cost": "€{:.2f}", "Annualised cost": "€{:.2f}"}), hide_index=True, width="stretch")

            if tariff.get("supplier_app_kwh") is not None:
                app_kwh = float(tariff["supplier_app_kwh"])
                app_cost = cost_for_period(app_kwh, days, sim_tariff, export_for_period)
                comparison = pd.DataFrame([["ESB-derived usage", selected_kwh, selected_cost["total"]], ["Supplier app reported usage", app_kwh, app_cost["total"]]], columns=["Scenario", "kWh", "Estimated total cost"])
                fig = px.bar(safe_plot_df(comparison), x="Scenario", y="kWh", text=comparison["kWh"].round(1))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

        if export_available:
            st.success(f"Export data appears to be present. Estimated uploaded-period export credit is {euro(total_cost['export_credit'])}.")
        else:
            st.caption("No export data was detected. Solar export credits are only included when export kWh is uploaded or entered in the simulator.")

        st.markdown("#### Monthly estimated costs")
        st.dataframe(
            make_unique_columns(monthly[["month", "total_kwh", "days", "usage_charge_eur", "standing_charge_eur", "pso_levy_eur", "total_estimated_cost_eur"]]).style.format(
                {
                    "total_kwh": "{:.1f}",
                    "usage_charge_eur": "€{:.2f}",
                    "standing_charge_eur": "€{:.2f}",
                    "pso_levy_eur": "€{:.2f}",
                    "total_estimated_cost_eur": "€{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    with tabs[4]:
        st.markdown("### Data quality")
        st.write("These checks tell you whether the upload is reliable enough for homeowner decisions, supplier conversations, and bill sense-checks.")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            metric_card("Confidence score", f"{confidence_score}/100", confidence_label)
        with q2:
            metric_card("Missing intervals", f"{len(missing):,}", "30-minute slots")
        with q3:
            metric_card("Estimated reads", f"{estimated_reads:,}", "Detected from read type text")
        with q4:
            metric_card("Uploaded files", f"{len(parsed or []):,}", "Detected automatically")

        with st.expander("Uploaded data diagnostics", expanded=False):
            if parsed is not None:
                render_file_detection(parsed)
            duplicate_rows = []
            for label, frame in [("daily", daily), ("interval", interval), ("monthly", monthly), ("quality", quality), ("missing intervals", missing)]:
                duplicate_rows.append({"Dataframe": label, "Columns": ", ".join(map(str, frame.columns)), "Duplicate columns currently present": ", ".join(duplicate_columns(frame)) or "None"})
            st.dataframe(make_unique_columns(pd.DataFrame(duplicate_rows)), hide_index=True, width="stretch")

        if not missing.empty:
            st.warning(f"Missing interval range: {missing['missing_timestamp'].min()} to {missing['missing_timestamp'].max()}.")
            gaps = missing.copy()
            gaps["missing_timestamp"] = pd.to_datetime(gaps["missing_timestamp"])
            gaps = gaps.sort_values("missing_timestamp")
            gaps["gap_id"] = (gaps["missing_timestamp"].diff().dt.total_seconds().fillna(1800) > 1800).cumsum()
            gap_summary = gaps.groupby("gap_id", as_index=False).agg(gap_start=("missing_timestamp", "min"), gap_end=("missing_timestamp", "max"), missing_intervals=("missing_timestamp", "count"))
            gap_summary["approx_hours"] = gap_summary["missing_intervals"] * 0.5
            st.markdown("#### Missing interval gaps")
            st.dataframe(make_unique_columns(gap_summary).style.format({"approx_hours": "{:.1f}"}), hide_index=True, width="stretch")
        else:
            st.success("No missing 30-minute intervals were detected in the interval file.")

        st.markdown("#### Quality check results")
        status_order = {"Problem": 0, "Check": 1, "OK": 2}
        quality_display = quality.copy()
        quality_display["_sort"] = quality_display["status"].map(status_order).fillna(3)
        quality_display = quality_display.sort_values("_sort").drop(columns=["_sort"])
        for _, row in quality_display.iterrows():
            st.markdown(f"<p><span class='{status_class(row['status'])}'>{row['status']}</span> — <b>{row['check']}</b>: {row['value']}</p>", unsafe_allow_html=True)

        st.markdown("#### Plain-English readout")
        if "Problem" in set(quality["status"]):
            st.error("There are data issues worth checking before relying on exact totals.")
        elif "Check" in set(quality["status"]):
            st.warning("The data is usable, but at least one check deserves attention.")
        else:
            st.success("The uploaded meter data looks usable for homeowner-level insight.")
        st.write("A supplier app can still be stale or partially synced even when the ESB export itself looks consistent.")
        with st.expander("Assumptions"):
            st.markdown(
                """
                - ESB timestamps are treated as local Irish meter time.
                - Interval readings are treated as 30-minute values ending at the timestamp shown.
                - Daily register files may be cumulative, so daily usage can be calculated by subtracting one day from the next.
                - If both daily and interval files are present, daily register totals are preferred for daily/monthly totals.
                - Export credit is only applied when export data is uploaded or entered in the cost simulator.
                """
            )


def render_upload_flow() -> tuple[dict | None, list[ParsedUpload]]:
    st.markdown("## Start here")
    mode = st.radio("Choose data source", ["Upload ESB CSV files", "Try demo mode"], horizontal=True)
    parsed: list[ParsedUpload] = []
    if mode == "Try demo mode":
        return demo_dataset(), parsed

    uploaded = st.file_uploader("Upload ESB Networks HDF CSV files", type=["csv"], accept_multiple_files=True)
    if not uploaded:
        st.info("Upload one or more ESB HDF CSV files to generate the dashboard. You can also switch to demo mode.")
        return None, parsed
    parsed = parse_uploads(uploaded)
    render_file_detection(parsed)
    dataset = build_dataset(parsed)
    if "error" in dataset:
        st.error(str(dataset["error"]))
        for message in dataset.get("messages", []):
            st.warning(str(message))
        return None, parsed
    return dataset, parsed


def tariff_form() -> dict:
    st.sidebar.header("Tariff details")
    supplier_name = st.sidebar.text_input("Supplier name", value=DEFAULT_TARIFF["supplier_name"], placeholder="e.g. Electric Ireland")
    plan_name = st.sidebar.text_input("Plan name", value=DEFAULT_TARIFF["plan_name"], placeholder="e.g. 24hr smart tariff")
    unit_rate_cent = st.sidebar.number_input("Unit rate, cent/kWh", min_value=0.0, value=DEFAULT_TARIFF["unit_rate_cent"], step=0.1)
    standing_charge_year = st.sidebar.number_input("Standing charge, €/year", min_value=0.0, value=DEFAULT_TARIFF["standing_charge_year"], step=1.0)
    pso_levy_year = st.sidebar.number_input("PSO levy, €/year", min_value=0.0, value=DEFAULT_TARIFF["pso_levy_year"], step=1.0)
    export_rate_cent = st.sidebar.number_input("Export rate, cent/kWh", min_value=0.0, value=DEFAULT_TARIFF["export_rate_cent"], step=0.1)
    app_toggle = st.sidebar.checkbox("I have a supplier app usage figure to compare")
    supplier_app_kwh = None
    if app_toggle:
        supplier_app_kwh = st.sidebar.number_input("Supplier app reported kWh", min_value=0.0, value=0.0, step=1.0)
    st.sidebar.caption("Enter rates including VAT where possible. If you are unsure, start with your latest bill or supplier tariff sheet.")
    return {
        "supplier_name": supplier_name,
        "plan_name": plan_name,
        "unit_rate_cent": unit_rate_cent,
        "standing_charge_year": standing_charge_year,
        "pso_levy_year": pso_levy_year,
        "export_rate_cent": export_rate_cent,
        "supplier_app_kwh": supplier_app_kwh,
    }


def main() -> None:
    landing_page()
    privacy_note()
    with st.expander("How to get your ESB data", expanded=True):
        onboarding_guide()
    tariff = tariff_form()
    dataset, parsed = render_upload_flow()
    if dataset is not None:
        render_analytics_dashboard(dataset, tariff, parsed)


if __name__ == "__main__":
    main()

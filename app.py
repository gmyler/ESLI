from __future__ import annotations

import math
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st


# =============================================================================
# ESLI Decision Twin - Enterprise-Scale Streamlit POC
# =============================================================================
#
# Expected input folder:
#
# data/systems.csv
# data/applications.csv
# data/business_services.csv
# data/dependencies.csv
# data/cost_lines.csv
# data/licensing_entitlements.csv
# data/vendor_claims.csv
# data/oem_lifecycle.csv
# data/vulnerabilities.csv
# data/security_controls.csv
# data/regulatory_mappings.csv
# data/market_signals.csv
# data/scenario_results.csv
# data/execution_actions.csv
# data/decision_outcomes.csv
# data/_dataset_manifest.csv
#
# =============================================================================


st.set_page_config(
    page_title="ESLI Decision Twin",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_FILES = {
    "systems": "systems.csv",
    "applications": "applications.csv",
    "business_services": "business_services.csv",
    "dependencies": "dependencies.csv",
    "cost_lines": "cost_lines.csv",
    "licensing_entitlements": "licensing_entitlements.csv",
    "vendor_claims": "vendor_claims.csv",
    "oem_lifecycle": "oem_lifecycle.csv",
    "vulnerabilities": "vulnerabilities.csv",
    "security_controls": "security_controls.csv",
    "regulatory_mappings": "regulatory_mappings.csv",
    "market_signals": "market_signals.csv",
    "scenario_results": "scenario_results.csv",
    "execution_actions": "execution_actions.csv",
    "decision_outcomes": "decision_outcomes.csv",
    "manifest": "_dataset_manifest.csv",
}


SCENARIO_ORDER = [
    "Retain As-Is",
    "Extend Independent Support",
    "Modernise / Upgrade",
    "Replace / Migrate",
    "Exit / Decommission",
    "Hybrid Transformation",
]


DECISION_COLORS = {
    "Retain As-Is": "#2ca02c",
    "Extend Independent Support": "#1f77b4",
    "Modernise / Upgrade": "#ff7f0e",
    "Replace / Migrate": "#9467bd",
    "Exit / Decommission": "#7f7f7f",
    "Hybrid Transformation": "#d62728",
}


CHALLENGEABLE_CLAIMS = {
    "Phantom Upgrade",
    "Commercial Pressure",
    "Vendor Narrative",
    "Negotiation Lever",
    "Overstated",
    "Partially True",
    "Context Dependent",
    "Requires Evidence",
    "Needs Entitlement Review",
}


# =============================================================================
# Basic helpers
# =============================================================================


def fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "€0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000_000:
        return f"{sign}€{value / 1_000_000_000:.2f}bn"
    if value >= 1_000_000:
        return f"{sign}€{value / 1_000_000:.2f}m"
    if value >= 1_000:
        return f"{sign}€{value / 1_000:.1f}k"
    return f"{sign}€{value:,.0f}"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{float(value):,.0f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0%"
    return f"{float(value):.1f}%"


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def boolish(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(bool)

    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def clamp(series_or_array, lo: float = 0.0, hi: float = 100.0):
    return np.minimum(np.maximum(series_or_array, lo), hi)


def unique_sorted(series: pd.Series) -> list[str]:
    if series is None or series.empty:
        return []
    values = series.dropna().astype(str).unique().tolist()
    return sorted(values)


def columns_present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def show_empty(message: str = "No data available for this view.") -> None:
    st.info(message)


def download_df_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def as_set(values: pd.Series | list[str]) -> set[str]:
    if isinstance(values, pd.Series):
        return set(values.dropna().astype(str).tolist())
    return set(str(v) for v in values if pd.notna(v))


def filter_by_system_ids(df: pd.DataFrame, system_ids: set[str]) -> pd.DataFrame:
    if df.empty or "system_id" not in df.columns or not system_ids:
        return empty_df()
    return df[df["system_id"].astype(str).isin(system_ids)].copy()


def filter_by_vendor_product(
    df: pd.DataFrame,
    vendors: set[str] | None = None,
    products: set[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if vendors and "vendor" in out.columns:
        out = out[out["vendor"].astype(str).isin(vendors)]

    if products and "product" in out.columns:
        out = out[out["product"].astype(str).isin(products)]

    return out


def score_bucket(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Unknown"
    score = float(score)
    if score >= 80:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def first_existing(row: pd.Series, candidates: list[str], default: str = ""):
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return row[c]
    return default


# =============================================================================
# Data loading
# =============================================================================


@st.cache_data(show_spinner="Loading ESLI data files...")
def load_all_data(data_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(data_dir)
    data: dict[str, pd.DataFrame] = {}

    for key, filename in REQUIRED_FILES.items():
        path = base / filename
        if not path.exists():
            data[key] = empty_df()
            continue

        try:
            data[key] = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            st.warning(f"Could not load {filename}: {exc}")
            data[key] = empty_df()

    return data


def file_health(data_dir: str) -> pd.DataFrame:
    base = Path(data_dir)
    rows = []

    for key, filename in REQUIRED_FILES.items():
        path = base / filename
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0

        rows.append(
            {
                "table": key,
                "file": filename,
                "exists": exists,
                "size_mb": round(size_mb, 2),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Portfolio enrichment / ESLI fusion layer
# =============================================================================


def map_criticality(series: pd.Series) -> pd.Series:
    mapping = {
        "Low": 20,
        "Medium": 45,
        "High": 70,
        "Critical": 90,
    }
    return series.astype(str).map(mapping).fillna(45).astype(float)


def map_risk(series: pd.Series) -> pd.Series:
    mapping = {
        "Low": 25,
        "Medium": 50,
        "High": 75,
        "Critical": 90,
        "Severe": 92,
    }
    return series.astype(str).map(mapping).fillna(50).astype(float)


def map_sensitivity(series: pd.Series) -> pd.Series:
    mapping = {
        "Public": 10,
        "Internal": 35,
        "Confidential": 65,
        "Restricted": 90,
    }
    return series.astype(str).map(mapping).fillna(35).astype(float)


def map_support_pressure(series: pd.Series) -> pd.Series:
    mapping = {
        "Supported": 20,
        "Vendor Announced EOL": 70,
        "Extended Support": 78,
        "Unsupported": 92,
    }
    return series.astype(str).map(mapping).fillna(50).astype(float)


def map_technical_debt(series: pd.Series) -> pd.Series:
    mapping = {
        "Low": 20,
        "Medium": 50,
        "High": 75,
        "Severe": 92,
    }
    return series.astype(str).map(mapping).fillna(50).astype(float)


def fallback_decision(row: pd.Series) -> str:
    lifecycle = float(row.get("lifecycle_pressure_score", 50))
    risk = float(row.get("risk_fusion_score", 50))
    blast = float(row.get("blast_radius_score", 50))
    vendor = float(row.get("vendor_coercion_score", 50))
    technical = float(row.get("technical_debt_num", 50))
    criticality = str(row.get("criticality", "Medium"))
    lifecycle_stage = str(row.get("lifecycle_stage", ""))

    if lifecycle_stage == "Retire" and criticality in {"Low", "Medium"}:
        return "Exit / Decommission"

    if risk >= 78 and blast >= 70:
        return "Hybrid Transformation"

    if vendor >= 72 and blast >= 65 and risk < 78:
        return "Extend Independent Support"

    if risk >= 72 and blast < 65:
        return "Modernise / Upgrade"

    if technical >= 82 and blast >= 60:
        return "Replace / Migrate"

    if lifecycle < 45 and risk < 55 and vendor < 55:
        return "Retain As-Is"

    if lifecycle >= 68 or vendor >= 68:
        return "Extend Independent Support"

    return "Hybrid Transformation"


@st.cache_data(show_spinner="Building ESLI intelligence profile...")
def build_system_profile(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    systems = data.get("systems", empty_df()).copy()

    if systems.empty:
        return empty_df()

    required_defaults = {
        "system_id": "",
        "system_name": "",
        "asset_type": "Unknown",
        "business_service_id": "",
        "business_service_name": "",
        "business_unit": "Unknown",
        "vendor": "Unknown",
        "product": "Unknown",
        "product_family": "Unknown",
        "version": "Unknown",
        "environment": "Unknown",
        "hosting_model": "Unknown",
        "region": "Unknown",
        "owner_team": "Unknown",
        "criticality": "Medium",
        "internet_exposed": False,
        "data_sensitivity": "Internal",
        "lifecycle_stage": "Tolerate",
        "support_status": "Supported",
        "security_risk": "Medium",
        "technical_debt": "Medium",
        "regulatory_exposure": "Medium",
        "annual_run_cost_eur": 0,
        "cmdb_confidence": 0.5,
        "duplicate_group_id": "",
    }

    for col, default in required_defaults.items():
        if col not in systems.columns:
            systems[col] = default

    systems["system_id"] = systems["system_id"].astype(str)
    systems["annual_run_cost_eur"] = safe_numeric(systems["annual_run_cost_eur"])
    systems["cmdb_confidence"] = safe_numeric(systems["cmdb_confidence"], default=0.5)
    systems["internet_exposed"] = boolish(systems["internet_exposed"])

    profile = systems.copy()

    deps = data.get("dependencies", empty_df()).copy()
    if not deps.empty:
        if {"source_system_id", "target_system_id"}.issubset(deps.columns):
            dep_out = (
                deps.groupby("source_system_id")
                .agg(
                    downstream_dependency_count=("target_system_id", "nunique"),
                    outbound_integration_count=("target_system_id", "count"),
                )
                .reset_index()
                .rename(columns={"source_system_id": "system_id"})
            )

            dep_in = (
                deps.groupby("target_system_id")
                .agg(
                    upstream_dependency_count=("source_system_id", "nunique"),
                    inbound_integration_count=("source_system_id", "count"),
                )
                .reset_index()
                .rename(columns={"target_system_id": "system_id"})
            )

            profile = profile.merge(dep_out, on="system_id", how="left")
            profile = profile.merge(dep_in, on="system_id", how="left")

            if "relationship_confidence" in deps.columns:
                dep_conf_out = (
                    deps.assign(relationship_confidence=safe_numeric(deps["relationship_confidence"], 0.5))
                    .groupby("source_system_id")
                    .agg(avg_outbound_relationship_confidence=("relationship_confidence", "mean"))
                    .reset_index()
                    .rename(columns={"source_system_id": "system_id"})
                )
                profile = profile.merge(dep_conf_out, on="system_id", how="left")

    cost = data.get("cost_lines", empty_df()).copy()
    if not cost.empty and {"system_id", "amount_eur"}.issubset(cost.columns):
        cost["amount_eur"] = safe_numeric(cost["amount_eur"])

        cost_agg = (
            cost.groupby("system_id")
            .agg(
                total_cost_lines_eur=("amount_eur", "sum"),
                cost_line_count=("amount_eur", "count"),
            )
            .reset_index()
        )
        profile = profile.merge(cost_agg, on="system_id", how="left")

    ent = data.get("licensing_entitlements", empty_df()).copy()
    if not ent.empty and "system_id" in ent.columns:
        for col in ["annual_maintenance_eur", "shelfware_units", "over_deployed_units"]:
            if col in ent.columns:
                ent[col] = safe_numeric(ent[col])

        ent["audit_high_flag"] = ent.get("audit_risk", pd.Series("", index=ent.index)).astype(str).eq("High")

        ent_agg = (
            ent.groupby("system_id")
            .agg(
                entitlement_count=("system_id", "count"),
                annual_maintenance_eur=("annual_maintenance_eur", "sum"),
                shelfware_units=("shelfware_units", "sum"),
                over_deployed_units=("over_deployed_units", "sum"),
                high_audit_risk_entitlements=("audit_high_flag", "sum"),
            )
            .reset_index()
        )
        profile = profile.merge(ent_agg, on="system_id", how="left")

    claims = data.get("vendor_claims", empty_df()).copy()
    if not claims.empty and "system_id" in claims.columns:
        if "vendor_pressure_score" in claims.columns:
            claims["vendor_pressure_score"] = safe_numeric(claims["vendor_pressure_score"])
        else:
            claims["vendor_pressure_score"] = 50

        if "commercial_impact_eur" in claims.columns:
            claims["commercial_impact_eur"] = safe_numeric(claims["commercial_impact_eur"])
        else:
            claims["commercial_impact_eur"] = 0

        claims["challengeable_flag"] = claims.get("classification", pd.Series("", index=claims.index)).astype(str).isin(
            CHALLENGEABLE_CLAIMS
        )

        claims["phantom_upgrade_flag"] = claims.get("classification", pd.Series("", index=claims.index)).astype(str).eq(
            "Phantom Upgrade"
        )

        claims_agg = (
            claims.groupby("system_id")
            .agg(
                vendor_claim_count=("system_id", "count"),
                avg_vendor_pressure_score=("vendor_pressure_score", "mean"),
                max_vendor_pressure_score=("vendor_pressure_score", "max"),
                challengeable_claim_count=("challengeable_flag", "sum"),
                phantom_upgrade_claim_count=("phantom_upgrade_flag", "sum"),
                vendor_claim_commercial_impact_eur=("commercial_impact_eur", "sum"),
            )
            .reset_index()
        )
        profile = profile.merge(claims_agg, on="system_id", how="left")

    vulns = data.get("vulnerabilities", empty_df()).copy()
    if not vulns.empty and "system_id" in vulns.columns:
        if "risk_score" in vulns.columns:
            vulns["risk_score"] = safe_numeric(vulns["risk_score"])
        else:
            vulns["risk_score"] = map_risk(vulns.get("severity", pd.Series("Medium", index=vulns.index)))

        vulns["open_vuln_flag"] = ~vulns.get("status", pd.Series("", index=vulns.index)).astype(str).isin(
            ["Remediated"]
        )
        vulns["critical_vuln_flag"] = vulns.get("severity", pd.Series("", index=vulns.index)).astype(str).eq("Critical")
        vulns["exploitable_flag"] = boolish(vulns.get("exploit_available", pd.Series(False, index=vulns.index)))

        vuln_agg = (
            vulns.groupby("system_id")
            .agg(
                vulnerability_count=("system_id", "count"),
                open_vulnerability_count=("open_vuln_flag", "sum"),
                critical_vulnerability_count=("critical_vuln_flag", "sum"),
                exploitable_vulnerability_count=("exploitable_flag", "sum"),
                avg_vulnerability_risk_score=("risk_score", "mean"),
                max_vulnerability_risk_score=("risk_score", "max"),
            )
            .reset_index()
        )
        profile = profile.merge(vuln_agg, on="system_id", how="left")

    controls = data.get("security_controls", empty_df()).copy()
    if not controls.empty and "system_id" in controls.columns:
        for col in ["control_strength", "coverage_pct", "evidence_quality"]:
            if col in controls.columns:
                controls[col] = safe_numeric(controls[col])
            else:
                controls[col] = 50

        controls["compensating_control_flag"] = boolish(
            controls.get("compensating_control", pd.Series(False, index=controls.index))
        )

        controls_agg = (
            controls.groupby("system_id")
            .agg(
                control_count=("system_id", "count"),
                avg_control_strength=("control_strength", "mean"),
                avg_control_coverage_pct=("coverage_pct", "mean"),
                avg_control_evidence_quality=("evidence_quality", "mean"),
                compensating_control_count=("compensating_control_flag", "sum"),
            )
            .reset_index()
        )
        profile = profile.merge(controls_agg, on="system_id", how="left")

    regs = data.get("regulatory_mappings", empty_df()).copy()
    if not regs.empty and "system_id" in regs.columns:
        if "compliance_exposure_score" in regs.columns:
            regs["compliance_exposure_score"] = safe_numeric(regs["compliance_exposure_score"])
        else:
            regs["compliance_exposure_score"] = 50

        regs["weak_audit_defensibility_flag"] = regs.get("audit_defensibility", pd.Series("", index=regs.index)).astype(
            str
        ).eq("Weak")

        regs_agg = (
            regs.groupby("system_id")
            .agg(
                regulatory_mapping_count=("system_id", "count"),
                avg_compliance_exposure_score=("compliance_exposure_score", "mean"),
                max_compliance_exposure_score=("compliance_exposure_score", "max"),
                weak_audit_defensibility_count=("weak_audit_defensibility_flag", "sum"),
            )
            .reset_index()
        )
        profile = profile.merge(regs_agg, on="system_id", how="left")

    lifecycle = data.get("oem_lifecycle", empty_df()).copy()
    if not lifecycle.empty and {"vendor", "product", "version"}.issubset(lifecycle.columns):
        if "lifecycle_pressure_score" in lifecycle.columns:
            lifecycle["lifecycle_pressure_score"] = safe_numeric(lifecycle["lifecycle_pressure_score"])
        else:
            lifecycle["lifecycle_pressure_score"] = 50

        lifecycle_agg = (
            lifecycle.groupby(["vendor", "product", "version"])
            .agg(
                oem_notice_count=("vendor", "count"),
                oem_lifecycle_pressure_score=("lifecycle_pressure_score", "mean"),
                max_oem_lifecycle_pressure_score=("lifecycle_pressure_score", "max"),
            )
            .reset_index()
        )

        profile = profile.merge(lifecycle_agg, on=["vendor", "product", "version"], how="left")

    market = data.get("market_signals", empty_df()).copy()
    if not market.empty and {"vendor", "product"}.issubset(market.columns):
        for col in ["estimated_cost_impact_eur", "severity", "likelihood"]:
            if col in market.columns:
                market[col] = safe_numeric(market[col])
            else:
                market[col] = 0

        market["market_pressure_score"] = clamp((market["severity"] * 18) + (market["likelihood"] * 25), 0, 100)

        market_agg = (
            market.groupby(["vendor", "product"])
            .agg(
                market_signal_count=("vendor", "count"),
                avg_market_pressure_score=("market_pressure_score", "mean"),
                market_cost_impact_eur=("estimated_cost_impact_eur", "sum"),
            )
            .reset_index()
        )

        profile = profile.merge(market_agg, on=["vendor", "product"], how="left")

    scenarios = data.get("scenario_results", empty_df()).copy()
    if not scenarios.empty and {"system_id", "scenario", "projection_year"}.issubset(scenarios.columns):
        for col in [
            "projected_cost_eur",
            "risk_score",
            "disruption_score",
            "vendor_leverage_score",
            "regulatory_defensibility",
            "strategic_fit_score",
            "execution_complexity",
            "recommendation_score",
        ]:
            if col in scenarios.columns:
                scenarios[col] = safe_numeric(scenarios[col])

        max_year = int(safe_numeric(scenarios["projection_year"]).max())
        y5 = scenarios[safe_numeric(scenarios["projection_year"]) == max_year].copy()

        if not y5.empty:
            idx = y5.groupby("system_id")["recommendation_score"].idxmax()

            best = (
                y5.loc[idx]
                .rename(
                    columns={
                        "scenario": "best_scenario",
                        "projected_cost_eur": "best_5y_cost_eur",
                        "risk_score": "best_5y_risk_score",
                        "disruption_score": "best_5y_disruption_score",
                        "vendor_leverage_score": "best_vendor_leverage_score",
                        "regulatory_defensibility": "best_regulatory_defensibility",
                        "strategic_fit_score": "best_strategic_fit_score",
                        "execution_complexity": "best_execution_complexity",
                        "recommendation_score": "best_recommendation_score",
                    }
                )[
                    [
                        "system_id",
                        "best_scenario",
                        "best_5y_cost_eur",
                        "best_5y_risk_score",
                        "best_5y_disruption_score",
                        "best_vendor_leverage_score",
                        "best_regulatory_defensibility",
                        "best_strategic_fit_score",
                        "best_execution_complexity",
                        "best_recommendation_score",
                    ]
                ]
            )

            profile = profile.merge(best, on="system_id", how="left")

    numeric_fill_zero = [
        "downstream_dependency_count",
        "upstream_dependency_count",
        "outbound_integration_count",
        "inbound_integration_count",
        "total_cost_lines_eur",
        "cost_line_count",
        "entitlement_count",
        "annual_maintenance_eur",
        "shelfware_units",
        "over_deployed_units",
        "high_audit_risk_entitlements",
        "vendor_claim_count",
        "avg_vendor_pressure_score",
        "max_vendor_pressure_score",
        "challengeable_claim_count",
        "phantom_upgrade_claim_count",
        "vendor_claim_commercial_impact_eur",
        "vulnerability_count",
        "open_vulnerability_count",
        "critical_vulnerability_count",
        "exploitable_vulnerability_count",
        "avg_vulnerability_risk_score",
        "max_vulnerability_risk_score",
        "control_count",
        "avg_control_strength",
        "avg_control_coverage_pct",
        "avg_control_evidence_quality",
        "compensating_control_count",
        "regulatory_mapping_count",
        "avg_compliance_exposure_score",
        "max_compliance_exposure_score",
        "weak_audit_defensibility_count",
        "oem_notice_count",
        "oem_lifecycle_pressure_score",
        "max_oem_lifecycle_pressure_score",
        "market_signal_count",
        "avg_market_pressure_score",
        "market_cost_impact_eur",
        "best_5y_cost_eur",
        "best_5y_risk_score",
        "best_5y_disruption_score",
        "best_vendor_leverage_score",
        "best_regulatory_defensibility",
        "best_strategic_fit_score",
        "best_execution_complexity",
        "best_recommendation_score",
    ]

    for col in numeric_fill_zero:
        if col not in profile.columns:
            profile[col] = 0
        profile[col] = safe_numeric(profile[col])

    if "avg_outbound_relationship_confidence" not in profile.columns:
        profile["avg_outbound_relationship_confidence"] = 0.5
    profile["avg_outbound_relationship_confidence"] = safe_numeric(profile["avg_outbound_relationship_confidence"], 0.5)

    profile["criticality_num"] = map_criticality(profile["criticality"])
    profile["security_risk_num"] = map_risk(profile["security_risk"])
    profile["technical_debt_num"] = map_technical_debt(profile["technical_debt"])
    profile["regulatory_exposure_num"] = map_risk(profile["regulatory_exposure"])
    profile["data_sensitivity_num"] = map_sensitivity(profile["data_sensitivity"])
    profile["support_pressure_score"] = map_support_pressure(profile["support_status"])

    profile["avg_vulnerability_risk_score"] = np.where(
        profile["avg_vulnerability_risk_score"] <= 0,
        profile["security_risk_num"],
        profile["avg_vulnerability_risk_score"],
    )

    profile["avg_control_strength"] = np.where(
        profile["avg_control_strength"] <= 0,
        50,
        profile["avg_control_strength"],
    )

    profile["avg_vendor_pressure_score"] = np.where(
        profile["avg_vendor_pressure_score"] <= 0,
        profile["support_pressure_score"] * 0.65,
        profile["avg_vendor_pressure_score"],
    )

    profile["oem_lifecycle_pressure_score"] = np.where(
        profile["oem_lifecycle_pressure_score"] <= 0,
        profile["support_pressure_score"],
        profile["oem_lifecycle_pressure_score"],
    )

    total_dependencies = profile["downstream_dependency_count"] + profile["upstream_dependency_count"]
    total_integrations = profile["outbound_integration_count"] + profile["inbound_integration_count"]

    profile["blast_radius_score"] = clamp(
        np.log1p(total_dependencies) * 17
        + np.log1p(total_integrations) * 7
        + profile["criticality_num"] * 0.18
        + np.where(profile["business_service_name"].astype(str).str.len() > 0, 4, 0),
        0,
        100,
    )

    profile["risk_fusion_score"] = clamp(
        profile["security_risk_num"] * 0.18
        + profile["avg_vulnerability_risk_score"] * 0.24
        + profile["support_pressure_score"] * 0.15
        + profile["regulatory_exposure_num"] * 0.13
        + profile["technical_debt_num"] * 0.14
        + (100 - profile["avg_control_strength"]) * 0.12
        + np.where(profile["internet_exposed"], 8, 0),
        0,
        100,
    )

    profile["vendor_coercion_score"] = clamp(
        profile["oem_lifecycle_pressure_score"] * 0.34
        + profile["avg_vendor_pressure_score"] * 0.32
        + profile["support_pressure_score"] * 0.16
        + profile["avg_market_pressure_score"] * 0.10
        + np.log1p(profile["vendor_claim_count"]) * 6,
        0,
        100,
    )

    profile["lifecycle_pressure_score"] = clamp(
        profile["support_pressure_score"] * 0.36
        + profile["oem_lifecycle_pressure_score"] * 0.32
        + profile["technical_debt_num"] * 0.16
        + profile["vendor_coercion_score"] * 0.16,
        0,
        100,
    )

    profile["sustainability_score"] = clamp(
        100
        - profile["risk_fusion_score"] * 0.36
        - profile["lifecycle_pressure_score"] * 0.26
        - profile["vendor_coercion_score"] * 0.16
        - profile["technical_debt_num"] * 0.10
        + profile["avg_control_strength"] * 0.12,
        0,
        100,
    )

    profile["modernisation_urgency_score"] = clamp(
        profile["risk_fusion_score"] * 0.29
        + profile["lifecycle_pressure_score"] * 0.24
        + profile["technical_debt_num"] * 0.18
        + profile["regulatory_exposure_num"] * 0.12
        + profile["blast_radius_score"] * 0.11
        + profile["vendor_coercion_score"] * 0.06,
        0,
        100,
    )

    profile["mna_liability_score"] = clamp(
        profile["risk_fusion_score"] * 0.22
        + profile["blast_radius_score"] * 0.22
        + profile["vendor_coercion_score"] * 0.18
        + profile["regulatory_exposure_num"] * 0.16
        + (100 - profile["cmdb_confidence"] * 100) * 0.14
        + profile["technical_debt_num"] * 0.08,
        0,
        100,
    )

    profile["decision_confidence_score"] = clamp(
        profile["cmdb_confidence"] * 42
        + profile["avg_outbound_relationship_confidence"] * 28
        + np.where(profile["cost_line_count"] > 0, 8, 0)
        + np.where(profile["regulatory_mapping_count"] > 0, 8, 0)
        + np.where(profile["control_count"] > 0, 7, 0)
        + np.where(profile["entitlement_count"] > 0, 7, 0),
        0,
        100,
    )

    if "best_scenario" not in profile.columns:
        profile["best_scenario"] = ""

    fallback = profile.apply(fallback_decision, axis=1)
    profile["recommended_decision"] = profile["best_scenario"].fillna("").astype(str)
    profile["recommended_decision"] = np.where(
        profile["recommended_decision"].str.len() == 0,
        fallback,
        profile["recommended_decision"],
    )

    profile["risk_bucket"] = profile["risk_fusion_score"].apply(score_bucket)
    profile["vendor_pressure_bucket"] = profile["vendor_coercion_score"].apply(score_bucket)
    profile["blast_radius_bucket"] = profile["blast_radius_score"].apply(score_bucket)
    profile["modernisation_bucket"] = profile["modernisation_urgency_score"].apply(score_bucket)

    return profile


# =============================================================================
# Filtering UI
# =============================================================================


def portfolio_filter_ui(profile: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if profile.empty:
        return profile

    with st.expander("Portfolio filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            vendors = st.multiselect(
                "Vendor",
                unique_sorted(profile["vendor"]),
                default=[],
                key=f"{key_prefix}_vendor",
                help="Leave empty for all vendors.",
            )

        with c2:
            business_units = st.multiselect(
                "Business unit",
                unique_sorted(profile["business_unit"]),
                default=[],
                key=f"{key_prefix}_bu",
                help="Leave empty for all business units.",
            )

        with c3:
            criticalities = st.multiselect(
                "Criticality",
                unique_sorted(profile["criticality"]),
                default=[],
                key=f"{key_prefix}_crit",
                help="Leave empty for all criticalities.",
            )

        with c4:
            decisions = st.multiselect(
                "Recommended decision",
                unique_sorted(profile["recommended_decision"]),
                default=[],
                key=f"{key_prefix}_decision",
                help="Leave empty for all recommendations.",
            )

        c5, c6, c7, c8 = st.columns(4)

        with c5:
            support = st.multiselect(
                "Support status",
                unique_sorted(profile["support_status"]),
                default=[],
                key=f"{key_prefix}_support",
                help="Leave empty for all support states.",
            )

        with c6:
            env = st.multiselect(
                "Environment",
                unique_sorted(profile["environment"]),
                default=[],
                key=f"{key_prefix}_env",
                help="Leave empty for all environments.",
            )

        with c7:
            min_risk = st.slider(
                "Minimum risk score",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key=f"{key_prefix}_minrisk",
            )

        with c8:
            min_vendor_pressure = st.slider(
                "Minimum vendor pressure",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key=f"{key_prefix}_minvendor",
            )

    out = profile.copy()

    if vendors:
        out = out[out["vendor"].astype(str).isin(vendors)]

    if business_units:
        out = out[out["business_unit"].astype(str).isin(business_units)]

    if criticalities:
        out = out[out["criticality"].astype(str).isin(criticalities)]

    if decisions:
        out = out[out["recommended_decision"].astype(str).isin(decisions)]

    if support:
        out = out[out["support_status"].astype(str).isin(support)]

    if env:
        out = out[out["environment"].astype(str).isin(env)]

    out = out[out["risk_fusion_score"] >= min_risk]
    out = out[out["vendor_coercion_score"] >= min_vendor_pressure]

    return out


# =============================================================================
# Executive / board narrative helpers
# =============================================================================


def portfolio_posture(profile: pd.DataFrame) -> str:
    if profile.empty:
        return "No portfolio scope selected."

    avg_risk = profile["risk_fusion_score"].mean()
    avg_vendor = profile["vendor_coercion_score"].mean()
    avg_blast = profile["blast_radius_score"].mean()
    unsupported = profile["support_status"].astype(str).isin(["Unsupported", "Extended Support"]).mean() * 100

    if avg_vendor >= 70 and avg_blast >= 65:
        return (
            "The estate is under material vendor pressure and has a high dependency blast radius. "
            "A blanket OEM-led upgrade path is unlikely to be economically optimal. The recommended posture is "
            "portfolio segmentation: extend stable workloads, remediate genuine risk, and modernise only where evidence supports it."
        )

    if avg_risk >= 70:
        return (
            "The estate has elevated security and compliance exposure. Lifecycle action is required, but the decision should separate "
            "true risk reduction from vendor-driven change."
        )

    if unsupported >= 35:
        return (
            "A significant portion of the estate is outside normal vendor support. This does not automatically mean upgrade is required, "
            "but it does require evidence, compensating controls, and a defensible sustain-or-change decision path."
        )

    return (
        "The estate is suitable for controlled lifecycle optimisation. The strongest value is likely in reducing vendor leverage, "
        "removing avoidable cost, and prioritising selective modernisation over broad transformation."
    )


def system_narrative(row: pd.Series) -> str:
    decision = row.get("recommended_decision", "Review")
    name = row.get("system_name", row.get("system_id", "Selected system"))
    vendor = row.get("vendor", "Unknown")
    product = row.get("product", "Unknown")

    risk = float(row.get("risk_fusion_score", 0))
    vendor_pressure = float(row.get("vendor_coercion_score", 0))
    blast = float(row.get("blast_radius_score", 0))
    sustainability = float(row.get("sustainability_score", 0))
    cost = float(row.get("annual_run_cost_eur", 0))

    if decision == "Extend Independent Support":
        rationale = (
            "The evidence indicates that vendor pressure is higher than the immediate technical requirement for full change. "
            "Independent support with targeted remediation preserves operational stability while reducing forced-migration economics."
        )
    elif decision == "Modernise / Upgrade":
        rationale = (
            "The evidence indicates that the system has sufficient risk, support, or technical-debt pressure to justify controlled modernisation."
        )
    elif decision == "Replace / Migrate":
        rationale = (
            "The system appears to carry structural debt and lifecycle pressure that may not be solved efficiently by incremental upgrade."
        )
    elif decision == "Exit / Decommission":
        rationale = (
            "The system appears to be a candidate for retirement or consolidation, subject to business-owner and dependency validation."
        )
    elif decision == "Hybrid Transformation":
        rationale = (
            "The system requires a mixed path: sustain stable components, remediate exposed areas, and sequence transformation around dependency risk."
        )
    else:
        rationale = (
            "The system appears sustainable under current conditions, provided monitoring and evidence remain current."
        )

    return f"""
### {name}

**Technology:** {vendor} / {product}  
**Recommended pathway:** {decision}  
**Annual run cost:** {fmt_money(cost)}  
**Risk score:** {risk:.1f} / 100  
**Vendor pressure:** {vendor_pressure:.1f} / 100  
**Blast radius:** {blast:.1f} / 100  
**Sustainability:** {sustainability:.1f} / 100  

{rationale}
""".strip()


def build_board_pack(
    scope_name: str,
    profile: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> str:
    if profile.empty:
        return "# ESLI Board Decision Pack\n\nNo systems selected."

    systems = profile.copy()
    system_ids = as_set(systems["system_id"])

    total_systems = len(systems)
    annual_cost = systems["annual_run_cost_eur"].sum()
    best_5y_cost = systems["best_5y_cost_eur"].sum()
    avg_risk = systems["risk_fusion_score"].mean()
    avg_vendor = systems["vendor_coercion_score"].mean()
    avg_blast = systems["blast_radius_score"].mean()
    avg_confidence = systems["decision_confidence_score"].mean()

    unsupported_count = systems["support_status"].astype(str).isin(["Unsupported", "Extended Support"]).sum()
    high_risk_count = (systems["risk_fusion_score"] >= 70).sum()
    high_vendor_count = (systems["vendor_coercion_score"] >= 70).sum()
    high_blast_count = (systems["blast_radius_score"] >= 70).sum()

    decision_counts = systems["recommended_decision"].value_counts().reindex(SCENARIO_ORDER).dropna()
    top_risk = systems.sort_values("modernisation_urgency_score", ascending=False).head(10)

    claims = filter_by_system_ids(data.get("vendor_claims", empty_df()), system_ids)
    claims_count = len(claims)
    challengeable = 0
    phantom = 0

    if not claims.empty and "classification" in claims.columns:
        challengeable = claims["classification"].astype(str).isin(CHALLENGEABLE_CLAIMS).sum()
        phantom = claims["classification"].astype(str).eq("Phantom Upgrade").sum()

    actions = filter_by_system_ids(data.get("execution_actions", empty_df()), system_ids)
    proposed_actions = len(actions)

    decision_lines = []
    for decision, count in decision_counts.items():
        pct = count / max(total_systems, 1) * 100
        decision_lines.append(f"- **{decision}:** {count:,.0f} systems ({pct:.1f}%)")

    top_lines = []
    for _, row in top_risk.iterrows():
        top_lines.append(
            f"- **{row['system_name']}** ({row['vendor']} / {row['product']}): "
            f"{row['recommended_decision']} | urgency {row['modernisation_urgency_score']:.1f}, "
            f"risk {row['risk_fusion_score']:.1f}, blast radius {row['blast_radius_score']:.1f}"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""
# ESLI Board Decision Pack

**Scope:** {scope_name}  
**Generated:** {generated}

## 1. Executive decision question

Which software systems should be retained, extended under independent support, modernised, replaced, or decommissioned — and what is the defensible economic and risk basis for that decision?

## 2. Portfolio summary

- **Systems in scope:** {total_systems:,.0f}
- **Current annual run cost:** {fmt_money(annual_cost)}
- **Modelled 5-year recommended-path cost:** {fmt_money(best_5y_cost)}
- **Average risk score:** {avg_risk:.1f} / 100
- **Average vendor pressure score:** {avg_vendor:.1f} / 100
- **Average blast-radius score:** {avg_blast:.1f} / 100
- **Average decision confidence:** {avg_confidence:.1f} / 100
- **Unsupported / extended-support systems:** {unsupported_count:,.0f}
- **High-risk systems:** {high_risk_count:,.0f}
- **High vendor-pressure systems:** {high_vendor_count:,.0f}
- **High blast-radius systems:** {high_blast_count:,.0f}

## 3. Recommended portfolio posture

{portfolio_posture(systems)}

## 4. Decision segmentation

{chr(10).join(decision_lines)}

## 5. OEM Truth Layer findings

- **Vendor claims analysed:** {claims_count:,.0f}
- **Challengeable claims:** {challengeable:,.0f}
- **Phantom upgrade claims:** {phantom:,.0f}

Interpretation: vendor pressure should be treated as an input to the decision model, not the decision itself. Claims requiring upgrade should be checked against entitlement, exposure, controls, and real dependency impact.

## 6. Highest-priority systems

{chr(10).join(top_lines)}

## 7. Execution implications

- **Generated execution actions:** {proposed_actions:,.0f}
- Execution should be sequenced around dependency blast radius, entitlement validation, security controls, and business-service criticality.
- The preferred path is not a single estate-wide recommendation. It is a segmented decision portfolio.

## 8. Key assumptions

- Synthetic estate data represents realistic enterprise software, dependency, cost, licensing, risk, and vendor-pressure patterns.
- Unsupported status is not treated as automatic non-compliance; exposure, compensating controls, business criticality, and regulatory mapping are evaluated together.
- Upgrade cost includes hidden blast-radius economics: regression testing, integration rework, operational disruption, training, downtime risk, and parallel-run overhead.
- Origina is not always the recommended path. The model supports retain, extend, modernise, replace, exit, and hybrid decisions.

## 9. Recommended next actions

1. Validate high-pressure vendor claims against entitlement and usage evidence.
2. Run blast-radius assessment on the top dependency-heavy systems.
3. Identify systems that can be safely sustained under independent support with compensating controls.
4. Separate genuinely required upgrades from vendor-narrative upgrades.
5. Convert the prioritised recommendations into execution roadmaps.
""".strip()


# =============================================================================
# Chart helpers
# =============================================================================


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta=delta)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty:
        show_empty()
        return

    fig = px.bar(df, x=x, y=y, color=color, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty:
        show_empty()
        return

    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    size: str | None = None,
    hover_name: str | None = None,
) -> None:
    if df.empty:
        show_empty()
        return

    plot_df = df.copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        title=title,
        opacity=0.75,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Blast radius engine
# =============================================================================


def build_adjacency(
    deps: pd.DataFrame,
    direction: str,
) -> dict[str, list[tuple[str, str, str, str]]]:
    adjacency: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    if deps.empty or not {"source_system_id", "target_system_id"}.issubset(deps.columns):
        return adjacency

    needed = ["source_system_id", "target_system_id"]
    for optional in ["dependency_id", "dependency_type", "criticality"]:
        if optional not in deps.columns:
            deps[optional] = ""

    for row in deps[["source_system_id", "target_system_id", "dependency_id", "dependency_type", "criticality"]].itertuples(
        index=False
    ):
        src = str(row.source_system_id)
        dst = str(row.target_system_id)
        dep_id = str(row.dependency_id)
        dep_type = str(row.dependency_type)
        crit = str(row.criticality)

        if direction in {"Downstream", "Both"}:
            adjacency[src].append((dst, dep_id, dep_type, crit))

        if direction in {"Upstream", "Both"}:
            adjacency[dst].append((src, dep_id, dep_type, crit))

    return adjacency


def compute_blast_radius(
    deps: pd.DataFrame,
    start_system_id: str,
    depth: int = 2,
    direction: str = "Downstream",
    max_edges: int = 25000,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    adjacency = build_adjacency(deps.copy(), direction)

    visited_depth: dict[str, int] = {start_system_id: 0}
    q: deque[tuple[str, int]] = deque([(start_system_id, 0)])
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    while q and len(edges) < max_edges:
        node, d = q.popleft()
        if d >= depth:
            continue

        for nxt, dep_id, dep_type, crit in adjacency.get(node, []):
            edge_key = (node, nxt, dep_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source_system_id": node,
                        "target_system_id": nxt,
                        "dependency_id": dep_id,
                        "dependency_type": dep_type,
                        "criticality": crit,
                        "depth": str(d + 1),
                    }
                )

            if nxt not in visited_depth:
                visited_depth[nxt] = d + 1
                q.append((nxt, d + 1))

    return visited_depth, edges


def plot_blast_graph(
    nodes_depth: dict[str, int],
    edge_rows: list[dict[str, str]],
    profile: pd.DataFrame,
    start_system_id: str,
    max_nodes: int = 160,
) -> None:
    if not nodes_depth:
        show_empty("No graph nodes found.")
        return

    profile_idx = profile.set_index("system_id")

    ordered_nodes = sorted(nodes_depth.keys(), key=lambda n: (nodes_depth[n], n))
    selected_nodes = ordered_nodes[:max_nodes]

    if start_system_id not in selected_nodes:
        selected_nodes = [start_system_id] + selected_nodes[: max_nodes - 1]

    selected_set = set(selected_nodes)

    G = nx.Graph()

    for node in selected_nodes:
        label = node
        if node in profile_idx.index:
            row = profile_idx.loc[node]
            label = f"{row.get('system_name', node)}<br>{row.get('vendor', '')} / {row.get('product', '')}"
        G.add_node(node, depth=nodes_depth.get(node, 0), label=label)

    for edge in edge_rows:
        src = edge["source_system_id"]
        dst = edge["target_system_id"]
        if src in selected_set and dst in selected_set:
            G.add_edge(src, dst, dep_type=edge.get("dependency_type", ""), criticality=edge.get("criticality", ""))

    if G.number_of_nodes() == 0:
        show_empty("No graph nodes to render.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.7)

    edge_x = []
    edge_y = []

    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6),
        hoverinfo="none",
        mode="lines",
        name="Dependency",
    )

    node_x = []
    node_y = []
    node_text = []
    node_depths = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_depths.append(nodes_depth.get(node, 0))

        if node in profile_idx.index:
            row = profile_idx.loc[node]
            dep_count = float(row.get("downstream_dependency_count", 0)) + float(row.get("upstream_dependency_count", 0))
            node_sizes.append(12 + min(28, math.log1p(dep_count) * 5))
            node_text.append(
                f"<b>{row.get('system_name', node)}</b><br>"
                f"ID: {node}<br>"
                f"{row.get('vendor', '')} / {row.get('product', '')}<br>"
                f"Decision: {row.get('recommended_decision', '')}<br>"
                f"Risk: {float(row.get('risk_fusion_score', 0)):.1f}<br>"
                f"Blast radius: {float(row.get('blast_radius_score', 0)):.1f}"
            )
        else:
            node_sizes.append(12)
            node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_depths,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Depth"),
            line=dict(width=1),
        ),
        name="Systems",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Blast-radius dependency graph | rendered nodes: {len(selected_nodes):,.0f}",
        height=650,
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Pages
# =============================================================================


def page_executive_overview(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("ESLI Decision Twin")
    st.caption("Synthetic enterprise-scale lifecycle intelligence demo")

    if profile.empty:
        show_empty("No systems loaded. Check the data folder path in the sidebar.")
        return

    scoped = portfolio_filter_ui(profile, "exec")
    if scoped.empty:
        show_empty("No systems match the selected filters.")
        return

    total_systems = len(scoped)
    annual_cost = scoped["annual_run_cost_eur"].sum()
    modelled_cost = scoped["best_5y_cost_eur"].sum()
    avg_risk = scoped["risk_fusion_score"].mean()
    avg_vendor = scoped["vendor_coercion_score"].mean()
    high_blast = (scoped["blast_radius_score"] >= 70).sum()
    unsupported = scoped["support_status"].astype(str).isin(["Unsupported", "Extended Support"]).sum()

    metric_row(
        [
            ("Systems in scope", fmt_num(total_systems), None),
            ("Annual run cost", fmt_money(annual_cost), None),
            ("5-year recommended cost", fmt_money(modelled_cost), None),
            ("Average risk", f"{avg_risk:.1f}/100", None),
            ("Vendor pressure", f"{avg_vendor:.1f}/100", None),
            ("High blast-radius systems", fmt_num(high_blast), f"{unsupported:,.0f} unsupported/extended"),
        ]
    )

    st.markdown("### Portfolio posture")
    st.write(portfolio_posture(scoped))

    c1, c2 = st.columns([1, 1])

    with c1:
        decision_df = (
            scoped["recommended_decision"]
            .value_counts()
            .reindex(SCENARIO_ORDER)
            .dropna()
            .reset_index()
        )
        decision_df.columns = ["decision", "systems"]
        bar_chart(decision_df, "decision", "systems", "Recommended lifecycle decisions", color="decision")

    with c2:
        vendor_df = (
            scoped.groupby("vendor")
            .agg(
                systems=("system_id", "count"),
                annual_cost_eur=("annual_run_cost_eur", "sum"),
                avg_vendor_pressure=("vendor_coercion_score", "mean"),
                avg_risk=("risk_fusion_score", "mean"),
            )
            .reset_index()
            .sort_values("annual_cost_eur", ascending=False)
            .head(20)
        )
        fig = px.scatter(
            vendor_df,
            x="avg_vendor_pressure",
            y="avg_risk",
            size="annual_cost_eur",
            hover_name="vendor",
            title="Vendor pressure vs risk by vendor",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    scenarios = data.get("scenario_results", empty_df())
    if not scenarios.empty:
        system_ids = as_set(scoped["system_id"])
        scn = scenarios[scenarios["system_id"].astype(str).isin(system_ids)].copy()

        if not scn.empty:
            scn["projection_year"] = safe_numeric(scn["projection_year"])
            scn["projected_cost_eur"] = safe_numeric(scn["projected_cost_eur"])
            scn["risk_score"] = safe_numeric(scn["risk_score"])

            agg = (
                scn.groupby(["scenario", "projection_year"])
                .agg(projected_cost_eur=("projected_cost_eur", "sum"), avg_risk=("risk_score", "mean"))
                .reset_index()
            )

            c3, c4 = st.columns([1, 1])

            with c3:
                line_chart(agg, "projection_year", "projected_cost_eur", "5-year cost envelope by pathway", color="scenario")

            with c4:
                line_chart(agg, "projection_year", "avg_risk", "Projected risk movement by pathway", color="scenario")

    st.markdown("### Highest-priority systems")
    top_cols = [
        "system_id",
        "system_name",
        "business_unit",
        "vendor",
        "product",
        "criticality",
        "support_status",
        "annual_run_cost_eur",
        "risk_fusion_score",
        "vendor_coercion_score",
        "blast_radius_score",
        "modernisation_urgency_score",
        "recommended_decision",
        "decision_confidence_score",
    ]

    top = scoped.sort_values("modernisation_urgency_score", ascending=False).head(50)
    st.dataframe(top[columns_present(top, top_cols)], use_container_width=True, height=500)
    download_df_button(top[columns_present(top, top_cols)], "esli_top_priority_systems.csv", "Download top-priority systems")


def page_data_fabric(data: dict[str, pd.DataFrame], profile: pd.DataFrame, data_dir: str) -> None:
    st.title("Data & Signal Fabric")
    st.caption("Synthetic internal estate data, external signals, CMDB quality, entity resolution, and normalisation health.")

    health = file_health(data_dir)
    loaded_rows = []

    for key, filename in REQUIRED_FILES.items():
        df = data.get(key, empty_df())
        loaded_rows.append({"table": key, "file": filename, "rows_loaded": len(df)})

    loaded = pd.DataFrame(loaded_rows)
    health = health.merge(loaded, on=["table", "file"], how="left")

    metric_row(
        [
            ("Files present", fmt_num(health["exists"].sum()), None),
            ("Tables loaded", fmt_num((health["rows_loaded"] > 0).sum()), None),
            ("Total rows loaded", fmt_num(health["rows_loaded"].sum()), None),
            ("Data size", f"{health['size_mb'].sum():,.1f} MB", None),
        ]
    )

    st.markdown("### File manifest")
    st.dataframe(health, use_container_width=True, height=460)

    manifest = data.get("manifest", empty_df())
    if not manifest.empty:
        st.markdown("### Generator manifest")
        st.dataframe(manifest, use_container_width=True, height=320)

    if profile.empty:
        return

    st.markdown("### Normalisation and data quality summary")

    duplicate_count = profile["duplicate_group_id"].fillna("").astype(str).str.len().gt(0).sum()
    low_cmdb = (profile["cmdb_confidence"] < 0.5).sum()
    low_rel = (profile["avg_outbound_relationship_confidence"] < 0.5).sum()
    missing_owner = profile["owner_team"].fillna("").astype(str).isin(["", "Unknown"]).sum()

    metric_row(
        [
            ("Resolved systems", fmt_num(len(profile)), None),
            ("Potential duplicates", fmt_num(duplicate_count), None),
            ("Low CMDB confidence", fmt_num(low_cmdb), None),
            ("Low relationship confidence", fmt_num(low_rel), None),
            ("Missing/unknown owners", fmt_num(missing_owner), None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = px.histogram(
            profile,
            x="cmdb_confidence",
            nbins=30,
            title="CMDB confidence distribution",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            profile,
            x="avg_outbound_relationship_confidence",
            nbins=30,
            title="Relationship confidence distribution",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Data fabric by source type")
    source_counts = (
        data.get("systems", empty_df())
        .get("source_system", pd.Series(dtype=str))
        .value_counts()
        .reset_index()
    )

    if not source_counts.empty:
        source_counts.columns = ["source_system", "records"]
        bar_chart(source_counts, "source_system", "records", "System records by source system")


def page_estate_explorer(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Estate Explorer")
    st.caption("Search the synthetic estate, inspect system-level intelligence, and download scoped system views.")

    if profile.empty:
        show_empty()
        return

    scoped = portfolio_filter_ui(profile, "estate")

    search = st.text_input(
        "Search system, product, service, owner, vendor, or business unit",
        value="",
        placeholder="Example: WebSphere, SAP, Payments, unsupported, finance",
    ).strip().lower()

    if search:
        search_cols = [
            "system_id",
            "system_name",
            "vendor",
            "product",
            "business_service_name",
            "business_unit",
            "owner_team",
            "support_status",
        ]
        mask = pd.Series(False, index=scoped.index)

        for col in columns_present(scoped, search_cols):
            mask = mask | scoped[col].astype(str).str.lower().str.contains(search, na=False)

        scoped = scoped[mask]

    if scoped.empty:
        show_empty("No systems match the current search and filters.")
        return

    metric_row(
        [
            ("Systems", fmt_num(len(scoped)), None),
            ("Annual run cost", fmt_money(scoped["annual_run_cost_eur"].sum()), None),
            ("Average risk", f"{scoped['risk_fusion_score'].mean():.1f}/100", None),
            ("Average vendor pressure", f"{scoped['vendor_coercion_score'].mean():.1f}/100", None),
            ("Average blast radius", f"{scoped['blast_radius_score'].mean():.1f}/100", None),
        ]
    )

    display_cols = [
        "system_id",
        "system_name",
        "asset_type",
        "business_unit",
        "business_service_name",
        "vendor",
        "product",
        "version",
        "environment",
        "hosting_model",
        "criticality",
        "support_status",
        "annual_run_cost_eur",
        "risk_fusion_score",
        "vendor_coercion_score",
        "blast_radius_score",
        "sustainability_score",
        "modernisation_urgency_score",
        "recommended_decision",
        "decision_confidence_score",
    ]

    st.dataframe(
        scoped[columns_present(scoped, display_cols)].sort_values("modernisation_urgency_score", ascending=False),
        use_container_width=True,
        height=520,
    )

    download_df_button(
        scoped[columns_present(scoped, display_cols)],
        "esli_estate_scope.csv",
        "Download scoped estate",
    )

    st.markdown("### System drill-down")

    option_df = scoped.sort_values("modernisation_urgency_score", ascending=False).head(5000).copy()
    option_df["select_label"] = (
        option_df["system_name"].astype(str)
        + " | "
        + option_df["vendor"].astype(str)
        + " / "
        + option_df["product"].astype(str)
        + " | "
        + option_df["system_id"].astype(str)
    )

    selected_label = st.selectbox("Select a system", option_df["select_label"].tolist())
    selected_system_id = selected_label.split("|")[-1].strip()

    row = profile[profile["system_id"].astype(str).eq(selected_system_id)].iloc[0]

    st.markdown(system_narrative(row))

    tabs = st.tabs(
        [
            "System facts",
            "Applications",
            "Dependencies",
            "Vulnerabilities",
            "Controls",
            "Licensing",
            "Vendor claims",
            "Regulatory",
            "Scenarios",
            "Actions",
        ]
    )

    with tabs[0]:
        fact_cols = [
            "system_id",
            "system_name",
            "asset_type",
            "business_service_name",
            "business_unit",
            "vendor",
            "product",
            "version",
            "environment",
            "hosting_model",
            "region",
            "owner_team",
            "criticality",
            "internet_exposed",
            "data_sensitivity",
            "lifecycle_stage",
            "support_status",
            "security_risk",
            "technical_debt",
            "regulatory_exposure",
            "annual_run_cost_eur",
            "cmdb_confidence",
            "recommended_decision",
        ]
        st.dataframe(pd.DataFrame(row[columns_present(pd.DataFrame([row]), fact_cols)]).T, use_container_width=True)

    with tabs[1]:
        apps = filter_by_system_ids(data.get("applications", empty_df()), {selected_system_id})
        st.dataframe(apps, use_container_width=True, height=360)

    with tabs[2]:
        deps = data.get("dependencies", empty_df())
        if not deps.empty:
            rel = deps[
                deps["source_system_id"].astype(str).eq(selected_system_id)
                | deps["target_system_id"].astype(str).eq(selected_system_id)
            ].copy()
            st.dataframe(rel.head(1000), use_container_width=True, height=420)
        else:
            show_empty()

    with tabs[3]:
        st.dataframe(filter_by_system_ids(data.get("vulnerabilities", empty_df()), {selected_system_id}), use_container_width=True, height=420)

    with tabs[4]:
        st.dataframe(filter_by_system_ids(data.get("security_controls", empty_df()), {selected_system_id}), use_container_width=True, height=420)

    with tabs[5]:
        st.dataframe(filter_by_system_ids(data.get("licensing_entitlements", empty_df()), {selected_system_id}), use_container_width=True, height=420)

    with tabs[6]:
        st.dataframe(filter_by_system_ids(data.get("vendor_claims", empty_df()), {selected_system_id}), use_container_width=True, height=420)

    with tabs[7]:
        st.dataframe(filter_by_system_ids(data.get("regulatory_mappings", empty_df()), {selected_system_id}), use_container_width=True, height=420)

    with tabs[8]:
        scn = filter_by_system_ids(data.get("scenario_results", empty_df()), {selected_system_id})
        st.dataframe(scn, use_container_width=True, height=420)

    with tabs[9]:
        acts = filter_by_system_ids(data.get("execution_actions", empty_df()), {selected_system_id})
        st.dataframe(acts, use_container_width=True, height=420)


def page_blast_radius(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Dependency & Blast Radius")
    st.caption("Select a system and calculate direct and indirect estate impact.")

    deps = data.get("dependencies", empty_df())

    if profile.empty or deps.empty:
        show_empty("Systems or dependencies are missing.")
        return

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])

    with c1:
        default_scope = profile.sort_values("blast_radius_score", ascending=False).head(3000).copy()
        default_scope["select_label"] = (
            default_scope["system_name"].astype(str)
            + " | "
            + default_scope["vendor"].astype(str)
            + " / "
            + default_scope["product"].astype(str)
            + " | "
            + default_scope["system_id"].astype(str)
        )
        selected_label = st.selectbox("System to assess", default_scope["select_label"].tolist())
        selected_system_id = selected_label.split("|")[-1].strip()

    with c2:
        depth = st.slider("Traversal depth", 1, 4, 2)

    with c3:
        direction = st.selectbox("Direction", ["Downstream", "Upstream", "Both"], index=0)

    with c4:
        max_nodes = st.slider("Graph nodes", 40, 350, 160, step=20)

    nodes_depth, edge_rows = compute_blast_radius(
        deps=deps,
        start_system_id=selected_system_id,
        depth=depth,
        direction=direction,
    )

    affected_ids = set(nodes_depth.keys())
    affected_profile = profile[profile["system_id"].astype(str).isin(affected_ids)].copy()

    direct_count = sum(1 for d in nodes_depth.values() if d == 1)
    indirect_count = sum(1 for d in nodes_depth.values() if d > 1)
    affected_cost = affected_profile["annual_run_cost_eur"].sum()

    edge_df = pd.DataFrame(edge_rows)
    critical_edges = 0
    if not edge_df.empty and "criticality" in edge_df.columns:
        critical_edges = edge_df["criticality"].astype(str).isin(["High", "Critical"]).sum()

    hidden_cost = (
        affected_cost * 0.18
        + len(edge_rows) * 32_500
        + critical_edges * 48_000
        + affected_profile["criticality"].astype(str).isin(["High", "Critical"]).sum() * 85_000
    )

    regression_packs = int(max(1, critical_edges * 1.4 + direct_count * 2 + indirect_count * 0.35))
    validation_interfaces = int(max(1, len(edge_rows) * 0.55))

    change_window_risk = score_bucket(
        min(100, len(edge_rows) * 0.18 + critical_edges * 0.7 + affected_profile["criticality_num"].mean() * 0.5)
    )

    metric_row(
        [
            ("Systems affected", fmt_num(len(affected_ids)), None),
            ("Direct dependencies", fmt_num(direct_count), None),
            ("Indirect dependencies", fmt_num(indirect_count), None),
            ("Critical integrations", fmt_num(critical_edges), None),
            ("Affected annual cost", fmt_money(affected_cost), None),
            ("Hidden blast-radius cost", fmt_money(hidden_cost), f"{change_window_risk} change risk"),
        ]
    )

    st.markdown(
        """
### Blast Radius Economics

The cost is not the upgrade. The cost is everything the upgrade forces: integration rework, regression testing,
data validation, outage risk, business disruption, retraining, parallel running, and delayed delivery.
"""
    )

    metric_row(
        [
            ("Regression packs required", fmt_num(regression_packs), None),
            ("Interfaces requiring validation", fmt_num(validation_interfaces), None),
            ("Estimated change risk", change_window_risk, None),
            ("Average affected risk", f"{affected_profile['risk_fusion_score'].mean():.1f}/100", None),
            ("Average decision confidence", f"{affected_profile['decision_confidence_score'].mean():.1f}/100", None),
        ]
    )

    plot_blast_graph(
        nodes_depth=nodes_depth,
        edge_rows=edge_rows,
        profile=profile,
        start_system_id=selected_system_id,
        max_nodes=max_nodes,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Affected systems")
        cols = [
            "system_id",
            "system_name",
            "vendor",
            "product",
            "criticality",
            "support_status",
            "annual_run_cost_eur",
            "risk_fusion_score",
            "blast_radius_score",
            "recommended_decision",
        ]
        st.dataframe(
            affected_profile[columns_present(affected_profile, cols)].sort_values("risk_fusion_score", ascending=False),
            use_container_width=True,
            height=420,
        )

    with c2:
        st.markdown("### Traversed dependencies")
        st.dataframe(edge_df.head(1000), use_container_width=True, height=420)


def page_intelligence_layer(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Intelligence Layer")
    st.caption("Cost, security, licensing, OEM, regulatory, market, and fused decision intelligence.")

    if profile.empty:
        show_empty()
        return

    scoped = portfolio_filter_ui(profile, "intel")
    system_ids = as_set(scoped["system_id"])

    tabs = st.tabs(
        [
            "Cost Intelligence",
            "Risk & Security",
            "Licensing & Entitlement",
            "Regulatory",
            "Market & Technology",
            "Fusion Layer",
        ]
    )

    with tabs[0]:
        cost = filter_by_system_ids(data.get("cost_lines", empty_df()), system_ids)

        metric_row(
            [
                ("Annual run cost", fmt_money(scoped["annual_run_cost_eur"].sum()), None),
                ("Cost lines", fmt_num(len(cost)), None),
                ("Cost-line total", fmt_money(cost["amount_eur"].pipe(safe_numeric).sum()) if not cost.empty and "amount_eur" in cost else "€0", None),
                ("Systems with cost evidence", fmt_num((scoped["cost_line_count"] > 0).sum()), None),
            ]
        )

        if not cost.empty:
            cost["amount_eur"] = safe_numeric(cost["amount_eur"])
            by_type = (
                cost.groupby("cost_type")
                .agg(amount_eur=("amount_eur", "sum"), lines=("amount_eur", "count"))
                .reset_index()
                .sort_values("amount_eur", ascending=False)
            )
            bar_chart(by_type, "cost_type", "amount_eur", "Cost intelligence by cost type")

            if "fiscal_year" in cost.columns:
                by_year = (
                    cost.groupby(["fiscal_year", "cost_type"])
                    .agg(amount_eur=("amount_eur", "sum"))
                    .reset_index()
                )
                line_chart(by_year, "fiscal_year", "amount_eur", "Cost movement by fiscal year", color="cost_type")

            st.markdown("### Highest TCO+ systems")
            cols = [
                "system_id",
                "system_name",
                "vendor",
                "product",
                "annual_run_cost_eur",
                "total_cost_lines_eur",
                "blast_radius_score",
                "risk_fusion_score",
                "recommended_decision",
            ]
            st.dataframe(
                scoped[columns_present(scoped, cols)].sort_values("total_cost_lines_eur", ascending=False).head(100),
                use_container_width=True,
                height=420,
            )
        else:
            show_empty("No cost lines for this scope.")

    with tabs[1]:
        vulns = filter_by_system_ids(data.get("vulnerabilities", empty_df()), system_ids)
        controls = filter_by_system_ids(data.get("security_controls", empty_df()), system_ids)

        metric_row(
            [
                ("Vulnerabilities", fmt_num(len(vulns)), None),
                ("Critical vulnerabilities", fmt_num(vulns.get("severity", pd.Series(dtype=str)).astype(str).eq("Critical").sum()), None),
                ("Exploitable", fmt_num(boolish(vulns.get("exploit_available", pd.Series(dtype=bool))).sum()) if not vulns.empty else "0", None),
                ("Controls", fmt_num(len(controls)), None),
                ("Avg control strength", f"{controls['control_strength'].pipe(safe_numeric).mean():.1f}/100" if not controls.empty and "control_strength" in controls else "0/100", None),
            ]
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            if not vulns.empty:
                sev = vulns["severity"].astype(str).value_counts().reset_index()
                sev.columns = ["severity", "count"]
                bar_chart(sev, "severity", "count", "Vulnerabilities by severity")
            else:
                show_empty("No vulnerabilities for this scope.")

        with c2:
            if not controls.empty:
                ctrl = controls["control_family"].astype(str).value_counts().head(20).reset_index()
                ctrl.columns = ["control_family", "count"]
                bar_chart(ctrl, "control_family", "count", "Controls by family")
            else:
                show_empty("No controls for this scope.")

        scatter_chart(
            scoped,
            "avg_control_strength",
            "risk_fusion_score",
            "Control strength vs risk",
            color="criticality",
            size="annual_run_cost_eur",
            hover_name="system_name",
        )

        st.markdown("### Highest unresolved exposure")
        cols = [
            "system_id",
            "system_name",
            "vendor",
            "product",
            "criticality",
            "internet_exposed",
            "support_status",
            "vulnerability_count",
            "critical_vulnerability_count",
            "exploitable_vulnerability_count",
            "avg_control_strength",
            "risk_fusion_score",
            "recommended_decision",
        ]
        st.dataframe(
            scoped[columns_present(scoped, cols)].sort_values("risk_fusion_score", ascending=False).head(100),
            use_container_width=True,
            height=420,
        )

    with tabs[2]:
        ent = filter_by_system_ids(data.get("licensing_entitlements", empty_df()), system_ids)

        metric_row(
            [
                ("Entitlement records", fmt_num(len(ent)), None),
                ("Annual maintenance", fmt_money(ent["annual_maintenance_eur"].pipe(safe_numeric).sum()) if not ent.empty and "annual_maintenance_eur" in ent else "€0", None),
                ("Shelfware units", fmt_num(ent["shelfware_units"].pipe(safe_numeric).sum()) if not ent.empty and "shelfware_units" in ent else "0", None),
                ("Over-deployed units", fmt_num(ent["over_deployed_units"].pipe(safe_numeric).sum()) if not ent.empty and "over_deployed_units" in ent else "0", None),
                ("High audit-risk entitlements", fmt_num(ent.get("audit_risk", pd.Series(dtype=str)).astype(str).eq("High").sum()) if not ent.empty else "0", None),
            ]
        )

        if not ent.empty:
            by_vendor = (
                ent.assign(
                    annual_maintenance_eur=safe_numeric(ent.get("annual_maintenance_eur", pd.Series(0, index=ent.index))),
                    over_deployed_units=safe_numeric(ent.get("over_deployed_units", pd.Series(0, index=ent.index))),
                    shelfware_units=safe_numeric(ent.get("shelfware_units", pd.Series(0, index=ent.index))),
                )
                .groupby("vendor")
                .agg(
                    records=("system_id", "count"),
                    annual_maintenance_eur=("annual_maintenance_eur", "sum"),
                    shelfware_units=("shelfware_units", "sum"),
                    over_deployed_units=("over_deployed_units", "sum"),
                )
                .reset_index()
                .sort_values("annual_maintenance_eur", ascending=False)
                .head(25)
            )
            bar_chart(by_vendor, "vendor", "annual_maintenance_eur", "Annual maintenance by vendor")

            audit = ent.get("audit_risk", pd.Series(dtype=str)).astype(str).value_counts().reset_index()
            audit.columns = ["audit_risk", "count"]
            bar_chart(audit, "audit_risk", "count", "Audit-risk distribution")

            st.markdown("### Highest licensing exposure systems")
            cols = [
                "system_id",
                "system_name",
                "vendor",
                "product",
                "annual_maintenance_eur",
                "shelfware_units",
                "over_deployed_units",
                "high_audit_risk_entitlements",
                "vendor_coercion_score",
                "recommended_decision",
            ]
            st.dataframe(
                scoped[columns_present(scoped, cols)].sort_values(
                    ["over_deployed_units", "high_audit_risk_entitlements"],
                    ascending=False,
                ).head(100),
                use_container_width=True,
                height=420,
            )
        else:
            show_empty("No entitlement records for this scope.")

    with tabs[3]:
        regs = filter_by_system_ids(data.get("regulatory_mappings", empty_df()), system_ids)

        metric_row(
            [
                ("Regulatory mappings", fmt_num(len(regs)), None),
                ("Weak audit defensibility", fmt_num(regs.get("audit_defensibility", pd.Series(dtype=str)).astype(str).eq("Weak").sum()) if not regs.empty else "0", None),
                ("Avg compliance exposure", f"{regs['compliance_exposure_score'].pipe(safe_numeric).mean():.1f}/100" if not regs.empty and "compliance_exposure_score" in regs else "0/100", None),
                ("Mapped systems", fmt_num(regs["system_id"].nunique()) if not regs.empty else "0", None),
            ]
        )

        if not regs.empty:
            by_reg = (
                regs.assign(compliance_exposure_score=safe_numeric(regs["compliance_exposure_score"]))
                .groupby("regulation")
                .agg(
                    mappings=("system_id", "count"),
                    systems=("system_id", "nunique"),
                    avg_exposure=("compliance_exposure_score", "mean"),
                )
                .reset_index()
                .sort_values("mappings", ascending=False)
            )
            bar_chart(by_reg, "regulation", "mappings", "Regulatory mappings by framework")

            defn = regs["audit_defensibility"].astype(str).value_counts().reset_index()
            defn.columns = ["audit_defensibility", "count"]
            bar_chart(defn, "audit_defensibility", "count", "Audit defensibility")

            st.markdown("### Highest compliance exposure")
            cols = [
                "system_id",
                "system_name",
                "vendor",
                "product",
                "criticality",
                "regulatory_mapping_count",
                "max_compliance_exposure_score",
                "weak_audit_defensibility_count",
                "recommended_decision",
            ]
            st.dataframe(
                scoped[columns_present(scoped, cols)].sort_values("max_compliance_exposure_score", ascending=False).head(100),
                use_container_width=True,
                height=420,
            )
        else:
            show_empty("No regulatory mappings for this scope.")

    with tabs[4]:
        vendors = as_set(scoped["vendor"])
        products = as_set(scoped["product"])
        market = filter_by_vendor_product(data.get("market_signals", empty_df()), vendors, products)

        metric_row(
            [
                ("Market signals", fmt_num(len(market)), None),
                ("Estimated market impact", fmt_money(market["estimated_cost_impact_eur"].pipe(safe_numeric).sum()) if not market.empty and "estimated_cost_impact_eur" in market else "€0", None),
                ("Mapped vendors", fmt_num(market["vendor"].nunique()) if not market.empty and "vendor" in market else "0", None),
                ("Mapped products", fmt_num(market["product"].nunique()) if not market.empty and "product" in market else "0", None),
            ]
        )

        if not market.empty:
            theme = (
                market.assign(estimated_cost_impact_eur=safe_numeric(market["estimated_cost_impact_eur"]))
                .groupby("signal_type")
                .agg(signals=("signal_id", "count"), impact=("estimated_cost_impact_eur", "sum"))
                .reset_index()
                .sort_values("impact", ascending=False)
            )
            bar_chart(theme, "signal_type", "impact", "Forward-looking market impact")

            st.markdown("### Signals requiring action")
            action_cols = [
                "signal_id",
                "signal_date",
                "signal_type",
                "vendor",
                "product",
                "severity",
                "likelihood",
                "time_horizon_months",
                "mapped_system_count",
                "estimated_cost_impact_eur",
                "recommended_watch_action",
                "confidence",
            ]
            st.dataframe(
                market[columns_present(market, action_cols)].sort_values(
                    "estimated_cost_impact_eur",
                    ascending=False,
                ).head(200),
                use_container_width=True,
                height=500,
            )
        else:
            show_empty("No market signals for this scope.")

    with tabs[5]:
        metric_row(
            [
                ("Avg lifecycle pressure", f"{scoped['lifecycle_pressure_score'].mean():.1f}/100", None),
                ("Avg vendor coercion", f"{scoped['vendor_coercion_score'].mean():.1f}/100", None),
                ("Avg sustainability", f"{scoped['sustainability_score'].mean():.1f}/100", None),
                ("Avg modernisation urgency", f"{scoped['modernisation_urgency_score'].mean():.1f}/100", None),
                ("Avg decision confidence", f"{scoped['decision_confidence_score'].mean():.1f}/100", None),
            ]
        )

        scatter_chart(
            scoped,
            "lifecycle_pressure_score",
            "blast_radius_score",
            "Lifecycle pressure vs blast radius",
            color="recommended_decision",
            size="annual_run_cost_eur",
            hover_name="system_name",
        )

        st.markdown("### Fused decision intelligence")
        cols = [
            "system_id",
            "system_name",
            "vendor",
            "product",
            "criticality",
            "support_status",
            "risk_fusion_score",
            "vendor_coercion_score",
            "lifecycle_pressure_score",
            "blast_radius_score",
            "sustainability_score",
            "modernisation_urgency_score",
            "mna_liability_score",
            "decision_confidence_score",
            "recommended_decision",
        ]
        st.dataframe(
            scoped[columns_present(scoped, cols)].sort_values("modernisation_urgency_score", ascending=False).head(300),
            use_container_width=True,
            height=520,
        )


def page_oem_truth(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("OEM Truth Layer")
    st.caption("Separate vendor narrative from evidence-backed estate decisions.")

    if profile.empty:
        show_empty()
        return

    scoped = portfolio_filter_ui(profile, "oem")
    system_ids = as_set(scoped["system_id"])
    vendors = as_set(scoped["vendor"])
    products = as_set(scoped["product"])

    claims = filter_by_system_ids(data.get("vendor_claims", empty_df()), system_ids)
    lifecycle = filter_by_vendor_product(data.get("oem_lifecycle", empty_df()), vendors, products)

    challengeable = 0
    phantom = 0
    commercial = 0
    pressure = 0
    impact = 0

    if not claims.empty:
        claims["vendor_pressure_score"] = safe_numeric(claims.get("vendor_pressure_score", pd.Series(0, index=claims.index)))
        claims["commercial_impact_eur"] = safe_numeric(claims.get("commercial_impact_eur", pd.Series(0, index=claims.index)))
        challengeable = claims.get("classification", pd.Series("", index=claims.index)).astype(str).isin(CHALLENGEABLE_CLAIMS).sum()
        phantom = claims.get("classification", pd.Series("", index=claims.index)).astype(str).eq("Phantom Upgrade").sum()
        commercial = claims.get("classification", pd.Series("", index=claims.index)).astype(str).eq("Commercial Pressure").sum()
        pressure = claims["vendor_pressure_score"].mean()
        impact = claims["commercial_impact_eur"].sum()

    metric_row(
        [
            ("Vendor claims", fmt_num(len(claims)), None),
            ("Challengeable claims", fmt_num(challengeable), None),
            ("Phantom upgrades", fmt_num(phantom), None),
            ("Commercial pressure claims", fmt_num(commercial), None),
            ("Avg claim pressure", f"{pressure:.1f}/100", None),
            ("Claimed commercial impact", fmt_money(impact), None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if not claims.empty:
            class_df = claims["classification"].astype(str).value_counts().reset_index()
            class_df.columns = ["classification", "claims"]
            bar_chart(class_df, "classification", "claims", "Vendor claims by ESLI classification")
        else:
            show_empty("No claims for this scope.")

    with c2:
        if not claims.empty:
            type_df = claims["claim_type"].astype(str).value_counts().reset_index()
            type_df.columns = ["claim_type", "claims"]
            bar_chart(type_df, "claim_type", "claims", "Vendor claims by claim type")
        else:
            show_empty("No claims for this scope.")

    st.markdown("### Claim challenge register")
    if not claims.empty:
        claim_cols = [
            "claim_id",
            "system_id",
            "vendor",
            "product",
            "claim_date",
            "claim_type",
            "claim_text",
            "vendor_pressure_score",
            "classification",
            "independent_assessment",
            "commercial_impact_eur",
            "recommended_response",
            "confidence",
        ]
        st.dataframe(
            claims[columns_present(claims, claim_cols)].sort_values("vendor_pressure_score", ascending=False).head(500),
            use_container_width=True,
            height=560,
        )
    else:
        show_empty()

    st.markdown("### OEM lifecycle and roadmap pressure")
    if not lifecycle.empty:
        if "lifecycle_pressure_score" in lifecycle.columns:
            lifecycle["lifecycle_pressure_score"] = safe_numeric(lifecycle["lifecycle_pressure_score"])

        lifecycle_summary = (
            lifecycle.groupby(["vendor", "product"])
            .agg(
                notices=("vendor", "count"),
                avg_pressure=("lifecycle_pressure_score", "mean"),
                max_pressure=("lifecycle_pressure_score", "max"),
                affected_system_count=("affected_system_count", "sum") if "affected_system_count" in lifecycle else ("vendor", "count"),
            )
            .reset_index()
            .sort_values("avg_pressure", ascending=False)
            .head(40)
        )
        st.dataframe(lifecycle_summary, use_container_width=True, height=420)

        bar_chart(lifecycle_summary, "product", "avg_pressure", "Lifecycle pressure by product", color="vendor")
    else:
        show_empty("No lifecycle records for this scope.")


def page_forward_radar(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Forward-Looking Radar")
    st.caption("Model external pressures before they force lifecycle decisions.")

    if profile.empty:
        show_empty()
        return

    scoped = portfolio_filter_ui(profile, "radar")
    system_ids = as_set(scoped["system_id"])
    vendors = as_set(scoped["vendor"])
    products = as_set(scoped["product"])

    signals = filter_by_vendor_product(data.get("market_signals", empty_df()), vendors, products)

    if not signals.empty:
        for col in ["estimated_cost_impact_eur", "severity", "likelihood", "time_horizon_months"]:
            if col in signals.columns:
                signals[col] = safe_numeric(signals[col])

    metric_row(
        [
            ("Systems in radar scope", fmt_num(len(scoped)), None),
            ("Market signals", fmt_num(len(signals)), None),
            ("Signal cost impact", fmt_money(signals["estimated_cost_impact_eur"].sum()) if not signals.empty and "estimated_cost_impact_eur" in signals else "€0", None),
            ("High vendor pressure systems", fmt_num((scoped["vendor_coercion_score"] >= 70).sum()), None),
            ("High regulatory exposure systems", fmt_num((scoped["regulatory_exposure_num"] >= 70).sum()), None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if not signals.empty:
            by_type = (
                signals.groupby("signal_type")
                .agg(
                    signals=("signal_id", "count"),
                    impact=("estimated_cost_impact_eur", "sum"),
                    avg_severity=("severity", "mean"),
                    avg_likelihood=("likelihood", "mean"),
                )
                .reset_index()
                .sort_values("impact", ascending=False)
            )
            bar_chart(by_type, "signal_type", "impact", "Market and technology radar impact")
        else:
            show_empty("No market signals for this scope.")

    with c2:
        scatter_chart(
            scoped,
            "vendor_coercion_score",
            "modernisation_urgency_score",
            "Vendor pressure vs modernisation urgency",
            color="vendor",
            size="annual_run_cost_eur",
            hover_name="system_name",
        )

    st.markdown("### Future threat and pressure simulator")

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        shock = st.selectbox(
            "Scenario",
            [
                "Vendor pricing shock",
                "Regulatory tightening",
                "Quantum crypto exposure",
                "AI-native rebuild pressure",
                "Open-source substitution window",
                "Critical CVE cluster",
                "Vendor ecosystem decline",
            ],
        )

    with c2:
        selected_vendor = st.selectbox("Vendor focus", ["All"] + unique_sorted(scoped["vendor"]))

    with c3:
        intensity = st.slider("Shock intensity", 5, 75, 35, step=5)

    scenario_scope = scoped.copy()
    if selected_vendor != "All":
        scenario_scope = scenario_scope[scenario_scope["vendor"].astype(str).eq(selected_vendor)]

    if shock == "Vendor pricing shock":
        affected = scenario_scope[scenario_scope["vendor_coercion_score"] >= 45].copy()
        affected["shock_cost_eur"] = affected["annual_run_cost_eur"] * (intensity / 100) * 3
        recommendation = "Prepare negotiation position, challenge vendor timeline, and model independent support alternatives."

    elif shock == "Regulatory tightening":
        affected = scenario_scope[
            (scenario_scope["regulatory_exposure_num"] >= 50)
            | (scenario_scope["data_sensitivity_num"] >= 65)
            | (scenario_scope["criticality_num"] >= 70)
        ].copy()
        affected["shock_cost_eur"] = affected["annual_run_cost_eur"] * (0.08 + intensity / 300)
        recommendation = "Prioritise evidence quality, compensating controls, and defensible sustain-or-change decisions."

    elif shock == "Quantum crypto exposure":
        affected = scenario_scope[
            scenario_scope["product_family"].astype(str).isin(["Database", "Middleware", "Operating System", "Integration", "Network"])
            | scenario_scope["internet_exposed"].astype(bool)
        ].copy()
        affected["shock_cost_eur"] = 125_000 + affected["annual_run_cost_eur"] * (intensity / 500)
        recommendation = "Create a cryptographic inventory and sequence remediation across externally exposed and regulated systems."

    elif shock == "AI-native rebuild pressure":
        affected = scenario_scope[
            scenario_scope["asset_type"].astype(str).isin(["Application", "Custom Application", "ERP Module"])
            & (scenario_scope["technical_debt_num"] >= 50)
        ].copy()
        affected["shock_cost_eur"] = affected["annual_run_cost_eur"] * (0.15 + intensity / 250)
        recommendation = "Classify rebuild candidates by supportability, governance risk, data sensitivity, and dependency complexity."

    elif shock == "Open-source substitution window":
        affected = scenario_scope[
            scenario_scope["vendor"].astype(str).isin(["IBM", "Oracle", "SAP", "VMware", "Broadcom"])
            & (scenario_scope["blast_radius_score"] < 70)
            & (scenario_scope["risk_fusion_score"] < 75)
        ].copy()
        affected["shock_cost_eur"] = -affected["annual_run_cost_eur"] * (0.08 + intensity / 500)
        recommendation = "Assess replacement candidates where dependency blast radius is manageable and operating model maturity is sufficient."

    elif shock == "Critical CVE cluster":
        affected = scenario_scope[
            (scenario_scope["internet_exposed"].astype(bool))
            | (scenario_scope["security_risk_num"] >= 70)
            | (scenario_scope["support_pressure_score"] >= 70)
        ].copy()
        affected["shock_cost_eur"] = affected["annual_run_cost_eur"] * (0.10 + intensity / 220)
        recommendation = "Prioritise externally exposed, unsupported, and weak-control systems for immediate compensating controls."

    else:
        affected = scenario_scope[
            (scenario_scope["market_signal_count"] > 0)
            | (scenario_scope["support_pressure_score"] >= 60)
            | (scenario_scope["vendor_coercion_score"] >= 60)
        ].copy()
        affected["shock_cost_eur"] = affected["annual_run_cost_eur"] * (0.10 + intensity / 300)
        recommendation = "Reduce ecosystem dependency through segmented support, partner optionality, and controlled modernisation."

    metric_row(
        [
            ("Affected systems", fmt_num(len(affected)), None),
            ("Affected annual cost", fmt_money(affected["annual_run_cost_eur"].sum()), None),
            ("Scenario cost impact", fmt_money(affected["shock_cost_eur"].sum()), None),
            ("Avg affected risk", f"{affected['risk_fusion_score'].mean():.1f}/100" if not affected.empty else "0/100", None),
            ("Avg affected blast radius", f"{affected['blast_radius_score'].mean():.1f}/100" if not affected.empty else "0/100", None),
        ]
    )

    st.markdown(f"**Recommended response:** {recommendation}")

    cols = [
        "system_id",
        "system_name",
        "vendor",
        "product",
        "criticality",
        "support_status",
        "annual_run_cost_eur",
        "shock_cost_eur",
        "risk_fusion_score",
        "vendor_coercion_score",
        "blast_radius_score",
        "recommended_decision",
    ]

    st.dataframe(
        affected[columns_present(affected, cols)].sort_values("shock_cost_eur", ascending=False).head(300),
        use_container_width=True,
        height=520,
    )


def scenario_custom_score(y5: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    df = y5.copy()

    for col in [
        "projected_cost_eur",
        "risk_score",
        "disruption_score",
        "vendor_leverage_score",
        "regulatory_defensibility",
        "strategic_fit_score",
        "execution_complexity",
    ]:
        if col not in df.columns:
            df[col] = 50
        df[col] = safe_numeric(df[col])

    def inverse_minmax(s: pd.Series) -> pd.Series:
        mn = s.min()
        mx = s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(50, index=s.index)
        return 100 - ((s - mn) / (mx - mn) * 100)

    def direct_minmax(s: pd.Series) -> pd.Series:
        mn = s.min()
        mx = s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(50, index=s.index)
        return ((s - mn) / (mx - mn) * 100)

    df["cost_score_custom"] = inverse_minmax(df["projected_cost_eur"])
    df["risk_score_custom"] = 100 - df["risk_score"]
    df["disruption_score_custom"] = 100 - df["disruption_score"]
    df["vendor_score_custom"] = df["vendor_leverage_score"]
    df["regulatory_score_custom"] = df["regulatory_defensibility"]
    df["fit_score_custom"] = df["strategic_fit_score"]
    df["complexity_score_custom"] = 100 - df["execution_complexity"]

    total_w = sum(weights.values()) or 1

    df["custom_decision_score"] = (
        df["cost_score_custom"] * weights["cost"]
        + df["risk_score_custom"] * weights["risk"]
        + df["disruption_score_custom"] * weights["disruption"]
        + df["vendor_score_custom"] * weights["vendor_leverage"]
        + df["regulatory_score_custom"] * weights["regulatory"]
        + df["fit_score_custom"] * weights["strategic_fit"]
        + df["complexity_score_custom"] * weights["complexity"]
    ) / total_w

    return df.sort_values("custom_decision_score", ascending=False)


def page_digital_twin(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Digital Twin Simulator")
    st.caption("Simulate retain, extend, modernise, replace, exit, and hybrid pathways over five years.")

    scenarios = data.get("scenario_results", empty_df())

    if profile.empty or scenarios.empty:
        show_empty("Systems or scenario results are missing.")
        return

    scoped = portfolio_filter_ui(profile, "twin")
    system_ids = as_set(scoped["system_id"])

    scn = filter_by_system_ids(scenarios, system_ids)

    if scn.empty:
        show_empty("No scenario results for this scope.")
        return

    for col in [
        "projection_year",
        "projected_cost_eur",
        "risk_score",
        "disruption_score",
        "vendor_leverage_score",
        "regulatory_defensibility",
        "strategic_fit_score",
        "execution_complexity",
        "recommendation_score",
    ]:
        if col in scn.columns:
            scn[col] = safe_numeric(scn[col])

    agg = (
        scn.groupby(["scenario", "projection_year"])
        .agg(
            projected_cost_eur=("projected_cost_eur", "sum"),
            avg_risk=("risk_score", "mean"),
            avg_disruption=("disruption_score", "mean"),
            avg_regulatory_defensibility=("regulatory_defensibility", "mean"),
            avg_recommendation=("recommendation_score", "mean"),
        )
        .reset_index()
    )

    max_year = int(scn["projection_year"].max())
    y5 = scn[scn["projection_year"] == max_year].copy()

    y5_summary = (
        y5.groupby("scenario")
        .agg(
            five_year_cost_proxy=("projected_cost_eur", "sum"),
            avg_risk=("risk_score", "mean"),
            avg_disruption=("disruption_score", "mean"),
            avg_regulatory_defensibility=("regulatory_defensibility", "mean"),
            avg_recommendation=("recommendation_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_recommendation", ascending=False)
    )

    metric_row(
        [
            ("Systems simulated", fmt_num(scoped["system_id"].nunique()), None),
            ("Scenario rows", fmt_num(len(scn)), None),
            ("Pathways", fmt_num(scn["scenario"].nunique()), None),
            ("Projection horizon", f"{max_year} years", None),
            ("Best pathway", y5_summary.iloc[0]["scenario"] if not y5_summary.empty else "N/A", None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        line_chart(agg, "projection_year", "projected_cost_eur", "Projected cost envelope", color="scenario")

    with c2:
        line_chart(agg, "projection_year", "avg_risk", "Projected average risk", color="scenario")

    st.markdown("### Pathway comparison at horizon")
    st.dataframe(y5_summary, use_container_width=True, height=360)

    st.markdown("### Decision sensitivity model")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        w_cost = st.slider("Cost weight", 0, 100, 25)
        w_risk = st.slider("Risk weight", 0, 100, 25)

    with c2:
        w_disruption = st.slider("Disruption weight", 0, 100, 15)
        w_vendor = st.slider("Vendor leverage weight", 0, 100, 15)

    with c3:
        w_reg = st.slider("Regulatory defensibility weight", 0, 100, 10)
        w_fit = st.slider("Strategic fit weight", 0, 100, 10)

    with c4:
        w_complexity = st.slider("Execution complexity weight", 0, 100, 10)

    weighted = scenario_custom_score(
        y5_summary.rename(
            columns={
                "five_year_cost_proxy": "projected_cost_eur",
                "avg_risk": "risk_score",
                "avg_disruption": "disruption_score",
                "avg_regulatory_defensibility": "regulatory_defensibility",
                "avg_recommendation": "recommendation_score",
            }
        ),
        {
            "cost": w_cost,
            "risk": w_risk,
            "disruption": w_disruption,
            "vendor_leverage": w_vendor,
            "regulatory": w_reg,
            "strategic_fit": w_fit,
            "complexity": w_complexity,
        },
    )

    st.dataframe(weighted, use_container_width=True, height=360)

    st.markdown("### System-level pathway drill-down")

    option_df = scoped.sort_values("modernisation_urgency_score", ascending=False).head(3000).copy()
    option_df["select_label"] = (
        option_df["system_name"].astype(str)
        + " | "
        + option_df["vendor"].astype(str)
        + " / "
        + option_df["product"].astype(str)
        + " | "
        + option_df["system_id"].astype(str)
    )

    selected_label = st.selectbox("Select system", option_df["select_label"].tolist(), key="twin_system")
    selected_system_id = selected_label.split("|")[-1].strip()

    sys_scn = scn[scn["system_id"].astype(str).eq(selected_system_id)].copy()

    c3, c4 = st.columns([1, 1])

    with c3:
        line_chart(sys_scn, "projection_year", "projected_cost_eur", "Selected-system cost pathway", color="scenario")

    with c4:
        line_chart(sys_scn, "projection_year", "risk_score", "Selected-system risk pathway", color="scenario")

    st.dataframe(sys_scn.sort_values(["scenario", "projection_year"]), use_container_width=True, height=420)


def page_decision_scoring(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Decision Scoring Engine")
    st.caption("Rank systems by lifecycle pressure, risk, blast radius, sustainability, and evidence confidence.")

    if profile.empty:
        show_empty()
        return

    scoped = portfolio_filter_ui(profile, "score")

    if scoped.empty:
        show_empty()
        return

    metric_row(
        [
            ("Systems scored", fmt_num(len(scoped)), None),
            ("Avg recommendation score", f"{scoped['best_recommendation_score'].mean():.1f}/100", None),
            ("Avg modernisation urgency", f"{scoped['modernisation_urgency_score'].mean():.1f}/100", None),
            ("Avg sustainability", f"{scoped['sustainability_score'].mean():.1f}/100", None),
            ("Avg confidence", f"{scoped['decision_confidence_score'].mean():.1f}/100", None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        dist = scoped["recommended_decision"].value_counts().reindex(SCENARIO_ORDER).dropna().reset_index()
        dist.columns = ["recommended_decision", "systems"]
        bar_chart(dist, "recommended_decision", "systems", "Recommended decision distribution", color="recommended_decision")

    with c2:
        bucket = (
            scoped.groupby(["risk_bucket", "vendor_pressure_bucket"])
            .agg(systems=("system_id", "count"), annual_cost=("annual_run_cost_eur", "sum"))
            .reset_index()
        )
        fig = px.density_heatmap(
            bucket,
            x="vendor_pressure_bucket",
            y="risk_bucket",
            z="systems",
            histfunc="sum",
            title="Risk vs vendor-pressure heatmap",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    scatter_chart(
        scoped,
        "sustainability_score",
        "modernisation_urgency_score",
        "Sustainability vs modernisation urgency",
        color="recommended_decision",
        size="annual_run_cost_eur",
        hover_name="system_name",
    )

    st.markdown("### Ranked decision register")

    cols = [
        "system_id",
        "system_name",
        "business_unit",
        "business_service_name",
        "vendor",
        "product",
        "version",
        "criticality",
        "support_status",
        "annual_run_cost_eur",
        "risk_fusion_score",
        "vendor_coercion_score",
        "lifecycle_pressure_score",
        "blast_radius_score",
        "sustainability_score",
        "modernisation_urgency_score",
        "mna_liability_score",
        "decision_confidence_score",
        "recommended_decision",
    ]

    ranked = scoped[columns_present(scoped, cols)].sort_values(
        ["modernisation_urgency_score", "annual_run_cost_eur"],
        ascending=False,
    )

    st.dataframe(ranked.head(1000), use_container_width=True, height=560)
    download_df_button(ranked, "esli_decision_register.csv", "Download decision register")

    st.markdown("### Recommendation narrative")

    option_df = scoped.sort_values("modernisation_urgency_score", ascending=False).head(3000).copy()
    option_df["select_label"] = (
        option_df["system_name"].astype(str)
        + " | "
        + option_df["vendor"].astype(str)
        + " / "
        + option_df["product"].astype(str)
        + " | "
        + option_df["system_id"].astype(str)
    )

    selected_label = st.selectbox("Select system for narrative", option_df["select_label"].tolist(), key="score_system")
    selected_system_id = selected_label.split("|")[-1].strip()
    row = profile[profile["system_id"].astype(str).eq(selected_system_id)].iloc[0]

    st.markdown(system_narrative(row))


def page_execution_layer(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Execution & Orchestration Layer")
    st.caption("Convert recommendations into action roadmaps, owners, service routes, and delivery sequencing.")

    if profile.empty:
        show_empty()
        return

    actions = data.get("execution_actions", empty_df())

    if actions.empty:
        show_empty("Execution actions are missing.")
        return

    scoped = portfolio_filter_ui(profile, "exec_layer")
    system_ids = as_set(scoped["system_id"])

    acts = filter_by_system_ids(actions, system_ids)

    if acts.empty:
        show_empty("No execution actions for this scope.")
        return

    for col in ["start_month", "duration_months", "estimated_cost_eur", "expected_risk_reduction"]:
        if col in acts.columns:
            acts[col] = safe_numeric(acts[col])

    metric_row(
        [
            ("Actions", fmt_num(len(acts)), None),
            ("Systems with actions", fmt_num(acts["system_id"].nunique()), None),
            ("Estimated action cost", fmt_money(acts["estimated_cost_eur"].sum()) if "estimated_cost_eur" in acts else "€0", None),
            ("Avg expected risk reduction", f"{acts['expected_risk_reduction'].mean():.1f}" if "expected_risk_reduction" in acts else "0", None),
            ("Completed actions", fmt_num(acts.get("status", pd.Series(dtype=str)).astype(str).eq("Completed").sum()), None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        route = acts["service_route"].astype(str).value_counts().reset_index()
        route.columns = ["service_route", "actions"]
        bar_chart(route, "service_route", "actions", "Execution actions by service route")

    with c2:
        status = acts["status"].astype(str).value_counts().reset_index()
        status.columns = ["status", "actions"]
        bar_chart(status, "status", "actions", "Execution action status")

    st.markdown("### Execution roadmap")

    timeline = acts.copy().head(400)
    if {"start_month", "duration_months"}.issubset(timeline.columns):
        base = pd.to_datetime("2026-01-01")
        timeline["start_date"] = base + pd.to_timedelta(timeline["start_month"] * 30, unit="D")
        timeline["end_date"] = timeline["start_date"] + pd.to_timedelta(timeline["duration_months"] * 30, unit="D")
        timeline["display_action"] = timeline["action_title"].astype(str) + " | " + timeline["system_id"].astype(str)

        fig = px.timeline(
            timeline,
            x_start="start_date",
            x_end="end_date",
            y="display_action",
            color="service_route",
            hover_data=["recommended_scenario", "owner_group", "status", "estimated_cost_eur"],
            title="Execution sequence sample",
        )
        fig.update_yaxes(visible=False)
        fig.update_layout(height=650, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Action register")
    cols = [
        "action_id",
        "system_id",
        "recommended_scenario",
        "action_sequence",
        "action_type",
        "action_title",
        "owner_group",
        "service_route",
        "start_month",
        "duration_months",
        "estimated_cost_eur",
        "expected_risk_reduction",
        "status",
    ]

    st.dataframe(
        acts[columns_present(acts, cols)].sort_values(["system_id", "action_sequence"]).head(2000),
        use_container_width=True,
        height=560,
    )
    download_df_button(acts[columns_present(acts, cols)], "esli_execution_actions.csv", "Download execution actions")


def page_feedback_loop(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Continuous Feedback Loop")
    st.caption("Compare expected recommendations with observed outcomes and identify where the model learns.")

    outcomes = data.get("decision_outcomes", empty_df())

    if profile.empty or outcomes.empty:
        show_empty("Profile or outcome records are missing.")
        return

    scoped = portfolio_filter_ui(profile, "feedback")
    system_ids = as_set(scoped["system_id"])

    outs = filter_by_system_ids(outcomes, system_ids)

    if outs.empty:
        show_empty("No outcomes for this scope.")
        return

    for col in ["expected_5y_cost_eur", "actual_5y_cost_proxy_eur", "expected_risk_reduction", "observed_risk_reduction", "cost_variance_pct", "delay_months"]:
        if col in outs.columns:
            outs[col] = safe_numeric(outs[col])

    good = outs.get("outcome_quality", pd.Series(dtype=str)).astype(str).isin(["Good", "Strong"]).mean() * 100

    metric_row(
        [
            ("Outcome records", fmt_num(len(outs)), None),
            ("Good/strong outcomes", fmt_pct(good), None),
            ("Avg cost variance", f"{outs['cost_variance_pct'].mean():.1f}%" if "cost_variance_pct" in outs else "0%", None),
            ("Avg risk reduction", f"{outs['observed_risk_reduction'].mean():.1f}" if "observed_risk_reduction" in outs else "0", None),
            ("Avg delay", f"{outs['delay_months'].mean():.1f} months" if "delay_months" in outs else "0 months", None),
        ]
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        quality = outs["outcome_quality"].astype(str).value_counts().reset_index()
        quality.columns = ["outcome_quality", "records"]
        bar_chart(quality, "outcome_quality", "records", "Outcome quality")

    with c2:
        if "decision_taken" in outs.columns:
            by_decision = (
                outs.groupby("decision_taken")
                .agg(
                    records=("outcome_id", "count"),
                    avg_cost_variance=("cost_variance_pct", "mean"),
                    avg_risk_reduction=("observed_risk_reduction", "mean"),
                    avg_delay=("delay_months", "mean"),
                )
                .reset_index()
                .sort_values("records", ascending=False)
            )
            bar_chart(by_decision, "decision_taken", "avg_cost_variance", "Average cost variance by decision")

    if "lessons_learned" in outs.columns:
        st.markdown("### Most common lessons learned")
        lessons = outs["lessons_learned"].astype(str).value_counts().head(20).reset_index()
        lessons.columns = ["lesson", "records"]
        st.dataframe(lessons, use_container_width=True, height=360)

    st.markdown("### Outcome register")
    st.dataframe(outs.sort_values("cost_variance_pct", ascending=False).head(1500), use_container_width=True, height=560)
    download_df_button(outs, "esli_decision_outcomes.csv", "Download outcome records")


def page_board_pack(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> None:
    st.title("Board Decision Pack")
    st.caption("Generate a CIO/CFO-ready decision pack from the selected portfolio scope.")

    if profile.empty:
        show_empty()
        return

    scope_mode = st.selectbox(
        "Scope type",
        ["Filtered portfolio", "Vendor", "Product", "Business unit", "Business service", "Single system"],
    )

    scoped = profile.copy()
    scope_name = "Filtered portfolio"

    if scope_mode == "Filtered portfolio":
        scoped = portfolio_filter_ui(profile, "board")
        scope_name = "Filtered portfolio"

    elif scope_mode == "Vendor":
        vendor = st.selectbox("Vendor", unique_sorted(profile["vendor"]))
        scoped = profile[profile["vendor"].astype(str).eq(vendor)].copy()
        scope_name = f"Vendor: {vendor}"

    elif scope_mode == "Product":
        product = st.selectbox("Product", unique_sorted(profile["product"]))
        scoped = profile[profile["product"].astype(str).eq(product)].copy()
        scope_name = f"Product: {product}"

    elif scope_mode == "Business unit":
        bu = st.selectbox("Business unit", unique_sorted(profile["business_unit"]))
        scoped = profile[profile["business_unit"].astype(str).eq(bu)].copy()
        scope_name = f"Business unit: {bu}"

    elif scope_mode == "Business service":
        svc = st.selectbox("Business service", unique_sorted(profile["business_service_name"]))
        scoped = profile[profile["business_service_name"].astype(str).eq(svc)].copy()
        scope_name = f"Business service: {svc}"

    else:
        option_df = profile.sort_values("modernisation_urgency_score", ascending=False).head(5000).copy()
        option_df["select_label"] = (
            option_df["system_name"].astype(str)
            + " | "
            + option_df["vendor"].astype(str)
            + " / "
            + option_df["product"].astype(str)
            + " | "
            + option_df["system_id"].astype(str)
        )
        selected_label = st.selectbox("System", option_df["select_label"].tolist())
        selected_system_id = selected_label.split("|")[-1].strip()
        scoped = profile[profile["system_id"].astype(str).eq(selected_system_id)].copy()
        scope_name = f"System: {selected_label}"

    if scoped.empty:
        show_empty("No systems in selected board-pack scope.")
        return

    pack = build_board_pack(scope_name, scoped, data)

    metric_row(
        [
            ("Systems", fmt_num(len(scoped)), None),
            ("Annual run cost", fmt_money(scoped["annual_run_cost_eur"].sum()), None),
            ("5-year recommended cost", fmt_money(scoped["best_5y_cost_eur"].sum()), None),
            ("Average risk", f"{scoped['risk_fusion_score'].mean():.1f}/100", None),
            ("Average vendor pressure", f"{scoped['vendor_coercion_score'].mean():.1f}/100", None),
        ]
    )

    st.markdown(pack)

    st.download_button(
        "Download board pack as Markdown",
        data=pack.encode("utf-8"),
        file_name="esli_board_decision_pack.md",
        mime="text/markdown",
        use_container_width=True,
    )

    csv = scoped.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download board-pack system scope",
        data=csv,
        file_name="esli_board_pack_scope.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# App shell
# =============================================================================


def render_sidebar(data: dict[str, pd.DataFrame], profile: pd.DataFrame, data_dir: str) -> str:
    st.sidebar.title("ESLI Decision Twin")

    st.sidebar.markdown(
        """
Synthetic enterprise-scale lifecycle intelligence POC.

This app expects the generated CSV files in a local `data/` folder.
"""
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Data & Signal Fabric",
            "Estate Explorer",
            "Dependency & Blast Radius",
            "Intelligence Layer",
            "OEM Truth Layer",
            "Forward-Looking Radar",
            "Digital Twin Simulator",
            "Decision Scoring",
            "Execution Layer",
            "Feedback Loop",
            "Board Pack",
        ],
    )

    st.sidebar.divider()

    health = file_health(data_dir)
    loaded_rows = sum(len(df) for df in data.values())

    st.sidebar.metric("Data folder", data_dir)
    st.sidebar.metric("Files present", f"{health['exists'].sum():,.0f}/{len(health)}")
    st.sidebar.metric("Rows loaded", fmt_num(loaded_rows))

    if not profile.empty:
        st.sidebar.metric("Systems profiled", fmt_num(len(profile)))
        st.sidebar.metric("Annual run cost", fmt_money(profile["annual_run_cost_eur"].sum()))
        st.sidebar.metric("Avg risk", f"{profile['risk_fusion_score'].mean():.1f}/100")

    missing = health[~health["exists"]]
    if not missing.empty:
        with st.sidebar.expander("Missing files"):
            st.dataframe(missing[["file"]], use_container_width=True, hide_index=True)

    st.sidebar.divider()
    st.sidebar.caption("POC scale: deterministic synthetic data, real calculations, no external APIs required.")

    return page


def main() -> None:
    st.sidebar.header("Data")
    data_dir = st.sidebar.text_input("Data directory", value="data")

    data = load_all_data(data_dir)
    profile = build_system_profile(data)

    page = render_sidebar(data, profile, data_dir)

    if page == "Executive Overview":
        page_executive_overview(data, profile)

    elif page == "Data & Signal Fabric":
        page_data_fabric(data, profile, data_dir)

    elif page == "Estate Explorer":
        page_estate_explorer(data, profile)

    elif page == "Dependency & Blast Radius":
        page_blast_radius(data, profile)

    elif page == "Intelligence Layer":
        page_intelligence_layer(data, profile)

    elif page == "OEM Truth Layer":
        page_oem_truth(data, profile)

    elif page == "Forward-Looking Radar":
        page_forward_radar(data, profile)

    elif page == "Digital Twin Simulator":
        page_digital_twin(data, profile)

    elif page == "Decision Scoring":
        page_decision_scoring(data, profile)

    elif page == "Execution Layer":
        page_execution_layer(data, profile)

    elif page == "Feedback Loop":
        page_feedback_loop(data, profile)

    elif page == "Board Pack":
        page_board_pack(data, profile)


if __name__ == "__main__":
    main()
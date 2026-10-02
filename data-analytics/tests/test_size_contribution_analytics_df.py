"""Unit tests for DataFrame-based size and contribution analytics."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lambda_functions.size_contribution_analytics import (
    get_analytics_df,
    get_contrib_analytics_df,
    get_date_validation_errors,
    get_size_analytics_df,
    prepare_filter_params,
    window,
)


@pytest.fixture
def sample_analytics_df():
    """Builds a merged mock DataFrame mimicking load_csv_data output."""
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame(
        {
            "org_id": [1, 2, 3, 4],
            "country_code": ["us", "us", "ca", "us"],
            "country_name": [
                "united states",
                "united states",
                "canada",
                "united states",
            ],
            "org_type": ["non_profit", "for_profit", "non_profit", "for_profit"],
            "org_size": ["small", "medium", "small", "large"],
            "is_collaborator": [True, True, False, False],
            "is_contributor": [True, False, True, False],
            "created_at": [
                today - pd.Timedelta(days=2),  # 7D
                today - pd.Timedelta(days=15),  # 30D
                today - pd.Timedelta(days=100),  # 1Y
                today - pd.Timedelta(days=500),  # All only
            ],
        }
    )


# ---------------------------------------------------------------------------
# UNIT TESTS: Filter Parameter Validation & Error Handling
# ---------------------------------------------------------------------------


def test_get_date_validation_errors_half_pairs():
    """Missing half of a date pair produces explicit error messages."""
    assert len(get_date_validation_errors("size", "2026-01-01", None)) == 1
    assert len(get_date_validation_errors("contribution", None, "2026-12-31")) == 1


def test_get_date_validation_errors_bad_format_and_order():
    """Bad date format and start > end are caught."""
    errs_fmt = get_date_validation_errors("size", "invalid-date", "2026-01-01")
    assert any("Invalid date format" in e for e in errs_fmt)

    errs_order = get_date_validation_errors("size", "2026-12-31", "2026-01-01")
    assert any("must be before" in e for e in errs_order)


def test_prepare_filter_params_defaults_and_normalization():
    """Checks defaults and case normalization."""
    payload = {"country": "USA", "organization_type": "NON_PROFIT"}
    validated = prepare_filter_params(payload)

    assert validated["country"] == "usa"
    assert validated["org_type"] == "non_profit"
    assert validated["size_start_date"] is None
    assert validated["contrib_start_date"] is None


def test_prepare_filter_params_invalid_org_type():
    """Invalid org_type raises ValueError with clear message."""
    with pytest.raises(ValueError, match="Invalid organization_type"):
        prepare_filter_params({"organization_type": "enterprise"})


# ---------------------------------------------------------------------------
# UNIT TESTS: get_size_analytics_df
# ---------------------------------------------------------------------------


def test_get_size_analytics_preserves_order_and_omits_missing():
    """Emits only categories present, strictly sorted small -> medium -> large."""
    df = pd.DataFrame(
        {
            "org_size": ["large", "small", "small"]  # 'medium' is missing
        }
    )
    result = get_size_analytics_df(df)

    assert result == [
        {"size": "small", "count": 2},
        {"size": "large", "count": 1},
    ]


def test_get_size_analytics_empty_dataframe():
    """Empty dataframe returns an empty list without crashing."""
    empty_df = pd.DataFrame(columns=["org_size"])
    assert get_size_analytics_df(empty_df) == []


# ---------------------------------------------------------------------------
# UNIT TESTS: get_contrib_analytics_df
# ---------------------------------------------------------------------------


def test_get_contrib_analytics_independent_counts_and_percentages():
    """Collaborator and Contributor counts are independent and do not sum to total."""
    df = pd.DataFrame(
        {
            "is_collaborator": [True, True, False, False],
            "is_contributor": [True, False, False, False],
        }
    )
    result = get_contrib_analytics_df(df)

    collab = next(r for r in result if r["type"] == "Collaborator")
    contrib = next(r for r in result if r["type"] == "Contributor")

    assert collab["count"] == 2
    assert collab["percentage"] == 50.0

    assert contrib["count"] == 1
    assert contrib["percentage"] == 25.0


def test_get_contrib_analytics_empty_df():
    """Empty df returns 0 counts and 0.0 percentages."""
    empty_df = pd.DataFrame(columns=["is_collaborator", "is_contributor"])
    result = get_contrib_analytics_df(empty_df)

    assert result == [
        {"type": "Collaborator", "count": 0, "percentage": 0.0},
        {"type": "Contributor", "count": 0, "percentage": 0.0},
    ]


# ---------------------------------------------------------------------------
# UNIT TESTS: window
# ---------------------------------------------------------------------------


def test_window_inclusive_filtering():
    """Window filters inclusive of the full end date day."""
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2026-05-01 00:00:00",
                    "2026-05-05 14:30:00",
                    "2026-05-05 23:59:59",
                    "2026-05-06 00:00:00",
                ]
            )
        }
    )
    filtered = window(df, "2026-05-01", "2026-05-05")
    assert len(filtered) == 3


# ---------------------------------------------------------------------------
# INTEGRATION / SHAPE TESTS: get_analytics_df
# ---------------------------------------------------------------------------


def test_get_analytics_no_custom_dates_returns_five_buckets(sample_analytics_df):
    """No custom date params returns exactly 5 keys, with empty Custom."""
    params = {
        "country": "all",
        "org_type": "all",
        "size_start_date": None,
        "size_end_date": None,
        "contrib_start_date": None,
        "contrib_end_date": None,
    }
    result = get_analytics_df(sample_analytics_df, params)

    assert set(result.keys()) == {"7D", "30D", "1Y", "All", "Custom"}
    assert result["Custom"]["organizations_by_size"] == []
    assert result["Custom"]["collaborator_vs_contributor"] == []


def test_get_analytics_only_size_custom(sample_analytics_df):
    """Only size range returns Custom-only, populates size, leaves contrib empty."""
    params = {
        "country": "all",
        "org_type": "all",
        "size_start_date": "2020-01-01",
        "size_end_date": "2030-01-01",
        "contrib_start_date": None,
        "contrib_end_date": None,
    }
    result = get_analytics_df(sample_analytics_df, params)

    assert list(result.keys()) == ["Custom"]
    assert len(result["Custom"]["organizations_by_size"]) > 0
    assert result["Custom"]["collaborator_vs_contributor"] == []


def test_get_analytics_only_contrib_custom(sample_analytics_df):
    """Only contrib range returns Custom-only, populates contrib, leaves size empty."""
    params = {
        "country": "all",
        "org_type": "all",
        "size_start_date": None,
        "size_end_date": None,
        "contrib_start_date": "2020-01-01",
        "contrib_end_date": "2030-01-01",
    }
    result = get_analytics_df(sample_analytics_df, params)

    assert list(result.keys()) == ["Custom"]
    assert result["Custom"]["organizations_by_size"] == []
    assert len(result["Custom"]["collaborator_vs_contributor"]) == 2


def test_get_analytics_both_custom_ranges(sample_analytics_df):
    """Both ranges return Custom-only with both sub-charts populated simultaneously."""
    params = {
        "country": "all",
        "org_type": "all",
        "size_start_date": "2020-01-01",
        "size_end_date": "2030-01-01",
        "contrib_start_date": "2020-01-01",
        "contrib_end_date": "2030-01-01",
    }
    result = get_analytics_df(sample_analytics_df, params)

    assert list(result.keys()) == ["Custom"]
    assert len(result["Custom"]["organizations_by_size"]) > 0
    assert len(result["Custom"]["collaborator_vs_contributor"]) == 2


def test_get_analytics_country_and_type_filters(sample_analytics_df):
    """Filters apply accurately against country code, country name, and org type."""
    # Filter by code
    p_code = {
        "country": "ca",
        "org_type": "all",
        "size_start_date": None,
        "size_end_date": None,
        "contrib_start_date": None,
        "contrib_end_date": None,
    }
    res_code = get_analytics_df(sample_analytics_df, p_code)
    assert sum(x["count"] for x in res_code["All"]["organizations_by_size"]) == 1

    # Filter by name
    p_name = {
        "country": "united states",
        "org_type": "for_profit",
        "size_start_date": None,
        "size_end_date": None,
        "contrib_start_date": None,
        "contrib_end_date": None,
    }
    res_name = get_analytics_df(sample_analytics_df, p_name)
    assert sum(x["count"] for x in res_name["All"]["organizations_by_size"]) == 2

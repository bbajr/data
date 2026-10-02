"""Unit tests for PostgreSQL-based size and contribution analytics."""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Force DB mode before importing the target module
os.environ["USE_MOCK_DATA"] = "false"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lambda_functions.size_contribution_analytics import (
    column_exists,
    fetch_analytics_db,
    fetch_contrib_analytics,
    fetch_contrib_analytics_custom,
    fetch_size_analytics,
    fetch_size_analytics_custom,
    get_base_sql_parts,
)


@pytest.fixture
def base_params():
    """Default parameters matching standard API defaults."""
    return {
        "country": "all",
        "org_type": "all",
        "size_start_date": None,
        "size_end_date": None,
        "contrib_start_date": None,
        "contrib_end_date": None,
    }


# ---------------------------------------------------------------------------
# UNIT TESTS: get_base_sql_parts & column_exists
# ---------------------------------------------------------------------------


def test_get_base_sql_parts_all_filters(base_params):
    """When filters are 'all', no WHERE conditions or args should be produced."""
    conditions, sql_args, base_from = get_base_sql_parts(base_params)

    assert conditions == []
    assert sql_args == []
    assert "LEFT JOIN" in base_from


def test_get_base_sql_parts_specific_filters():
    """Applies country and org_type conditions with parameter binding."""
    params = {"country": "us", "org_type": "non_profit"}
    conditions, sql_args, _ = get_base_sql_parts(params)

    assert len(conditions) == 2
    assert any("LOWER(c.country_code)" in cond for cond in conditions)
    assert any("LOWER(REPLACE(o.org_type" in cond for cond in conditions)
    assert sql_args == ["us", "us", "non_profit"]


def test_column_exists():
    """Checks whether cursor lookup returns boolean correctly."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)
    assert column_exists(mock_cursor, "organizations", "is_contributor") is True

    mock_cursor.fetchone.return_value = None
    assert column_exists(mock_cursor, "organizations", "is_contributor") is False


# ---------------------------------------------------------------------------
# UNIT TESTS: fetch_size_analytics
# ---------------------------------------------------------------------------


def test_fetch_size_analytics_preserves_order_and_omits_zero_counts(base_params):
    """Parses database rows across 4 buckets, omitting 0 counts and preserving order."""
    mock_cursor = MagicMock()
    # Simulates DB rows returned from GROUP BY org_size
    mock_cursor.fetchall.return_value = [
        {"size": "small", "count_7d": 1, "count_30d": 2, "count_1y": 5, "count_all": 10},
        {"size": "large", "count_7d": 0, "count_30d": 1, "count_1y": 2, "count_all": 3},
    ]

    result = fetch_size_analytics(mock_cursor, base_params)

    assert set(result.keys()) == {"7D", "30D", "1Y", "All"}
    # 7D has small (1), but large has count 0 so it's omitted
    assert result["7D"] == [{"size": "small", "count": 1}]
    # 30D includes both
    assert result["30D"] == [
        {"size": "small", "count": 2},
        {"size": "large", "count": 1},
    ]


def test_fetch_size_analytics_empty_result(base_params):
    """Empty DB response yields empty lists across all 4 buckets."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []

    result = fetch_size_analytics(mock_cursor, base_params)
    assert result == {"7D": [], "30D": [], "1Y": [], "All": []}


# ---------------------------------------------------------------------------
# UNIT TESTS: fetch_contrib_analytics
# ---------------------------------------------------------------------------


def test_fetch_contrib_analytics_independent_counts_and_percentages(base_params):
    """Calculates collaborator and contributor percentages correctly per bucket."""
    mock_cursor = MagicMock()
    # Mock column_exists check inside fetch_contrib_analytics
    mock_cursor.fetchone.side_effect = [
        (1,),  # column_exists -> True
        {      # Query aggregate row
            "total_all": 100, "collab_all": 50, "contrib_all": 25,
            "total_1y": 50, "collab_1y": 10, "contrib_1y": 5,
            "total_30d": 10, "collab_30d": 2, "contrib_30d": 0,
            "total_7d": 0, "collab_7d": 0, "contrib_7d": 0,
        },
    ]

    result = fetch_contrib_analytics(mock_cursor, base_params)

    # 100 total, 50 collab (50.0%), 25 contrib (25.0%)
    assert result["All"] == [
        {"type": "Collaborator", "count": 50, "percentage": 50.0},
        {"type": "Contributor", "count": 25, "percentage": 25.0},
    ]
    # Total = 0 defaults percentage to 0.0 without ZeroDivisionError
    assert result["7D"] == [
        {"type": "Collaborator", "count": 0, "percentage": 0.0},
        {"type": "Contributor", "count": 0, "percentage": 0.0},
    ]


def test_fetch_contrib_analytics_missing_contributor_column(base_params):
    """When is_contributor column does not exist, query substitutes 'false'."""
    mock_cursor = MagicMock()
    # Column check returns None
    mock_cursor.fetchone.side_effect = [
        None,  # column_exists -> False
        {
            "total_all": 10, "collab_all": 5, "contrib_all": 0,
            "total_1y": 0, "collab_1y": 0, "contrib_1y": 0,
            "total_30d": 0, "collab_30d": 0, "contrib_30d": 0,
            "total_7d": 0, "collab_7d": 0, "contrib_7d": 0,
        },
    ]

    fetch_contrib_analytics(mock_cursor, base_params)
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert "WHERE false" in executed_sql


# ---------------------------------------------------------------------------
# UNIT TESTS: Custom Date Range Functions
# ---------------------------------------------------------------------------


def test_fetch_size_analytics_custom():
    """Custom size query appends date window and parameters properly."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"size": "medium", "count": 4}]

    params = {
        "country": "all",
        "org_type": "all",
        "size_start_date": "2026-05-01",
        "size_end_date": "2026-05-31",
    }
    result = fetch_size_analytics_custom(mock_cursor, params)

    assert result == [{"size": "medium", "count": 4}]

    # Validate SQL execution and args
    called_args = mock_cursor.execute.call_args[0][1]
    assert "2026-05-01" in called_args
    assert "2026-05-31" in called_args


def test_fetch_contrib_analytics_custom():
    """Custom contributor query returns single bucket calculation."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        (1,),  # column_exists -> True
        {"total": 20, "collab": 10, "contrib": 2},
    ]

    params = {
        "country": "all",
        "org_type": "all",
        "contrib_start_date": "2026-06-01",
        "contrib_end_date": "2026-06-15",
    }
    result = fetch_contrib_analytics_custom(mock_cursor, params)

    assert result == [
        {"type": "Collaborator", "count": 10, "percentage": 50.0},
        {"type": "Contributor", "count": 2, "percentage": 10.0},
    ]


# ---------------------------------------------------------------------------
# INTEGRATION / SHAPE TESTS: fetch_analytics_db Orchestrator
# ---------------------------------------------------------------------------


def test_fetch_analytics_db_no_custom_dates(monkeypatch, base_params):
    """Standard call without dates returns 5 buckets including empty Custom."""
    mock_cursor = MagicMock()

    # Stub out the lower-level fetchers to test orchestration logic cleanly
    monkeypatch.setattr(
        "lambda_functions.size_contribution_analytics.fetch_size_analytics",
        lambda cur, p: {"7D": [{"size": "small", "count": 1}], "30D": [], "1Y": [], "All": []},
    )
    monkeypatch.setattr(
        "lambda_functions.size_contribution_analytics.fetch_contrib_analytics",
        lambda cur, p: {
            "7D": [{"type": "Collaborator", "count": 1, "percentage": 100.0}],
            "30D": [],
            "1Y": [],
            "All": [],
        },
    )

    result = fetch_analytics_db(mock_cursor, base_params)

    assert set(result.keys()) == {"7D", "30D", "1Y", "All", "Custom"}
    assert result["Custom"]["organizations_by_size"] == []
    assert result["Custom"]["collaborator_vs_contributor"] == []
    assert len(result["7D"]["organizations_by_size"]) == 1


def test_fetch_analytics_db_custom_branching(monkeypatch):
    """When custom dates are present, only returns 'Custom' bucket."""
    mock_cursor = MagicMock()

    monkeypatch.setattr(
        "lambda_functions.size_contribution_analytics.fetch_size_analytics_custom",
        lambda cur, p: [{"size": "large", "count": 5}],
    )
    monkeypatch.setattr(
        "lambda_functions.size_contribution_analytics.fetch_contrib_analytics_custom",
        lambda cur, p: [{"type": "Collaborator", "count": 5, "percentage": 50.0}],
    )

    # 1. Size only
    params_size = {
        "country": "all",
        "org_type": "all",
        "size_start_date": "2026-01-01",
        "size_end_date": "2026-02-01",
        "contrib_start_date": None,
        "contrib_end_date": None,
    }
    res_size = fetch_analytics_db(mock_cursor, params_size)
    assert list(res_size.keys()) == ["Custom"]
    assert len(res_size["Custom"]["organizations_by_size"]) == 1
    assert res_size["Custom"]["collaborator_vs_contributor"] == []

    # 2. Both custom ranges
    params_both = {
        "country": "all",
        "org_type": "all",
        "size_start_date": "2026-01-01",
        "size_end_date": "2026-02-01",
        "contrib_start_date": "2026-01-01",
        "contrib_end_date": "2026-02-01",
    }
    res_both = fetch_analytics_db(mock_cursor, params_both)
    assert list(res_both.keys()) == ["Custom"]
    assert len(res_both["Custom"]["organizations_by_size"]) == 1
    assert len(res_both["Custom"]["collaborator_vs_contributor"]) == 1

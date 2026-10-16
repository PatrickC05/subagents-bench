"""Fix-time estimates scored against the replayed fix times (``estimates.csv``) and the status
checks' coverage (``status_pairs.csv``).

# stub: replaced by the analysis-eta branch. Only the names and return shapes the levels analysis
# codes against (docs: the levels build contract, section 6).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .load import Trial


def estimate_rows(trial: Trial, fixes: dict[str, dict]) -> list[dict[str, Any]]:  # stub: replaced by the analysis-eta branch
    return []


def status_pair_rows(trial: Trial, fixes: dict[str, dict]) -> list[dict[str, Any]]:  # stub: replaced by the analysis-eta branch
    return []


def estimate_table(rows: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:  # stub: replaced by the analysis-eta branch
    """Per (sweep, condition): the estimates' scores and the status checks' coverage."""
    return pd.DataFrame()

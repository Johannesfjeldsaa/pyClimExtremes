"""Quantile-based climate indices (temperature and precipitation)."""

import numpy as np

from pyClimExtremes.indices.registry import register_index
from pyClimExtremes.indices.base_index import QuantileIndex
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


# ==================== TEMPERATURE QUANTILE INDICES ====================


@register_index
class tn_q10pIndex(QuantileIndex):
    """
    10th percentile of daily minimum temperature (tn_q10p)
    """

    index_type = "temperature"
    index_id = "tn_q10p"
    index_aliases = ["tn_q10p", "TN_Q10P"]
    index_long_name = "10th percentile of daily minimum temperature (degC) during the baseline period"
    index_units = "degC"
    unit_after_aggregation = {
        'deg_C': "degC",
        'K': "K"
    }
    quantile = 0.1
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn_q10p"
    baseline_period = (1981, 2010)

    wet_day_threshold = None    # Not applicable for temperature indices,
                                # but required by the QuantileIndex

@register_index
class tn_q90pIndex(QuantileIndex):
    """
    90th percentile of daily minimum temperature (tn_q90p).
    """

    index_type = "temperature"
    index_id = "tn_q90p"
    index_aliases = ["tn_q90p", "T_Q90P"]
    index_long_name = "90th percentile of daily minimum temperature (degC) during the baseline period"
    index_units = "degC"
    unit_after_aggregation = {
        'deg_C': "degC",
        'K': "K"
    }
    quantile = 0.9
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn_q90p"
    baseline_period = (1981, 2010)

    wet_day_threshold = None    # Not applicable for temperature indices,
                                # but required by the QuantileIndex

@register_index
class tx_q10pIndex(QuantileIndex):
    """
    10th percentile of daily maximum temperature (tx_q10p).
    """

    index_type = "temperature"
    index_id = "tx_q10p"
    index_aliases = ["tx_q10p", "T_Q10P"]
    index_long_name = "10th percentile of daily maximum temperature (degC) during the baseline period"
    index_units = "degC"
    unit_after_aggregation = {
        'deg_C': "degC",
        'K': "K"
    }
    quantile = 0.1
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx_q10p"
    baseline_period = (1981, 2010)

    wet_day_threshold = None    # Not applicable for temperature indices,
                                # but required by the QuantileIndex


@register_index
class tx_q90pIndex(QuantileIndex):
    """
    90th percentile of daily maximum temperature (tx_q90p).
    """

    index_type = "temperature"
    index_id = "tx_q90p"
    index_aliases = ["tx_q90p", "TX_Q90P"]
    index_long_name = "90th percentile of daily maximum temperature (degC) during the baseline period"
    index_units = "degC"
    unit_after_aggregation = {
        'deg_C': "degC",
        'K': "K"
    }
    quantile = 0.9
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx_q90p"
    baseline_period = (1981, 2010)

    wet_day_threshold = None    # Not applicable for temperature indices,
                                # but required by the QuantileIndex

# ==================== PRECIPITATION QUANTILE INDICES ====================


@register_index
class pr_q95pIndex(QuantileIndex):
    """
    95th percentile of daily precipitation (pr_q95p).
    """

    index_type = "precipitation"
    index_id = "pr_q95p"
    index_aliases = ["pr_q95p", "PR_Q95P"]
    index_long_name = "95th percentile of daily precipitation (mm/day) during the baseline period"
    index_units = "mm"
    unit_after_aggregation = {
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    quantile = 0.95
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "pr_q95p"
    baseline_period = (1981, 2010)

    wet_day_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }

@register_index
class pr_q99pIndex(QuantileIndex):
    """
    99th percentile of daily precipitation (pr_q99p).
    """

    index_type = "precipitation"
    index_id = "pr_q99p"
    index_aliases = ["pr_q99p", "PR_Q99P"]
    index_long_name = "99th percentile of daily precipitation (mm/day) during the baseline period"
    index_units = "mm"
    unit_after_aggregation = {
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    quantile = 0.99
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "pr_q99p"
    baseline_period = (1981, 2010)
    wet_day_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }

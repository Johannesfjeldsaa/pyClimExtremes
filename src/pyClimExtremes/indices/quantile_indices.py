"""Quantile-based climate indices (temperature and precipitation)."""

import numpy as np

from pyClimExtremes.indices.registry import register_index
from pyClimExtremes.indices.base_index import QuantileIndex
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


# ==================== TEMPERATURE QUANTILE INDICES ====================


@register_index
class TX90pIndex(QuantileIndex):
    """
    90th percentile of daily maximum temperature (tx90pETCCDI).
    """

    index_type = "temperature"
    index_id = "tx90pETCCDI"
    index_aliases = ["tx90p", "TX90p", "tx90pETCCDI"]
    index_long_name = "Annual number of days with TX > 90th percentile of baseline"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    quantile = 0.9
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx90p"
    baseline_period = (1981, 2010)


@register_index
class TN90pIndex(QuantileIndex):
    """
    90th percentile of daily minimum temperature (tn90pETCCDI).
    """

    index_type = "temperature"
    index_id = "tn90pETCCDI"
    index_aliases = ["tn90p", "TN90p", "tn90pETCCDI"]
    index_long_name = "Annual number of days with TN > 90th percentile of baseline"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    quantile = 0.9
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn90p"
    baseline_period = (1981, 2010)


@register_index
class TX10pIndex(QuantileIndex):
    """
    10th percentile of daily maximum temperature (tx10pETCCDI).
    """

    index_type = "temperature"
    index_id = "tx10pETCCDI"
    index_aliases = ["tx10p", "TX10p", "tx10pETCCDI"]
    index_long_name = "Annual number of days with TX < 10th percentile of baseline"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    quantile = 0.1
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx10p"
    baseline_period = (1981, 2010)


@register_index
class TN10pIndex(QuantileIndex):
    """
    10th percentile of daily minimum temperature (tn10pETCCDI).
    """

    index_type = "temperature"
    index_id = "tn10pETCCDI"
    index_aliases = ["tn10p", "TN10p", "tn10pETCCDI"]
    index_long_name = "Annual number of days with TN < 10th percentile of baseline"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    quantile = 0.1
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn10p"
    baseline_period = (1981, 2010)


# ==================== PRECIPITATION QUANTILE INDICES ====================


@register_index
class q95prIndex(QuantileIndex):
    """
    Annual total precipitation on very wet days (q95pr).

    Very wet days are defined as days with daily precipitation > 95th percentile.
    """

    index_type = "precipitation"
    index_id = "q95pr"
    index_aliases = ["q95pr", "Q95pr"]
    index_long_name = "Annual total precipitation on very wet days (>95th percentile)"
    index_units = "mm"
    unit_after_aggregation = {
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    quantile = 0.95
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "q95pr"
    baseline_period = (1981, 2010)

    wet_day_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }

@register_index
class q99prIndex(QuantileIndex):
    """
    Annual total precipitation on extremely wet days (q99pr).

    Extremely wet days are defined as days with daily precipitation > 99th percentile.
    """

    index_type = "precipitation"
    index_id = "q99pr"
    index_aliases = ["q99pr", "Q99pr"]
    index_long_name = "Annual total precipitation on extremely wet days (>99th percentile)"
    index_units = "mm"
    unit_after_aggregation = {
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    quantile = 0.99
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "q99pr"
    baseline_period = (1981, 2010)
    wet_day_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }

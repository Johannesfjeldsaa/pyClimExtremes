
import numpy as np

from pyClimExtremes.indices.registry import register_index
from pyClimExtremes.indices.base_index import (
    BaseIndex, ThresholdIndex,
    QuantileThresholdIndex, SpellDurationQuantileThresholdIndex
)
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


@register_index
class FDINDEX(BaseIndex):
    """
    Frost days (fdETCCDI).
    """

    index_type = "temperature"
    index_id = "fdETCCDI"
    index_aliases = ["fd", "FD", "fdETCCDI"]
    index_long_name = "Frost days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': "days",
        'K': "days"
    }
    fixed_threshold = {
        'deg_C': 0.0,
        'K': 273.15
    }
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "fd"


@register_index
class SUINDEX(ThresholdIndex):
    """
    Summer days (suETCCDI).
    """

    index_id = "suETCCDI"
    index_aliases = ["su", "SU", "suETCCDI"]
    index_long_name = "Summer days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': "days",
        'K': "days"
    }
    default_threshold = {
        'deg_C': 25.0,
        'K': 298.15
    }
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "su"
    index_type = "temperature"


@register_index
class IDINDEX(BaseIndex):
    """
    Ice days (idETCCDI).
    """

    index_type = "temperature"
    index_id = "idETCCDI"
    index_aliases = ["id", "ID", "idETCCDI"]
    index_long_name = "Ice days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': "days",
        'K': "days"
    }
    fixed_threshold = {
        'deg_C': 0.0,
        'K': 273.15
    }
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "id"


@register_index
class TRINDEX(BaseIndex):
    """
    Tropical nights (trETCCDI).
    """

    index_type = "temperature"
    index_id = "trETCCDI"
    index_aliases = ["tr", "TR", "trETCCDI"]
    index_long_name = "Tropical nights"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': "days",
        'K': "days"
    }
    fixed_threshold = {
        'deg_C': 20.0,
        'K': 293.15
    }
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tr"


#@register_index
class GSLIndex(BaseIndex):
    """
    Growing season length (gslETCCDI).
    """

    index_type = "temperature"
    index_id = "gslETCCDI"
    index_aliases = ["gsl", "GSL", "gslETCCDI"]
    index_long_name = "Growing season length"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': "days",
        'K': "days"
    }
    fixed_threshold = {
        'deg_C': 5.0,
        'K': 278.15
    }
    required_vars = ["tas"]
    frequencies = ["yr"]
    backend_callable_name = "gsl"


@register_index
class TXxIndex(BaseIndex):
    """
    Maximum of daily maximum temperature (txxETCCDI).
    """

    index_type = "temperature"
    index_id = "txxETCCDI"
    index_aliases = ["txx", "TXx", "txxETCCDI"]
    index_long_name = "Maximum of daily maximum temperature"
    index_units = "deg_C"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': 'deg_C',
        'K': 'K'
    }
    fixed_threshold = None  # Not used
    required_vars = ["tasmax"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "txx"


@register_index
class TXnIndex(BaseIndex):
    """
    Minimum of daily maximum temperature (txnETCCDI).
    """

    index_type = "temperature"
    index_id = "txnETCCDI"
    index_aliases = ["txn", "TXn", "txnETCCDI"]
    index_long_name = "Minimum of daily maximum temperature"
    index_units = "deg_C"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': 'deg_C',
        'K': 'K'
    }
    fixed_threshold = None  # Not used
    required_vars = ["tasmax"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "txn"


@register_index
class TNxIndex(BaseIndex):
    """
    Maximum of daily minimum temperature (tnxETCCDI).
    """

    index_type = "temperature"
    index_id = "tnxETCCDI"
    index_aliases = ["tnx", "TNx", "tnxETCCDI"]
    index_long_name = "Maximum of daily minimum temperature"
    index_units = "deg_C"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': 'deg_C',
        'K': 'K'
    }
    fixed_threshold = None  # Not used
    required_vars = ["tasmin"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "tnx"


@register_index
class TNnIndex(BaseIndex):
    """
    Minimum of daily minimum temperature (tnnETCCDI).
    """

    index_type = "temperature"
    index_id = "tnnETCCDI"
    index_aliases = ["tnn", "TNn", "tnnETCCDI"]
    index_long_name = "Minimum of daily minimum temperature"
    index_units = "deg_C"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': 'deg_C',
        'K': 'K'
    }
    fixed_threshold = None  # Not used
    required_vars = ["tasmin"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "tnn"


@register_index
class DTRIndex(BaseIndex):
    """Daily temperature range"""

    index_type = "temperature"
    index_id = "dtrETCCDI"
    index_aliases = ["dtr", "DTR", "dtrETCCDI"]
    index_long_name = "Daily temperature range"
    index_units = "deg_C"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        'deg_C': 'deg_C',
        'K': 'K'
    }
    fixed_threshold = None  # Not used
    required_vars = ["tasmax", "tasmin"]
    frequencies = ["mon", "yr"]
    backend_callable_name = 'dtr'

@register_index
class TN10pIndex(QuantileThresholdIndex):
    """Percentage of days when daily minimum temperature < 10th percentile
    of daily minimum temperature during the baseline period"""

    index_type = "temperature"
    index_id = "tn10pETCCDI"
    index_aliases = ["tn10p", "TN10P", "tn10pETCCDI"]
    index_long_name = "Percentage of days when daily minimum temperature < 10th percentile of daily minimum temperature during the baseline period"
    index_units = "%"
    unit_after_aggregation = {
        'deg_C': "%",
        'K': "%"
    }
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn10p"
    fixed_threshold = None
    quantile_threshold_index_id = "tn_q10p"
    threshold_is_doy_dependent = True

@register_index
class TN90pIndex(QuantileThresholdIndex):
    """Percentage of days when daily minimum temperature > 90th percentile
    of daily minimum temperature during the baseline period"""

    index_type = "temperature"
    index_id = "tn90pETCCDI"
    index_aliases = ["tn90p", "TN90P", "tn90pETCCDI"]
    index_long_name = "Percentage of days when daily minimum temperature > 90th percentile of daily minimum temperature during the baseline period"
    index_units = "%"
    unit_after_aggregation = {
        'deg_C': "%",
        'K': "%"
    }
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "tn90p"
    fixed_threshold = None
    quantile_threshold_index_id = "tn_q90p"
    threshold_is_doy_dependent = True

@register_index
class TX10pIndex(QuantileThresholdIndex):
    """Percentage of days when daily maximum temperature < 10th percentile
    of daily maximum temperature during the baseline period"""

    index_type = "temperature"
    index_id = "tx10pETCCDI"
    index_aliases = ["tx10p", "TX10P", "tx10pETCCDI"]
    index_long_name = "Percentage of days when daily maximum temperature < 10th percentile of daily maximum temperature during the baseline period"
    index_units = "%"
    unit_after_aggregation = {
        'deg_C': "%",
        'K': "%"
    }
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx10p"
    fixed_threshold = None
    quantile_threshold_index_id = "tx_q10p"
    threshold_is_doy_dependent = True

@register_index
class TX90pIndex(QuantileThresholdIndex):
    """Percentage of days when daily maximum temperature > 90th percentile
    of daily maximum temperature during the baseline period"""

    index_type = "temperature"
    index_id = "tx90pETCCDI"
    index_aliases = ["tx90p", "TX90P", "tx90pETCCDI"]
    index_long_name = "Percentage of days when daily maximum temperature > 90th percentile of daily maximum temperature during the baseline period"
    index_units = "%"
    unit_after_aggregation = {
        'deg_C': "%",
        'K': "%"
    }
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "tx90p"
    fixed_threshold = None
    quantile_threshold_index_id = "tx_q90p"
    threshold_is_doy_dependent = True


@register_index
class WSDIIndex(SpellDurationQuantileThresholdIndex):
    """Warm spell duration index (wsdiETCCDI).

    Annual count of days contributing to spells of at least 6 consecutive days
    where daily maximum temperature > 90th percentile of daily maximum
    temperature during the baseline period.
    """

    index_type = "temperature"
    index_id = "wsdiETCCDI"
    index_aliases = ["wsdi", "WSDI", "wsdiETCCDI"]
    index_long_name = "Warm spell duration index"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    required_vars = ["tasmax"]
    frequencies = ["yr"]
    backend_callable_name = "wsdi"
    fixed_threshold = None
    quantile_threshold_index_id = "tx_q90p"
    threshold_is_doy_dependent = True
    spells_can_span_groups = False


@register_index
class CSDIIndex(SpellDurationQuantileThresholdIndex):
    """Cold spell duration index (csdiETCCDI).

    Annual count of days contributing to spells of at least 6 consecutive days
    where daily minimum temperature < 10th percentile of daily minimum
    temperature during the baseline period.
    """

    index_type = "temperature"
    index_id = "csdiETCCDI"
    index_aliases = ["csdi", "CSDI", "csdiETCCDI"]
    index_long_name = "Cold spell duration index"
    index_units = "days"
    unit_after_aggregation = {
        'deg_C': "days",
        'K': "days"
    }
    required_vars = ["tasmin"]
    frequencies = ["yr"]
    backend_callable_name = "csdi"
    fixed_threshold = None
    quantile_threshold_index_id = "tn_q10p"
    threshold_is_doy_dependent = True
    spells_can_span_groups = False


import numpy as np

from pyClimExtremes.indices.registry import register_index
from pyClimExtremes.indices.base_index import BaseIndex, ThresholdIndex
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


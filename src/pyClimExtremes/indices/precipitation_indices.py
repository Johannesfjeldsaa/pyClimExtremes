import numpy as np
from pyClimExtremes.indices.registry import register_index
from pyClimExtremes.indices.base_index import BaseIndex, ThresholdIndex
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


# not done yet - @register_index
class CDDINDEX(BaseIndex):
    """
    Consecutive dry days (cddETCCDI).
    """

    index_type = "precipitation"
    index_id = "cddETCCDI"
    index_aliases = ["cdd", "CDD", "cddETCCDI"]
    index_long_name = "Consecutive dry days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    fixed_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "cdd"


# not done yet - @register_index
class CWDINDEX(BaseIndex):
    """
    Consecutive wet days (cwdETCCDI).
    """

    index_type = "precipitation"
    index_id = "cwdETCCDI"
    index_aliases = ["cwd", "CWD", "cwdETCCDI"]
    index_long_name = "Consecutive wet days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    fixed_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "cwd"


@register_index
class PRCPTOTINDEX(BaseIndex):
    """
    Annual total precipitation in wet days (prcptotETCCDI).
    """

    index_type = "precipitation"
    index_id = "prcptotETCCDI"
    index_aliases = ["prcptot", "PRCPTOT", "prcptotETCCDI"]
    index_long_name = "Annual total precipitation in wet days"
    index_units = "mm"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    fixed_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "prcptot"


@register_index
class RnnmmINDEX(ThresholdIndex):
    """
    Number of heavy precipitation days (rnnmmETCCDI).
    """

    index_type = "precipitation"
    index_id = "rnnmmETCCDI"
    index_aliases = ["rnnmm", "Rnnmm", "rnnmmETCCDI"]
    index_long_name = "Number of heavy precipitation days"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    default_threshold = None  # Must be provided by user
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "rnnmm"


@register_index
class r1mmINDEX(BaseIndex):
    """
    Annual total wet-day precipitation (r1mmETCCDI).
    """

    index_type = "precipitation"
    index_id = "r1mmETCCDI"
    index_aliases = ["r1mm", "R1mm", "r1mmETCCDI"]
    index_long_name = "Annual total wet-day precipitation"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    fixed_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "r1mm"


@register_index
class r10mmINDEX(BaseIndex):
    """
    Annual total heavy precipitation (r10mmETCCDI).
    """

    index_type = "precipitation"
    index_id = "r10mmETCCDI"
    index_aliases = ["r10mm", "R10mm", "r10mmETCCDI"]
    index_long_name = "Annual total heavy precipitation"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    fixed_threshold = {
        "mm d-1": 10.0,
        "kg m-2 s-1": 10.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "r10mm"


@register_index
class r20mmINDEX(BaseIndex):
    """
    Annual total very heavy precipitation (r20mmETCCDI).
    """

    index_type = "precipitation"
    index_id = "r20mmETCCDI"
    index_aliases = ["r20mm", "R20mm", "r20mmETCCDI"]
    index_long_name = "Annual total very heavy precipitation"
    index_units = "days"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "days",
        "kg m-2 s-1": "days"
    }
    fixed_threshold = {
        "mm d-1": 20.0,
        "kg m-2 s-1": 20.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "r20mm"

@register_index
class RXndayINDEX(ThresholdIndex):
    """
    Maximum n-day precipitation (rxndayETCCDI).
    """

    index_type = "precipitation"
    index_id = "rxndayETCCDI"
    index_aliases = ["rxnday", "Rxnday", "rxndayETCCDI"]
    index_long_name = "Maximum n-day precipitation"
    index_units = "mm"
    unit_after_aggregation = {   # before aggregation -> after aggregation
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    default_threshold = None  # Must be provided by user
    required_vars = ["pr"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "rxnday"


@register_index
class Rx1dayINDEX(BaseIndex):
    """
    Monthly maximum 1-day precipitation (rx1dayETCCDI).
    """

    index_type = "precipitation"
    index_id = "rx1dayETCCDI"
    index_aliases = ["rx1day", "Rx1day", "rx1dayETCCDI"]
    index_long_name = "Monthly maximum 1-day precipitation"
    index_units = "mm"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    fixed_threshold = None  # Not used
    required_vars = ["pr"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "rx1day"


@register_index
class Rx5dayINDEX(BaseIndex):
    """
    Monthly maximum 5-day precipitation (rx5dayETCCDI).
    """

    index_type = "precipitation"
    index_id = "rx5dayETCCDI"
    index_aliases = ["rx5day", "Rx5day", "rx5dayETCCDI"]
    index_long_name = "Monthly maximum 5-day precipitation"
    index_units = "mm"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "mm",
        "kg m-2 s-1": "kg m-2"
    }
    fixed_threshold = None  # Not used
    required_vars = ["pr"]
    frequencies = ["mon", "yr"]
    backend_callable_name = "rx5day"


@register_index
class SDIIINDEX(BaseIndex):
    """
    Simple daily intensity index (sdiiETCCDI).
    """

    index_type = "precipitation"
    index_id = "sdiiETCCDI"
    index_aliases = ["sdii", "SDII", "sdiiETCCDI"]
    index_long_name = "Simple daily intensity index"
    index_units = "mm d-1"
    unit_after_aggregation = {  # before aggregation -> after aggregation
        "mm d-1": "mm d-1",
        "kg m-2 s-1": "kg m-2 s-1"
    }
    fixed_threshold = {
        "mm d-1": 1.0,
        "kg m-2 s-1": 1.0 / 86400.0
    }
    required_vars = ["pr"]
    frequencies = ["yr"]
    backend_callable_name = "sdii"
import numpy as np

from reversclim.utils.preprocessing.variables.extremes.indices.registry import (
    register_index
)
from reversclim.utils.preprocessing.variables.extremes.indices.base_index import (
    BaseIndex,
)
from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)


@register_index
class Rx1dayINDEX(BaseIndex):
    """
    Monthly maximum 1-day precipitation (rx1dayETCCDI).
    """

    index_id = "rx1dayETCCDI"
    index_aliases = ["rx1day", "Rx1day", "rx1dayETCCDI"]
    index_long_name = "Monthly maximum 1-day precipitation"
    index_units = "mm day-1"
    required_vars = ["pr"]
    frequencies = ["mon", "ann"]
    backend_callable_name = "rx1day"


@register_index
class Rx5dayINDEX(BaseIndex):
    """
    Monthly maximum 5-day precipitation (rx5dayETCCDI).
    """

    index_id = "rx5dayETCCDI"
    index_aliases = ["rx5day", "Rx5day", "rx5dayETCCDI"]
    index_long_name = "Monthly maximum 5-day precipitation"
    index_units = "mm day-1"
    required_vars = ["pr"]
    frequencies = ["mon", "ann"]
    backend_callable_name = "rx5day"

@register_index
class SDIIINDEX(BaseIndex):
    """
    Simple daily intensity index (sdiiETCCDI).
    """

    index_id = "sdiiETCCDI"
    index_aliases = ["sdii", "SDII", "sdiiETCCDI"]
    index_long_name = "Simple daily intensity index"
    index_units = "mm day-1"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "sdii"

@register_index
class RnnmmINDEX(BaseIndex):
    """
    Number of heavy precipitation days (rnnmmETCCDI).
    """

    index_id = "rnnmmETCCDI"
    index_aliases = ["rnnmm", "Rnnmm", "rnnmmETCCDI"]
    index_long_name = "Number of heavy precipitation days"
    index_units = "days"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "rnnmm"

@register_index
class r1mmINDEX(BaseIndex):
    """
    Annual total wet-day precipitation (r1mmETCCDI).
    """

    index_id = "r1mmETCCDI"
    index_aliases = ["r1mm", "R1mm", "r1mmETCCDI"]
    index_long_name = "Annual total wet-day precipitation"
    index_units = "mm"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "r1mm"

@register_index
class r10mmINDEX(BaseIndex):
    """
    Annual total heavy precipitation (r10mmETCCDI).
    """

    index_id = "r10mmETCCDI"
    index_aliases = ["r10mm", "R10mm", "r10mmETCCDI"]
    index_long_name = "Annual total heavy precipitation"
    index_units = "mm"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "r10mm"

@register_index
class r20mmINDEX(BaseIndex):
    """
    Annual total very heavy precipitation (r20mmETCCDI).
    """

    index_id = "r20mmETCCDI"
    index_aliases = ["r20mm", "R20mm", "r20mmETCCDI"]
    index_long_name = "Annual total very heavy precipitation"
    index_units = "mm"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "r20mm"

@register_index
class PRCPTOTINDEX(BaseIndex):
    """
    Annual total precipitation in wet days (prcptotETCCDI).
    """

    index_id = "prcptotETCCDI"
    index_aliases = ["prcptot", "PRCPTOT", "prcptotETCCDI"]
    index_long_name = "Annual total precipitation in wet days"
    index_units = "mm"
    required_vars = ["pr"]
    frequencies = ["ann"]
    backend_callable_name = "prcptot"
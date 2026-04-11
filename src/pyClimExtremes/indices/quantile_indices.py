"""Quantile-based climate indices (temperature and precipitation)."""

import numpy as np

from pyClimExtremes.indices.registry import register_index, QUANTILE_REGISTRY
from pyClimExtremes.indices.base_index import QuantileIndex
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


_RUNTIME_QUANTILE_CLASS_CACHE: dict[tuple[str, float], type[QuantileIndex]] = {}


@register_index
class tn_qXXpIndex(QuantileIndex):
  """Template for custom daily minimum temperature quantiles."""

  index_type = "temperature_quantile"
  index_id = "tn_qXXp"
  index_aliases = ["tn_qXXp", "TN_QXXP"]
  index_long_name = "Custom percentile of daily minimum temperature during the baseline period"
  long_name_template = "{percentile:g}th percentile of daily minimum temperature during the baseline period"
  index_units = "degC"
  unit_after_aggregation = {
    "deg_C": "degC",
    "K": "K"
  }
  quantile = np.nan
  default_quantile = None
  is_generic_template = True
  required_vars = ["tasmin"]
  frequencies = ["yr"]
  backend_callable_name = "tn_qXXp"
  baseline_period = (1981, 2010)
  wet_day_threshold = None


@register_index
class tx_qXXpIndex(QuantileIndex):
  """Template for custom daily maximum temperature quantiles."""

  index_type = "temperature_quantile"
  index_id = "tx_qXXp"
  index_aliases = ["tx_qXXp", "TX_QXXP"]
  index_long_name = "Custom percentile of daily maximum temperature during the baseline period"
  long_name_template = "{percentile:g}th percentile of daily maximum temperature during the baseline period"
  index_units = "degC"
  unit_after_aggregation = {
    "deg_C": "degC",
    "K": "K"
  }
  quantile = np.nan
  default_quantile = None
  is_generic_template = True
  required_vars = ["tasmax"]
  frequencies = ["yr"]
  backend_callable_name = "tx_qXXp"
  baseline_period = (1981, 2010)
  wet_day_threshold = None


@register_index
class pr_qXXpIndex(QuantileIndex):
  """Template for custom daily precipitation quantiles."""

  index_type = "precipitation_quantile"
  index_id = "pr_qXXp"
  index_aliases = ["pr_qXXp", "PR_QXXP"]
  index_long_name = "Custom percentile of daily precipitation during the baseline period"
  long_name_template = "{percentile:g}th percentile of daily precipitation during the baseline period"
  index_units = "mm"
  unit_after_aggregation = {
    "mm d-1": "mm",
    "kg m-2 s-1": "kg m-2"
  }
  quantile = np.nan
  default_quantile = None
  is_generic_template = True
  required_vars = ["pr"]
  frequencies = ["yr"]
  backend_callable_name = "pr_qXXp"
  baseline_period = (1981, 2010)
  wet_day_threshold = {
    "mm d-1": 1.0,
    "kg m-2 s-1": 1.0 / 86400.0
  }


def _format_percentile_id(percentile: float) -> str:
  """Format percentiles for stable runtime index ids such as q10p or q12_5p."""
  if float(percentile).is_integer():
    return str(int(percentile))
  return str(percentile).replace(".", "_")


def build_runtime_quantile_class(
  family_id: str,
  percentile: float,
) -> type[QuantileIndex]:
  """Create or reuse a runtime QuantileIndex subclass for qXXp requests."""
  cache_key = (family_id, percentile)
  if cache_key in _RUNTIME_QUANTILE_CLASS_CACHE:
    return _RUNTIME_QUANTILE_CLASS_CACHE[cache_key]

  template = QUANTILE_REGISTRY.get(family_id)
  if template is None or not getattr(template, "is_generic_template", False):
    err_msg = (
      f"Unsupported generic quantile family '{family_id}'. Supported values are: "
      f"{sorted(k for k, v in QUANTILE_REGISTRY.items() if getattr(v, 'is_generic_template', False))}"
    )
    logger.error(err_msg, stack_info=True)
    raise ValueError(err_msg)

  percentile_id = _format_percentile_id(percentile)
  index_id = family_id.replace("XX", percentile_id)
  attrs = {
    "index_type": template.index_type,
    "index_id": index_id,
    "index_aliases": [index_id, index_id.upper()],
    "index_long_name": template.long_name_template.format(percentile=percentile),
    "index_units": template.index_units,
    "unit_after_aggregation": dict(template.unit_after_aggregation),
    "quantile": percentile / 100.0,
    "required_vars": list(template.required_vars),
    "frequencies": ["yr"],
    "backend_callable_name": template.backend_callable_name,
    "baseline_period": template.baseline_period,
    "wet_day_threshold": template.wet_day_threshold,
  }
  quantile_class = type(f"{index_id}RuntimeIndex", (QuantileIndex,), attrs)
  _RUNTIME_QUANTILE_CLASS_CACHE[cache_key] = quantile_class
  return quantile_class


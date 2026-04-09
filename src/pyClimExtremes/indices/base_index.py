"""Base class for ETCCDI indices and delegation to compute backends."""

from typing import Any
import inspect
import numpy as np

from pyClimExtremes.compute_backend.backend_registry import get_compute_backend
from pyClimExtremes.indices.registry import INPUT_VAR_ALIASES
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


def validate_data_array(
    data_array: np.ndarray | dict[str, np.ndarray],
    required_vars: list[str]
) -> np.ndarray | dict[str, np.ndarray]:
    """Helper function to validate and extract the relevant data array
    from the input.

    Parameters
    ----------
    data_array : np.ndarray | dict[str, np.ndarray]
        Input data array, either as a single ndarray or a dict of ndarrays.
    required_vars : list[str]
        List of required variable names to look for in the data_array dict.

    Returns
    -------
    dict[str, np.ndarray]
        Extracted data arrays for the required variables.

    Raises
    ------
    ValueError
        If data_array is a dict but none of the valid keys are found.
    TypeError
        If data_array is neither an ndarray nor a dict.
    """
    if not required_vars or len(required_vars) == 0:
        err_msg = "No required variables specified for validation."
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    if not isinstance(data_array, (np.ndarray, dict)):
        err_msg = (
            "data_array must be either a np.ndarray or a dict of "
            "np.ndarrays."
        )
        logger.error(err_msg, stack_info=True)
        raise TypeError(err_msg)

    if len(required_vars) == 1: # single required variable return single array
        valid_keys = INPUT_VAR_ALIASES[required_vars[0]]
        if isinstance(data_array, np.ndarray):
            return data_array
        elif isinstance(data_array, dict):
            for key in valid_keys:
                if key in data_array:
                    return data_array[key]

            err_msg = (
                f"Input data_array dict must contain one of the keys: "
                f"{valid_keys} for variable '{required_vars[0]}'."
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
    else: # multiple required variables, return dict of arrays
        return_data = {}
        if isinstance(data_array, dict):
            for required_var in required_vars:
                valid_keys = INPUT_VAR_ALIASES[required_var]
                for key in valid_keys:
                    if key in data_array:
                        return_data[required_var] = data_array[key]
                        break
                else:
                    err_msg = (
                        f"Input data_array dict must contain one of the keys: "
                        f"{valid_keys} for required variable '{required_var}'."
                    )
                    logger.error(err_msg, stack_info=True)
                    raise ValueError(err_msg)
            return return_data
        else:
            err_msg = (
                "When multiple required variables are specified, "
                "data_array must be a dict of np.ndarrays."
            )
            logger.error(err_msg, stack_info=True)
            raise TypeError(err_msg)

class BaseIndex:
    """Abstract base for ETCCDI indices.

    Provides common metadata fields and wires each index instance to a
    registered compute backend (e.g., the in-tree Python implementation).
    """

    # --- Metadata (to be overridden) ---
    index_type:             str = ""  # e.g., 'temperature' or 'precipitation'
    index_id:               str = ""  # e.g. 'txxETCCDI'
    index_aliases:          list[str] = []  # e.g. ['txx', 'TXx', 'txxETCCDI']
    index_long_name:        str = ""  # e.g.'Annual maximum ...'
    index_units:            str = ""  # e.g. 'deg_C'
    unit_after_aggregation: dict[str, str] = {}  # e.g. {before: after}
    required_vars:          list[str] = []  # e.g. ['tasmax']
    frequencies:            list[str] = []  # e.g. ['yr', 'mon']
    backend_callable_name:  str = "" # e.g. txx
    fixed_threshold:        dict[str, float] | None = None

    # --- Public API ---
    def __init__(self, compute_backend: str, **kwargs: dict[str, Any]):
        self.backend_name = compute_backend
        self.compute_backend = get_compute_backend(compute_backend)
        self.backend_kwargs = kwargs
        if self.index_type == "temperature":
            self.allowed_input_units = ["deg_C", "K"]
        elif self.index_type == "precipitation":
            self.allowed_input_units = ["mm d-1", "kg m-2 s-1"]
        else:
            err_msg = (
                f"Index '{self.index_id}' has unrecognized index_type "
                f"'{self.index_type}'."
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

    def compute(
        self,
        compute_fq:         str,
        data_array:         np.ndarray | dict[str, np.ndarray],
        group_index:        np.ndarray,
        time_array:         np.ndarray | None = None,
        time_units:         str | None = None,
        calendar:           str | None = None,
        lat:                np.ndarray | None = None,
        fixed_threshold:    dict[str, float] | None = None,
    ):
        """Compute the index using backend method specified by
        backend_callable_name. Common function to all indices that handles
        data validation and delegation to the backend.

        Parameters
        ----------
        compute_fq : str
            The computation frequency (e.g., 'mon', 'yr').
        data_array : np.ndarray | dict[str, np.ndarray]
            Input data array(s). For single-variable indices, a single ndarray
            can be passed. For multi-variable indices, a dict with variable names
            as keys is required.
        group_index : np.ndarray
            Time grouping indices for aggregation.
        time_array : np.ndarray | None, optional
            Time array, passed to backend if needed.
        fixed_threshold : dict[str, float] | None, optional
            Fixed threshold values for indices that require them.

        Returns
        -------
        np.ndarray
            Computed index values at the requested frequency.

        Raises
        ------
        ValueError
            If backend_callable_name is not set.
        AttributeError
            If the backend does not have the required method.
        """
        # Validate and extract data
        validated_data = validate_data_array(data_array, self.required_vars)

        # Get the backend method dynamically
        if not self.backend_callable_name:
            err_msg = (
                f"backend_callable_name not set for {self.__class__.__name__}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        try:
            backend_method = getattr(self.compute_backend, self.backend_callable_name)
        except AttributeError as e:
            err_msg = (
                f"Backend '{self.backend_name}' has no method "
                f"'{self.backend_callable_name}'"
            )
            logger.error(err_msg, stack_info=True)
            raise AttributeError(err_msg) from e

        # Build kwargs for the backend method
        kwargs = {
            "compute_fq": compute_fq,
            "group_index": group_index,
            "fixed_threshold": fixed_threshold
        }

        # Add data with backend-friendly variable names ({var}_data)
        if len(self.required_vars) == 1:
            # Single variable: pass as {var}_data
            var_name = self.required_vars[0]
            backend_var_name = f"{var_name}_data"
            kwargs[backend_var_name] = validated_data
        else:
            # Multiple variables: pass each as {var}
            # double check that validated_data is a dict
            if not isinstance(validated_data, dict):
                err_msg = (
                    "Validated data should be a dict when multiple "
                    "required variables are specified."
                )
                logger.error(err_msg, stack_info=True)
                raise TypeError(err_msg)
            for var_name, arr in validated_data.items():
                backend_var_name = f"{var_name}_data"
                kwargs[backend_var_name] = arr

        # Add optional arrays if provided
        if time_array is not None:
            kwargs["time_array"] = time_array
        if time_units is not None:
            kwargs["time_units"] = time_units
        if calendar is not None:
            kwargs["calendar"] = calendar
        if lat is not None:
            kwargs["lat"] = lat

        # check if filtering is needed for this kwarg-index pair
        # Log unfiltered kwargs
        logger.debug(f"Unfiltered kwargs keys: {list(kwargs.keys())}")

        # Filter kwargs to only include parameters the backend method accepts
        # since the python backend wraps methods, we need to unwrap it first
        unwrapped_method = inspect.unwrap(backend_method)
        sig = inspect.signature(unwrapped_method)

        logger.debug(
            "Unwrapped method signature parameters: %s",
            list(sig.parameters.keys())
        )

        # Filter to explicit parameters - only include if explicitly in sign
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        # Log what we're passing
        logger.debug(
            "Calling %s with kwargs: %s",
            self.backend_callable_name, list(filtered_kwargs.keys())
        )

        return backend_method(**filtered_kwargs)


class ThresholdIndex(BaseIndex):
    """Index subclass for threshold-based indices.

    Extends BaseIndex to support indices that use a scalar threshold value
    (e.g., SU for summer days where Tmax > threshold). Subclasses should
    define `default_threshold` as a class attribute.

    Examples
    --------
    class SUINDEX(ThresholdIndex):
        index_id = "suETCCDI"
        default_threshold = 25.0  # degrees Celsius
        backend_callable_name = "su"  # backend method must accept threshold param
    """

    default_threshold:  dict[str, float] | None = None  # e.g., 25.0 deg C

    def compute(
        self,
        compute_fq:         str,
        data_array:         np.ndarray | dict[str, np.ndarray],
        group_index:        np.ndarray,
        threshold:          float,
        time_array:         np.ndarray | None = None,
        time_units:         str | None = None,
        calendar:           str | None = None,
        lat:                np.ndarray | None = None,
        threshold_array:    np.ndarray | None = None,
    ):
        """Compute a threshold-based index with optional custom threshold.

        Parameters
        ----------
        compute_fq : str
            The computation frequency (e.g., 'mon', 'yr').
        data_array : np.ndarray | dict[str, np.ndarray]
            Input data array(s).
        group_index : np.ndarray
            Time grouping indices for aggregation.
        threshold : float
            Threshold value to use for the computation.
        time_array : np.ndarray | None, optional
            Time array, passed to backend if needed.
        threshold_array : np.ndarray | None, optional
            Threshold/quantile array, passed to backend if needed.

        Returns
        -------
        np.ndarray
            Computed index values at the requested frequency.
        """

        # Validate and extract data
        validated_data = validate_data_array(data_array, self.required_vars)

        # Get the backend method dynamically
        if not self.backend_callable_name:
            err_msg = (
                f"backend_callable_name not set for {self.__class__.__name__}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        try:
            backend_method = getattr(self.compute_backend, self.backend_callable_name)
        except AttributeError as e:
            err_msg = (
                f"Backend '{self.backend_name}' has no method "
                f"'{self.backend_callable_name}'"
            )
            logger.error(err_msg, stack_info=True)
            raise AttributeError(err_msg) from e

        # Build kwargs for the backend method
        kwargs = {"compute_fq": compute_fq, "group_index": group_index}

        # Add data with backend-friendly variable names ({var}_data)
        if len(self.required_vars) == 1:
            # Single variable: pass as {var}_data
            var_name = self.required_vars[0]
            backend_var_name = f"{var_name}_data"
            kwargs[backend_var_name] = validated_data
        else:
            # Multiple variables: pass each as {var}_data
            if not isinstance(validated_data, dict):
                err_msg = (
                    "Validated data should be a dict when multiple "
                    "required variables are specified."
                )
                logger.error(err_msg, stack_info=True)
                raise TypeError(err_msg)
            for var_name, arr in validated_data.items():
                backend_var_name = f"{var_name}_data"
                kwargs[backend_var_name] = arr

        # Add optional arrays if provided
        if time_array is not None:
            kwargs["time_array"] = time_array
        if time_units is not None:
            kwargs["time_units"] = time_units
        if calendar is not None:
            kwargs["calendar"] = calendar
        if lat is not None:
            kwargs["lat"] = lat
        if threshold is not None:
            kwargs["threshold"] = threshold
        if threshold_array is not None:
            kwargs["threshold_array"] = threshold_array

        # Log unfiltered kwargs
        logger.debug(f"Unfiltered kwargs keys: {list(kwargs.keys())}")

        # Filter kwargs to only include parameters the backend method accepts
        unwrapped_method = inspect.unwrap(backend_method)
        sig = inspect.signature(unwrapped_method)

        logger.debug(
            "Unwrapped method signature parameters: %s",
            list(sig.parameters.keys())
        )

        # Filter to explicit parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        # Log what we're passing
        logger.debug(
            "Calling %s with kwargs: %s",
            self.backend_callable_name, list(filtered_kwargs.keys())
        )

        return backend_method(**filtered_kwargs)

class QuantileIndex(BaseIndex):
    """Index subclass for quantile-based indices.

    Indexes that are based on quantiles of the data:
    * tn10p
    * tn90p
    * tx10p
    * tx90p
    * R95p(tot)
    * R99p(tot)
    And then further used for threshold-based indices, e.g.:
        WSDI (Warm Spell Duration Index) which measures prolonged heat events.
        Here we need the 90th percentile of daily maximum temperature (tx90p)
        as the threshold for defining a "warm day".
        The WSDI is then calculated as the number of days
        in spells of at least 6 consecutive warm days.

    In addition to BaseIndex metadata, QuantileIndex holds metadata such as:
    * quantile value
    * baseline period for quantile calculation
    """

    quantile: float = 0.0                   # e.g., 0.1 for 10th percentile
    baseline_period: tuple = (1981, 2010)   # (start_year, end_year)

    def __init__(self, compute_backend: str, **kwargs: dict[str, Any]):
        super().__init__(compute_backend, **kwargs)
        self.thresholds_by_doy: np.ndarray | None = None

    def compute(
        self,
        compute_fq:         str,
        data_array:         np.ndarray | dict[str, np.ndarray],
        group_index:        np.ndarray,
        time_array:         np.ndarray | None = None,
        time_units:         str | None = None,
        calendar:           str | None = None,
        lat:                np.ndarray | None = None,
        base_period_mask:   np.ndarray | None = None,
        window_size:        int = 5,
        bootstrap_samples:  int = 1000,
        random_seed:        int | None = None,
    ):
        """Compute quantile-based index with optional bootstrap for base period.

        For temperature indices, computes yearly exceedance frequencies using:
        - Fixed daily thresholds (5-day window percentiles) for years outside base period
        - Bootstrap-averaged exceedance frequencies for years inside base period

        For precipitation indices, computes a single quantile value per grid point.

        Parameters
        ----------
        compute_fq : str
            The computation frequency (e.g., 'mon', 'yr').
        data_array : np.ndarray | dict[str, np.ndarray]
            Input data array(s).
        group_index : np.ndarray
            Time grouping indices for aggregation (maps daily times to years/months).
        time_array : np.ndarray | None, optional
            Time array, passed to backend if needed.
        base_period_mask : np.ndarray | None, optional
            Boolean array of shape (num_years,) marking base-period years.
            Required for temperature quantiles with bootstrap.
        window_size : int, optional
            Size of rolling window for daily threshold estimation, by default 5
        bootstrap_samples : int, optional
            Number of bootstrap resamples for years in base period, by default 1000
        random_seed : int | None, optional
            Seed for bootstrap random number generator

        Returns
        -------
        np.ndarray
            Computed index values (yearly exceedance frequencies for temperature,
            scalar quantile for precipitation).
        """
        # Validate and extract data
        validated_data = validate_data_array(data_array, self.required_vars)

        # Get the backend method dynamically
        if not self.backend_callable_name:
            err_msg = (
                f"backend_callable_name not set for {self.__class__.__name__}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        try:
            backend_method = getattr(self.compute_backend, self.backend_callable_name)
        except AttributeError as e:
            err_msg = (
                f"Backend '{self.backend_name}' has no method "
                f"'{self.backend_callable_name}'"
            )
            logger.error(err_msg, stack_info=True)
            raise AttributeError(err_msg) from e

        # Build kwargs for the backend method
        kwargs = {
            "compute_fq": compute_fq,
            "group_index": group_index,
            "quantile": self.quantile,
        }

        # Add data with backend-friendly variable names ({var}_data)
        if len(self.required_vars) == 1:
            var_name = self.required_vars[0]
            backend_var_name = f"{var_name}_data"
            kwargs[backend_var_name] = validated_data
        else:
            if not isinstance(validated_data, dict):
                err_msg = (
                    "Validated data should be a dict when multiple "
                    "required variables are specified."
                )
                logger.error(err_msg, stack_info=True)
                raise TypeError(err_msg)
            for var_name, arr in validated_data.items():
                backend_var_name = f"{var_name}_data"
                kwargs[backend_var_name] = arr

        # Add optional parameters for temperature quantiles
        if self.index_type == "temperature":
            if base_period_mask is None:
                err_msg = "base_period_mask is required for temperature quantile indices"
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            kwargs["base_period_mask"] = base_period_mask
            kwargs["window_size"] = window_size
            kwargs["bootstrap_samples"] = bootstrap_samples
            if random_seed is not None:
                kwargs["random_seed"] = random_seed
            if time_array is not None:
                kwargs["time_array"] = time_array
            if time_units is not None:
                kwargs["time_units"] = time_units
            if calendar is not None:
                kwargs["calendar"] = calendar

        # Add optional arrays if provided
        if lat is not None:
            kwargs["lat"] = lat

        # Log unfiltered kwargs
        logger.debug(f"Unfiltered kwargs keys: {list(kwargs.keys())}")

        # Filter kwargs to only include parameters the backend method accepts
        unwrapped_method = inspect.unwrap(backend_method)
        sig = inspect.signature(unwrapped_method)

        logger.debug(
            "Unwrapped method signature parameters: %s",
            list(sig.parameters.keys())
        )

        # Filter to explicit parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        # Log what we're passing
        logger.debug(
            "Calling %s with kwargs: %s",
            self.backend_callable_name, list(filtered_kwargs.keys())
        )

        result = backend_method(**filtered_kwargs)

        # For temperature quantiles, backend may return dict with 'result' and 'thresholds'
        if isinstance(result, dict):
            if 'thresholds' in result:
                self.thresholds_by_doy = result['thresholds']
                result = result['result']

        return result


class QuantileThresholdIndex(BaseIndex):
    """Index subclass for indices that use quantile thresholds.

    These are indices that depend on pre-computed quantile thresholds.
    Examples:
    * WSDI (Warm Spell Duration Index) uses tx90p as threshold
    * CSDI (Cold Spell Duration Index) uses tn10p as threshold

    The thresholds can be passed as:
    1. A pre-computed array via quantile_thresholds parameter
    2. By reference to a QuantileIndex instance via quantile_index parameter
    """

    def compute(
        self,
        compute_fq:         str,
        data_array:         np.ndarray | dict[str, np.ndarray],
        group_index:        np.ndarray,
        time_array:         np.ndarray | None = None,
        time_units:         str | None = None,
        calendar:           str | None = None,
        lat:                np.ndarray | None = None,
        quantile_thresholds: np.ndarray | None = None,
        quantile_index:     'QuantileIndex | None' = None,
    ):
        """Compute a threshold-based index using quantile thresholds.

        Parameters
        ----------
        compute_fq : str
            The computation frequency (e.g., 'mon', 'yr').
        data_array : np.ndarray | dict[str, np.ndarray]
            Input data array(s).
        group_index : np.ndarray
            Time grouping indices for aggregation.
        time_array : np.ndarray | None, optional
            Time array, passed to backend if needed.
        quantile_thresholds : np.ndarray | None, optional
            Pre-computed threshold array (shape: 365/366 × lat × lon for daily, or scalar).
        quantile_index : QuantileIndex | None, optional
            QuantileIndex instance to use; will use its cached thresholds_by_doy.

        Returns
        -------
        np.ndarray
            Computed index values at the requested frequency.
        """
        # Get thresholds from either quantile_index or quantile_thresholds
        if quantile_index is not None:
            thresholds_array = quantile_index.thresholds_by_doy
            if thresholds_array is None:
                err_msg = (
                    f"quantile_index '{quantile_index.index_id}' has not been computed "
                    "yet; thresholds_by_doy is None."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
        elif quantile_thresholds is not None:
            thresholds_array = quantile_thresholds
        else:
            err_msg = (
                "Either quantile_thresholds or quantile_index must be provided"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        # Validate and extract data
        validated_data = validate_data_array(data_array, self.required_vars)

        # Get the backend method dynamically
        if not self.backend_callable_name:
            err_msg = (
                f"backend_callable_name not set for {self.__class__.__name__}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        try:
            backend_method = getattr(self.compute_backend, self.backend_callable_name)
        except AttributeError as e:
            err_msg = (
                f"Backend '{self.backend_name}' has no method "
                f"'{self.backend_callable_name}'"
            )
            logger.error(err_msg, stack_info=True)
            raise AttributeError(err_msg) from e

        # Build kwargs for the backend method
        kwargs = {"compute_fq": compute_fq, "group_index": group_index}

        # Add data with backend-friendly variable names ({var}_data)
        if len(self.required_vars) == 1:
            var_name = self.required_vars[0]
            backend_var_name = f"{var_name}_data"
            kwargs[backend_var_name] = validated_data
        else:
            if not isinstance(validated_data, dict):
                err_msg = (
                    "Validated data should be a dict when multiple "
                    "required variables are specified."
                )
                logger.error(err_msg, stack_info=True)
                raise TypeError(err_msg)
            for var_name, arr in validated_data.items():
                backend_var_name = f"{var_name}_data"
                kwargs[backend_var_name] = arr

        # Pass threshold array
        kwargs["threshold_array"] = thresholds_array

        # Add optional arrays if provided
        if time_array is not None:
            kwargs["time_array"] = time_array
        if time_units is not None:
            kwargs["time_units"] = time_units
        if calendar is not None:
            kwargs["calendar"] = calendar
        if lat is not None:
            kwargs["lat"] = lat

        # Log unfiltered kwargs
        logger.debug(f"Unfiltered kwargs keys: {list(kwargs.keys())}")

        # Filter kwargs to only include parameters the backend method accepts
        unwrapped_method = inspect.unwrap(backend_method)
        sig = inspect.signature(unwrapped_method)

        logger.debug(
            "Unwrapped method signature parameters: %s",
            list(sig.parameters.keys())
        )

        # Filter to explicit parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        # Log what we're passing
        logger.debug(
            "Calling %s with kwargs: %s",
            self.backend_callable_name, list(filtered_kwargs.keys())
        )

        return backend_method(**filtered_kwargs)
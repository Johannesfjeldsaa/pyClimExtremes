# compute_backends/python_backend.py
import numpy as np
from netCDF4 import num2date
from functools import wraps

from reversclim.utils.preprocessing.variables.extremes.compute_backend.backend_registry import (
    register_backend,
)
from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)

# TODO: ETCCDI_TEMPERATURE_VARIABLES = [
#    "txx",   # Maximum value of daily maximum temperature
#    "tnn",   # Minimum value of daily minimum temperature
#    "txn",   # Minimum value of daily maximum temperature
#    "tnx",   # Maximum value of daily minimum temperature
#    "fd",    # Frost days (number of days with Tmin < 0°C)
#    "id",    # Ice days (number of days with Tmax < 0°C)
#    "su",    # Summer days (number of days with Tmax > 25°C)
#    "tr",    # Tropical nights (number of days with Tmin > 20°C)
#    "gsl",   # Growing season length
#    "wsdi",  # Warm spell duration index
#    "csdi",  # Cold spell duration index
 #   "dtr",   # Diurnal temperature range (Tmax - Tmin)
#    "tx10p", # Percentage of days when Tmax < 10th percentile
#    "tx90p", # Percentage of days when Tmax > 90th percentile
#    "tn10p", # Percentage of days when Tmin < 10th percentile
#    "tn90p", # Percentage of days when Tmin > 90th percentile
#]

supported_compute_frequencies = ["mon", "ann"]

def check_supported_compute_frequencies(func):
    """Decorator to check if compute_fq is supported."""
    @wraps(func)
    def wrapper(self, compute_fq, *args, **kwargs):
        if compute_fq not in supported_compute_frequencies:
            err_msg = (
                f"Unsupported compute_fq '{compute_fq}', expected one of: "
                f"{', '.join(supported_compute_frequencies)}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        return func(self, compute_fq, *args, **kwargs)
    return wrapper

@register_backend("python")
class PythonBackend:
    """Python backend for ETCCDI index calculations."""

    def get_time_out(
        self,
        compute_fq:     str,
        time_array:     np.ndarray,
        time_units:     str,
        calendar:       str,
        group_index:    np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute output time coordinate using mean-of-period timestamps.

        If `group_index` is provided (the `inv` array from `np.unique`), it is
        reused to avoid recomputing grouping. Otherwise grouping is derived
        from the time array.
        """
        if time_array is None or time_array.size == 0:
            return np.array([])

        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.fromiter((d.year for d in dates), dtype=int, count=len(time_array))

        if compute_fq == "mon":
            months = np.fromiter((d.month for d in dates), dtype=int, count=len(time_array))
            group_key = years * 12 + (months - 1)
        elif compute_fq == "ann":
            group_key = years
        else:
            err_msg = (
                f"Unsupported compute_fq '{compute_fq}', expected one of"
                ", ".join(supported_compute_frequencies)
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        if group_index is None:
            _, inv = np.unique(group_key, return_inverse=True)
        else:
            inv = group_index

        n_groups = int(inv.max()) + 1 if inv.size else 0
        out = np.empty(n_groups, dtype=float)
        for i in range(n_groups):
            out[i] = float(time_array[inv == i].mean())

        return out

    def group_indices(
        self,
        compute_fq: str,
        time_array: np.ndarray,
        time_units: str,
        calendar: str = "standard",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (uniq, inv) grouping arrays for time aggregation."""

        if time_array is None or time_array.size == 0:
            return np.array([]), np.array([], dtype=int)

        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.fromiter((d.year for d in dates), dtype=int, count=len(time_array))

        if compute_fq == "mon":
            months = np.fromiter((d.month for d in dates), dtype=int, count=len(time_array))
            group_key = years * 12 + (months - 1)
        elif compute_fq == "ann":
            group_key = years
        else:
            raise ValueError(
                f"Unsupported compute_fq '{compute_fq}', expected 'mon' or 'ann'."
            )

        uniq, inv = np.unique(group_key, return_inverse=True)
        return uniq, inv

    def _aggregate_by_group(
        self,
        data: np.ndarray,
        group_index: np.ndarray | None,
        reducer,
    ) -> np.ndarray:
        """Aggregate over the time dimension using provided group mapping."""

        if group_index is None:
            raise ValueError("group_index is required for aggregation")

        n_groups = int(group_index.max()) + 1 if group_index.size else 0
        out_shape = (n_groups,) + tuple(data.shape[1:])
        out = np.empty(out_shape, dtype=data.dtype)

        for i in range(n_groups):
            out[i] = reducer(data[group_index == i], axis=0)

        return out

    @check_supported_compute_frequencies
    def txx(
        self,
        compute_fq:     str,
        tasmax_data:    np.ndarray,
        group_index:    np.ndarray | None = None,
    ):
        """Maximum of daily maximum temperature aggregated by period."""

        txx = self._aggregate_by_group(tasmax_data, group_index, np.max)
        logger.debug(f"Computed TXx with shape {txx.shape}")

        return txx

    @check_supported_compute_frequencies
    def txn(
        self,
        compute_fq:     str,
        tasmax_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Minimum of daily maximum temperature aggregated by period."""

        txn = self._aggregate_by_group(tasmax_data, group_index, np.min)
        logger.debug(f"Computed TXn with shape {txn.shape}")

        return txn

    @check_supported_compute_frequencies
    def tnx(
        self,
        compute_fq:     str,
        tasmin_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Maximum of daily minimum temperature aggregated by period."""

        tnx = self._aggregate_by_group(tasmin_data, group_index, np.max)
        logger.debug(f"Computed TNx with shape {tnx.shape}")

        return tnx

    @check_supported_compute_frequencies
    def tnn(
        self,
        compute_fq:     str,
        tasmin_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Minimum of daily minimum temperature aggregated by period."""

        tnn = self._aggregate_by_group(tasmin_data, group_index, np.min)
        logger.debug(f"Computed TNn with shape {tnn.shape}")

        return tnn

    @check_supported_compute_frequencies
    def dtr(
        self,
        compute_fq:     str,
        tasmax_data:    np.ndarray,
        tasmin_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Diurnal temperature range aggregated by period."""

        dtr_daily = tasmax_data - tasmin_data
        dtr = self._aggregate_by_group(dtr_daily, group_index, np.mean)
        logger.debug(f"Computed DTR with shape {dtr.shape}")

        return dtr

    @check_supported_compute_frequencies
    def fd(
        self,
        compute_fq:     str,
        tasmin_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Frost days (number of days with Tmin < 0°C) for year"""

        frost_days_daily = (tasmin_data < 0).astype(int)
        fd = self._aggregate_by_group(frost_days_daily, group_index, np.sum)
        logger.debug(f"Computed FD with shape {fd.shape}")

        return fd

    @check_supported_compute_frequencies
    def su(
        self,
        compute_fq:     str,
        tasmax_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Summer days (number of days with Tmax > 25°C) for year"""

        summer_days_daily = (tasmax_data > 25).astype(int)
        su = self._aggregate_by_group(summer_days_daily, group_index, np.sum)
        logger.debug(f"Computed SU with shape {su.shape}")

        return su

    @check_supported_compute_frequencies
    def id(
        self,
        compute_fq:     str, # passed for decorator to check, but not used
        tasmax_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Ice days (number of days with Tmax < 0°C) for year"""

        ice_days_daily = (tasmax_data < 0).astype(int)
        iday = self._aggregate_by_group(ice_days_daily, group_index, np.sum)
        logger.debug(f"Computed ID with shape {iday.shape}")

        return iday


    @check_supported_compute_frequencies
    def tr(
        self,
        compute_fq:     str,
        tasmin_data:    np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Tropical nights (number of days with Tmin > 20°C) for year"""

        tropical_nights_daily = (tasmin_data > 20).astype(int)
        tr = self._aggregate_by_group(tropical_nights_daily, group_index, np.sum)
        logger.debug(f"Computed TR with shape {tr.shape}")

        return tr


    @check_supported_compute_frequencies
    def gsl(
        self,
        compute_fq:     str,
        tas_data:       np.ndarray,
        group_index:    np.ndarray | None = None
    ):
        """Growing season length (number of days with Tmin > 5°C) for year"""
        pass

    # ---
    # precipitation
    # ---

    @check_supported_compute_frequencies
    def rx1day(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Monthly maximum 1-day precipitation
        """

        rx1day = self._aggregate_by_group(pr_data, group_index, np.max)
        logger.debug(f"Computed rx1day with shape {rx1day.shape}")

        return rx1day

    # TODO: make rxnday more general with n_days parameter
    # I don't think the rolling sum is done correctly here.
    @check_supported_compute_frequencies
    def rx5day(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Monthly maximum 5-day precipitation
        """

        # Compute rolling 5-day sums
        n_time = pr_data.shape[0]
        rolling_sums = np.empty((n_time - 4,) + pr_data.shape[1:], dtype=pr_data.dtype)
        for i in range(n_time - 4):
            rolling_sums[i] = pr_data[i:i+5].sum(axis=0)

        # Adjust group_index to match rolling_sums time dimension
        if group_index is not None:
            adjusted_group_index = group_index[2:-2]  # Centered adjustment
        else:
            adjusted_group_index = None

        rx5day = self._aggregate_by_group(rolling_sums, adjusted_group_index, np.max)
        logger.debug(f"Computed rx5day with shape {rx5day.shape}")

        return rx5day

    @check_supported_compute_frequencies
    def sdii(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Simple daily intensity index (SDII): total precipitation on wet days
        divided by number of wet days.
        A wet day is defined as a day with precipitation >= threshold (e.g., 1 mm).
        """

        wet_days = pr_data >= 1.0
        total_precip = self._aggregate_by_group(pr_data * wet_days, group_index, np.sum)
        num_wet_days = self._aggregate_by_group(wet_days.astype(int), group_index, np.sum)

        # Avoid division by zero
        sdii = np.where(num_wet_days > 0, total_precip / num_wet_days, 0)
        logger.debug(f"Computed SDII with shape {sdii.shape}")

        return sdii

    @check_supported_compute_frequencies
    def rnnmm(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        threshold:      float,
        group_index:    np.ndarray
    ):
        """
        Number of days with precipitation >= threshold (e.g., 1 mm).
        """

        wet_days = pr_data >= threshold
        rnnmm = self._aggregate_by_group(wet_days.astype(int), group_index, np.sum)
        logger.debug(f"Computed R{threshold}mm with shape {rnnmm.shape}")

        return rnnmm

    @check_supported_compute_frequencies
    def r1mm(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Annual total wet-day precipitation (sum of daily precipitation on days
        with precipitation >= 1 mm).
        """
        r1mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=1.0,
            group_index=group_index,
        )

        return r1mm
    
    @check_supported_compute_frequencies
    def r10mm(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Annual total precipitation on days with precipitation >= 10 mm.
        """
        r10mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=10.0,
            group_index=group_index,
        )

        return r10mm
    
    @check_supported_compute_frequencies
    def r20mm(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Annual total precipitation on days with precipitation >= 20 mm.
        """
        r20mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=20.0,
            group_index=group_index,
        )

        return r20mm


    @check_supported_compute_frequencies
    def prcptot(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray
    ):
        """
        Annual sum of precipitation on days with precipitation >= 1 mm.
        """
        wet_days = pr_data >= 1.0
        prcptot = self._aggregate_by_group(pr_data * wet_days, group_index, np.sum)
        logger.debug(f"Computed PRCPTOT with shape {prcptot.shape}")

        return prcptot
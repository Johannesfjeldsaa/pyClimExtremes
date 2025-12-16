# compute_backends/python_backend.py
from functools import wraps
from typing import Callable

import numpy as np
from netCDF4 import num2date
from scipy.signal import convolve # use signal convolve for n-dim arrays

from general_backend.logging.setup_logging import get_logger
from reversclim.utils.preprocessing.variables.extremes.compute_backend.backend_registry import (
    register_backend,
)

logger = get_logger(__name__)


supported_compute_frequencies = ["mon", "yr"]
scf_str = ", ".join(supported_compute_frequencies)


def check_supported_compute_frequencies(
    func: Callable
) -> Callable:
    """Decorator to check if compute_fq is supported."""

    @wraps(func)
    def wrapper(self, compute_fq, *args, **kwargs):
        if compute_fq not in supported_compute_frequencies:
            err_msg = (
                f"Unsupported compute_fq '{compute_fq}', expected one of: "
                f"{scf_str}"
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
        compute_fq: str,
        time_array: np.ndarray,
        time_units: str,
        calendar: str,
        group_index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute output time coordinate using mean-of-period timestamps.

        If `group_index` is provided (the `inv` array from `np.unique`), it is
        reused to avoid recomputing grouping. Otherwise grouping is derived
        from the time array.
        """
        if time_array is None or time_array.size == 0:
            return np.array([])

        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.fromiter(
            (d.year for d in dates), dtype=int, count=len(time_array)
        )

        if compute_fq == "mon":
            months = np.fromiter(
                (d.month for d in dates), dtype=int, count=len(time_array)
            )
            group_key = years * 12 + (months - 1)
        elif compute_fq == "yr":
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
        years = np.fromiter(
            (d.year for d in dates), dtype=int, count=len(time_array)
        )

        if compute_fq == "mon":
            months = np.fromiter(
                (d.month for d in dates), dtype=int, count=len(time_array)
            )
            group_key = years * 12 + (months - 1)
        elif compute_fq == "yr":
            group_key = years
        else:
            err_msg = (
                f"Unsupported compute_fq '{compute_fq}', expected one of",
                scf_str,
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

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
            # assuming axis 0 is time we reduce over that axis
            out[i] = reducer(data[group_index == i], axis=0)

        return out


    def _count_consecutive_days(
        self,
        bool_array: np.ndarray
    ) -> np.ndarray:
        # Initialize the output array
        cumulative = np.zeros_like(bool_array, dtype=np.int32)

        # Initialize a tracker for the current run length for each spatial point
        current_run = np.zeros(bool_array.shape[1:], dtype=np.int32)

        # Iterate over the time dimension
        for t in range(bool_array.shape[0]):
            # Logic:
            # If bool_array[t] is True (1): current_run becomes current_run + 1
            # If bool_array[t] is False (0): current_run becomes 0
            # This can be vectorized as: (current_run + 1) * bool_array[t]

            current_run = (current_run + 1) * bool_array[t]
            cumulative[t] = current_run

        return cumulative


    @check_supported_compute_frequencies
    def txx(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """Maximum of daily maximum temperature aggregated by period."""

        txx = self._aggregate_by_group(tasmax_data, group_index, np.max)
        logger.debug(f"Computed TXx with shape {txx.shape}")

        return txx

    @check_supported_compute_frequencies
    def txn(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """Minimum of daily maximum temperature aggregated by period."""

        txn = self._aggregate_by_group(tasmax_data, group_index, np.min)
        logger.debug(f"Computed TXn with shape {txn.shape}")

        return txn


    @check_supported_compute_frequencies
    def tnx(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """Maximum of daily minimum temperature aggregated by period."""

        tnx = self._aggregate_by_group(tasmin_data, group_index, np.max)
        logger.debug(f"Computed TNx with shape {tnx.shape}")

        return tnx

    @check_supported_compute_frequencies
    def tnn(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray ,
        fixed_threshold:    None = None # not used
    ):
        """Minimum of daily minimum temperature aggregated by period."""

        tnn = self._aggregate_by_group(tasmin_data, group_index, np.min)
        logger.debug(f"Computed TNn with shape {tnn.shape}")

        return tnn

    @check_supported_compute_frequencies
    def dtr(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """Diurnal temperature range aggregated by period."""

        dtr_daily = tasmax_data - tasmin_data
        dtr = self._aggregate_by_group(dtr_daily, group_index, np.mean)
        logger.debug(f"Computed DTR with shape {dtr.shape}")

        return dtr

    @check_supported_compute_frequencies
    def fd(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """Frost days (number of days with Tmin < 0°C or 273.15 K) for year

        Parameters
        ----------
        freeze_threshold : float
            Temperature threshold, use 0°C if input data is in degrees Celsius,
            or 273.15 K if input data is in Kelvin.
        """

        frost_days_daily = (tasmin_data < fixed_threshold).astype(int)
        fd = self._aggregate_by_group(frost_days_daily, group_index, np.sum)
        logger.debug(f"Computed FD with shape {fd.shape}")

        return fd

    @check_supported_compute_frequencies
    def su(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        threshold:          float
    ):
        """Summer days (number of days with Tmax > 25°C or 298.15 K) for year

        Parameters
        ----------
        threshold : float
            Temperature threshold, default definition is 25°C if input data is
            in degrees Celsius, or 298.15 K if input data is in Kelvin.
        """

        summer_days_daily = (tasmax_data > threshold).astype(int)
        su = self._aggregate_by_group(summer_days_daily, group_index, np.sum)
        logger.debug(
            "Computed SU with threshold=%s and shape %s",
            threshold, su.shape
        )

        return su

    @check_supported_compute_frequencies
    def id(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """Ice days (number of days with Tmax < 0°C or 273.15 K) for year

        Parameters
        ----------
        fixed_threshold : float
            Temperature threshold, use 0°C if input data is in degrees Celsius,
            or 273.15 K if input data is in Kelvin.
        """

        ice_days_daily = (tasmax_data < fixed_threshold).astype(int)
        iday = self._aggregate_by_group(ice_days_daily, group_index, np.sum)
        logger.debug(f"Computed ID with shape {iday.shape}")

        return iday

    @check_supported_compute_frequencies
    def tr(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """Tropical nights (number of days with Tmin > 20°C or 293.15 K)
        for year

        Parameters
        ----------
        fixed_threshold : float
            Temperature threshold, default definition is 20°C if input data is
            in degrees Celsius, or 293.15 K if input data is in Kelvin.
        """

        tropical_nights_daily = (tasmin_data > fixed_threshold).astype(int)
        tr = self._aggregate_by_group(
            tropical_nights_daily, group_index, np.sum
        )
        logger.debug(f"Computed TR with shape {tr.shape}")

        return tr

    @check_supported_compute_frequencies
    def gsl(
        self,
        compute_fq:         str,
        tas_data:           np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """Growing season length (number of days with Tmin > 5°C) for year"""
        pass


    # ---
    # precipitation
    # ---


    @check_supported_compute_frequencies
    def cdd(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Consecutive dry days (cddETCCDI) within period.

        Let RRij be the daily precipitation amount on day i in period j.
        Count the largest number of consecutive days where RRij < 1mm (or
        1/86400 kg m-2) within period j.
        """

        # indicate which days are dry
        is_dry = pr_data < fixed_threshold

        # count consecutive dry days
        cdd = self._count_consecutive_days(is_dry)

        max_cdd = self._aggregate_by_group(cdd, group_index, np.max)

        logger.debug(f"Computed CDD with shape {max_cdd.shape}")

        return max_cdd

    @check_supported_compute_frequencies
    def cwd(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Consecutive wet days (cwdETCCDI).

        Let RRij be the daily precipitation amount on day i in period j.
        Count the largest number of consecutive days where RRij >= 1mm (or
        1/86400 kg m-2) within period j.
        """

        # indicate which days are wet
        is_wet = pr_data >= fixed_threshold

        # count consecutive wet days
        cwd = self._count_consecutive_days(is_wet)

        max_cwd = self._aggregate_by_group(cwd, group_index, np.max)

        logger.debug(f"Computed CWD with shape {max_cwd.shape}")

        return max_cwd


    @check_supported_compute_frequencies
    def rxnday(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold:          int
    ):
        """
        Maximum n-day precipitation over period.
        Let RR_kj be the precipitation amount for n-day period ending at day
        k in period j. Then,
            RXnday_j = max(RR_kj) for all k in period j

        Parameters
        ----------
        n_days : int
            Number of days over which to compute maximum precipitation sum.
        """
         # we have to convolve with a window of 5 days to get the rolling sums
        window_size = threshold
        if window_size < 1:
            err_msg = f"Invalid window_size {window_size}, must be >= 1"
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        elif window_size == 1:
            # rx1day is just the maximum daily precipitation
            rxnday = self._aggregate_by_group(pr_data, group_index, np.max)
        else:
            if pr_data.shape[0] < window_size:
                err_msg = (
                    f"Insufficient time steps ({pr_data.shape[0]}) "
                    f"for computing rx5day requiring at least {window_size}"
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)


            # To compute the rolling sum over the time dimension (axis 0), we
            # can take the cumulative sum over the array. Requiring all time steps
            # to be observed the valid indices are cumsum[window_size - 1 :].
            # We then subtract the cumsum shifted by window_size to get the rolling
            # sum. This is equivalent to convolving with a window of ones of size
            # 5, but more efficient for large arrays.
            # Alternatively, we could use scipy.signal.convolve, with mode='valid',
            # and kernel = np.ones(
            #   (window_size,) + (1,) * (pr_data.ndim - 1),
            #   dtype=pr_data.dtype
            # ) but the implementation below was faster for test on large arrays.
            cumsum = np.cumsum(pr_data, axis=0)
            rolling_sum = (
                cumsum[window_size - 1 :] -
                np.concatenate(
                    (np.zeros((1,) + pr_data.shape[1:], dtype=pr_data.dtype),
                    cumsum[:-window_size]),
                    axis=0
                )
            )

            # the shape of rolling_sum is (time - window_size + 1, lat, lon)
            # therefor we need to adjust group_index accordingly by
            # shifing it by window_size - 1
            adjusted_group_index = group_index[window_size - 1 :]

            # find group-wise maximum of the rolling sum result
            rxnday = self._aggregate_by_group(
                rolling_sum, adjusted_group_index, np.max
            )

        logger.debug(
            "Computed rx%sday with shape %s",
            window_size, rxnday.shape
        )

        return rxnday


    @check_supported_compute_frequencies
    def rx1day(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """
        Maximum 1-day precipitation over period
        """

        rx1day = self.rxnday(
            compute_fq=compute_fq,
            pr_data=pr_data,
            group_index=group_index,
            threshold=1
        )

        return rx1day


    @check_supported_compute_frequencies
    def rx5day(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    None = None # not used
    ):
        """Maximum 5-day precipitation over period

        Let RR_kj be the precipitation amount for 5-day period ending at day
        k in period j. Then,
            RX5day_j = max(RR_kj) for all k in period j
        """
        rx5day = self.rxnday(
            compute_fq=compute_fq,
            pr_data=pr_data,
            group_index=group_index,
            threshold=5
        )

        return rx5day

    @check_supported_compute_frequencies
    def sdii(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Simple daily intensity index (SDII): total precipitation on wet days
        divided by number of wet days. A wet day is defined as a day with
        precipitation >= 1 mm or 1/86400 kg m-2.

        Parameters
        ----------
        threshold : float
            Precipitation threshold to define a wet day.
        """

        wet_days = pr_data >= fixed_threshold
        total_precip = self._aggregate_by_group(
            pr_data * wet_days, group_index, np.sum
        )
        num_wet_days = self._aggregate_by_group(
            wet_days.astype(int), group_index, np.sum
        )

        # Avoid division by zero
        sdii = np.where(num_wet_days > 0, total_precip / num_wet_days, 0)
        logger.debug(f"Computed SDII with shape {sdii.shape}")

        return sdii

    @check_supported_compute_frequencies
    def rnnmm(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray,
        threshold:      float
    ):
        """
        Number of days with precipitation >= threshold (e.g., 1 mm).
        """

        wet_days = pr_data >= threshold
        rnnmm = self._aggregate_by_group(
            wet_days.astype(int), group_index, np.sum
        )
        logger.debug(f"Computed R{threshold}mm with shape {rnnmm.shape}")

        return rnnmm

    @check_supported_compute_frequencies
    def r1mm(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Annual total wet-day precipitation (sum of daily precipitation on days
        with precipitation >= 1 mm or 1/86400 kg m-2).
        """
        r1mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=fixed_threshold,
            group_index=group_index,
        )

        return r1mm

    @check_supported_compute_frequencies
    def r10mm(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Annual total precipitation on days with precipitation >= 10 mm
        or 10/86400 kg m-2.
        """
        r10mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=fixed_threshold,
            group_index=group_index,
        )

        return r10mm

    @check_supported_compute_frequencies
    def r20mm(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float
    ):
        """
        Annual total precipitation on days with precipitation >= 20 mm
        or 20/86400 kg m-2.
        """
        r20mm = self.rnnmm(
            compute_fq=compute_fq,
            pr_data=pr_data,
            threshold=fixed_threshold,
            group_index=group_index,
        )

        return r20mm

    @check_supported_compute_frequencies
    def prcptot(
        self,
        compute_fq:     str,
        pr_data:        np.ndarray,
        group_index:    np.ndarray,
        fixed_threshold:      float
    ):
        """
        Annual sum of precipitation on days with precipitation >= 1 mm (or
        1/86400 kg m-2).
        """
        wet_days = pr_data >= fixed_threshold
        prcptot = self._aggregate_by_group(
            pr_data * wet_days, group_index, np.sum
        )
        logger.debug(f"Computed PRCPTOT with shape {prcptot.shape}")

        return prcptot

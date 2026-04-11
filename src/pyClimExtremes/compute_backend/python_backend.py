# compute_backends/python_backend.py
from functools import partial, wraps
from typing import Any, Callable

import numpy as np
from numba import cuda
from netCDF4 import num2date

from pyClimExtremes.logging.setup_logging import get_logger
from pyClimExtremes.compute_backend.backend_registry import register_backend
from pyClimExtremes.compute_backend.python_backend_helper_methods import (
    temperature_quantiles_estimation,
    precipitation_quantiles_estimation,
    growing_season_length,
    max_spell_length_by_group
)

logger = get_logger(__name__)

supported_compute_frequencies = ["mon", "yr"]
scf_str = ", ".join(supported_compute_frequencies)


def _normalize_dates_sequence(dates: Any) -> list[Any]:
    """Convert num2date output to a concrete sequence for downstream iteration."""
    if isinstance(dates, np.ndarray):
        return dates.reshape(-1).tolist()
    if np.isscalar(dates):
        return [dates]
    return list(dates)


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

def _cuda_available() -> bool:
    """Check CUDA availability by probing the driver directly.

    ``cuda.is_available()`` can return False even when the GPU is present if
    CUDA toolkit libraries are not on the dynamic linker path. Calling
    ``cuda.detect()`` (or listing devices) actually initialises the driver and
    is a more reliable check.
    """
    try:
        return len(cuda.gpus) > 0
    except Exception:
        return False


@register_backend("python")
class PythonBackend:
    """Python backend for ETCCDI index calculations."""

    def __init__(self, use_cuda_if_available: bool = True):
        self.use_cuda = use_cuda_if_available and _cuda_available()
        self._temperature_quantiles_estimation = partial(
            temperature_quantiles_estimation, use_cuda=self.use_cuda
        )
        self._precipitation_quantiles_estimation = precipitation_quantiles_estimation
        self._growing_season_length = growing_season_length
        self._max_spell_length_by_group = max_spell_length_by_group


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

        dates = _normalize_dates_sequence(
            num2date(time_array, units=time_units, calendar=calendar)
        )
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

        dates = _normalize_dates_sequence(
            num2date(time_array, units=time_units, calendar=calendar)
        )
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
            out[i] = reducer(data[group_index == i], axis=0)

        return out

    # ================================================================ #
    # === Quantiles ================================================== #
    # ================================================================ #
    # These indices are used to estimate quantile thresholds from daily
    # data. Then the thresholds are applied to compute the impact
    # indices that are computed relative to these quantiles, e.g.
    # tn10p, tn90p, tx10p, tx90p for temperature and r95p, r99p for
    # precipitation.

    # --- temperature quantiles ---
    @check_supported_compute_frequencies
    def tn_qXXp(
        self,
        compute_fq,
        tasmin_data,
        group_index,
        quantile,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute one or more TN percentile thresholds."""
        return self._temperature_quantiles_estimation(
            temp_data=tasmin_data,
            quantile=quantile,
            base_period_mask=base_period_mask,
            group_index=group_index,
            time_array=time_array,
            time_units=time_units,
            calendar=calendar,
            window_size=window_size,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )

    @check_supported_compute_frequencies
    def tn_q10p(
        self,
        compute_fq,
        tasmin_data,
        group_index,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute TN10p thresholds (10th percentile of daily minimum temp)."""
        return self.tn_qXXp(
            compute_fq=compute_fq,
            tasmin_data=tasmin_data,
            group_index=group_index,
            quantile=0.1,
            base_period_mask=base_period_mask,
            time_array=time_array,
            time_units=time_units,
            calendar=calendar,
            window_size=window_size,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )

    @check_supported_compute_frequencies
    def tn_q90p(
        self,
        compute_fq,
        tasmin_data,
        group_index,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute TN90p thresholds (90th percentile of daily minimum temp)."""
        return self.tn_qXXp(
            compute_fq=compute_fq,
            tasmin_data=tasmin_data,
            group_index=group_index,
            quantile=0.9,
            base_period_mask=base_period_mask,
            time_array=time_array,
            time_units=time_units,
            calendar=calendar,
            window_size=window_size,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )

    @check_supported_compute_frequencies
    def tx_qXXp(
        self,
        compute_fq,
        tasmax_data,
        group_index,
        quantile,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute one or more TX percentile thresholds."""
        return self._temperature_quantiles_estimation(
                temp_data=tasmax_data,
                quantile=quantile,
                base_period_mask=base_period_mask,
                group_index=group_index,
                time_array=time_array,
                time_units=time_units,
                calendar=calendar,
                window_size=window_size,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )

    @check_supported_compute_frequencies
    def tx_q10p(
        self,
        compute_fq,
        tasmax_data,
        group_index,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute TX10p thresholds (10th percentile of daily maximum temp)."""
        return self.tx_qXXp(
            compute_fq=compute_fq,
            tasmax_data=tasmax_data,
            group_index=group_index,
            quantile=0.1,
                base_period_mask=base_period_mask,
                time_array=time_array,
                time_units=time_units,
                calendar=calendar,
                window_size=window_size,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )

    @check_supported_compute_frequencies
    def tx_q90p(
        self,
        compute_fq,
        tasmax_data,
        group_index,
        base_period_mask,
        time_array,
        time_units,
        calendar,
        window_size,
        bootstrap_samples,
        random_seed=None,
    ):
        """Compute TX90p thresholds (90th percentile of daily maximum temp)."""
        return self.tx_qXXp(
            compute_fq=compute_fq,
            tasmax_data=tasmax_data,
            group_index=group_index,
            quantile=0.9,
                base_period_mask=base_period_mask,
                time_array=time_array,
                time_units=time_units,
                calendar=calendar,
                window_size=window_size,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )


    # --- precipitation quantiles ---

    @check_supported_compute_frequencies
    def pr_qXXp(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        quantile,
        base_period_mask:   np.ndarray,
        wet_day_threshold:  float,
    ) -> np.ndarray:
        """Compute one or more wet-day precipitation percentile thresholds."""
        return self._precipitation_quantiles_estimation(
            pr_data=pr_data,
            quantile=quantile,
            base_period_mask=base_period_mask,
            group_index=group_index,
            wet_day_threshold=wet_day_threshold,
        )

    @check_supported_compute_frequencies
    def pr_q95p(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        wet_day_threshold:  float,
    ) -> np.ndarray:
        """Wet-day 95th percentile precipitation threshold per grid point."""
        return self.pr_qXXp(
            compute_fq=compute_fq,
            pr_data=pr_data,
            group_index=group_index,
            quantile=quantile,
            base_period_mask=base_period_mask,
            wet_day_threshold=wet_day_threshold,
        )

    @check_supported_compute_frequencies
    def pr_q99p(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        wet_day_threshold:  float,
    ) -> np.ndarray:
        """Wet-day 99th percentile precipitation threshold per grid point."""
        return self.pr_qXXp(
            compute_fq=compute_fq,
            pr_data=pr_data,
            group_index=group_index,
            quantile=quantile,
            base_period_mask=base_period_mask,
            wet_day_threshold=wet_day_threshold,
        )


    # ================================================================ #
    # --- Temperature Indices ======================================== #
    # ================================================================ #
    # These are impact indices using temperature data directly or
    # applying quantile thresholds estimated from daily data.

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
        fixed_threshold:    float,
        time_array:         np.ndarray | None = None,
        time_units:         str | None = None,
        calendar:           str | None = None,
        lat:                np.ndarray | None = None,
    ):
        if compute_fq != "yr":
            err_msg = f"GSL only supports annual frequency 'yr', got '{compute_fq}'"
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        if time_array is None or time_units is None or calendar is None:
            err_msg = (
                "GSL requires time_array, time_units, and calendar parameters. "
                "These must be passed from the index computation pipeline."
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        dates = _normalize_dates_sequence(
            num2date(time_array, units=time_units, calendar=calendar)
        )
        years = np.fromiter((d.year for d in dates), dtype=int, count=len(time_array))
        months = np.fromiter((d.month for d in dates), dtype=int, count=len(time_array))

        n_groups = int(group_index.max()) + 1 if group_index.size else 0
        out = np.full((n_groups,) + tas_data.shape[1:], 0.0, dtype=np.float32)

        if lat is None:
            raise ValueError("Latitude array is required for GSL computation to determine hemispheres.")

        nh_mask = lat >= 0
        sh_mask = lat < 0

        gsl_nh = self._growing_season_length(
            tas_data=tas_data,
            group_index=group_index,
            dates=dates,
            fixed_threshold=fixed_threshold,
            first_half_months=(1, 2, 3, 4, 5, 6),
        )
        out[:, nh_mask, :] = gsl_nh[:, nh_mask, :]

        valid_years = (years.min(), years.max())
        years_gsl = years - (months < 7).astype(int)

        inset = years_gsl >= valid_years[0]
        tas_sh = tas_data[inset]
        dates_sh = np.asarray(dates)[inset]
        years_gsl_sh = years_gsl[inset]

        unique_gsl_years, inv_gsl = np.unique(years_gsl_sh, return_inverse=True)

        gsl_sh = self._growing_season_length(
            tas_data=tas_sh,
            group_index=inv_gsl,
            dates=dates_sh,
            fixed_threshold=fixed_threshold,
            first_half_months=(7, 8, 9, 10, 11, 12),
        )

        for i, yg in enumerate(unique_gsl_years):
            year_idx = int(yg) - years.min()
            if 0 <= year_idx < n_groups:
                out[year_idx, sh_mask, :] = gsl_sh[i][sh_mask, :]

        logger.debug(f"Computed GSL with shape {out.shape}")
        return out

    @check_supported_compute_frequencies
    def tn10p(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Number of days when daily minimum temperature < 10th percentile (TN10p).
        """
        # Identify days below threshold
        exceed_mask = tasmin_data < threshold_array[np.newaxis, ...]

        # Sum days below threshold
        tn10p = self._aggregate_by_group(exceed_mask, group_index, np.sum)

        logger.debug(f"Computed TN10p with shape {tn10p.shape}")
        return tn10p

    @check_supported_compute_frequencies
    def tn90p(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Number of days when daily minimum temperature > 90th percentile (TN90p).
        """
        # Identify days above threshold
        exceed_mask = tasmin_data > threshold_array[np.newaxis, ...]

        # Sum days above threshold
        tn90p = self._aggregate_by_group(exceed_mask, group_index, np.sum)

        logger.debug(f"Computed TN90p with shape {tn90p.shape}")
        return tn90p

    @check_supported_compute_frequencies
    def tx10p(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Number of days when daily maximum temperature < 10th percentile (TX10p).
        """
        # Identify days below threshold
        exceed_mask = tasmax_data < threshold_array[np.newaxis, ...]

        # Sum days below threshold
        tx10p = self._aggregate_by_group(exceed_mask, group_index, np.sum)

        logger.debug(f"Computed TX10p with shape {tx10p.shape}")
        return tx10p

    @check_supported_compute_frequencies
    def tx90p(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Number of days when daily maximum temperature > 90th percentile (TX90p).
        """
        # Identify days above threshold
        exceed_mask = tasmax_data > threshold_array[np.newaxis, ...]

        # Sum days above threshold
        tx90p = self._aggregate_by_group(exceed_mask, group_index, np.sum)

        logger.debug(f"Computed TX90p with shape {tx90p.shape}")
        return tx90p

    # ================================================================ #
    # === Precipitation Indices ====================================== #
    # ================================================================ #
    # These are impact indices using precipitation data directly or
    # applying quantile thresholds estimated from daily data.

    @check_supported_compute_frequencies
    def cdd(
        self,
        compute_fq: str,
        pr_data: np.ndarray,
        group_index: np.ndarray,
        fixed_threshold: float,
        spells_can_span_groups: bool,
        mask: np.ndarray | None = None,
    ):
        """
        Consecutive dry days (cddETCCDI) within period.

        Let RRij be the daily precipitation amount on day i in period j.
        Count the largest number of consecutive days where RRij < 1mm (or
        1/86400 kg m-2) within period j.
        """
        # Handle NaN values - exclude from consecutive counting
        valid_data = ~np.isnan(pr_data)
        is_dry = (pr_data < fixed_threshold) & valid_data

        max_cdd, has_spell_end = self._max_spell_length_by_group(
            is_dry,
            group_index=group_index,
            spells_can_span_groups=spells_can_span_groups,
        )

        result = max_cdd.astype(np.float32, copy=False)
        if spells_can_span_groups:
            all_dry_in_group = self._aggregate_by_group(is_dry, group_index, np.all)
            result = np.where(all_dry_in_group & ~has_spell_end, np.nan, result)

        # Apply data quality mask (like R's na.mask)
        if mask is not None:
            result = result * mask
            result[mask == 0] = np.nan  # Explicitly set to NA for quality mask

        logger.debug(f"Computed CDD with shape {result.shape}")
        return result

    @check_supported_compute_frequencies
    def cwd(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        fixed_threshold:    float,
        spells_can_span_groups: bool,
    ):
        """
        Consecutive wet days (cwdETCCDI).

        Let RRij be the daily precipitation amount on day i in period j.
        Count the largest number of consecutive days where RRij >= 1mm (or
        1/86400 kg m-2) within period j.
        """

        # indicate which days are wet
        is_wet = (pr_data >= fixed_threshold) & ~np.isnan(pr_data)

        max_cwd, has_spell_end = self._max_spell_length_by_group(
            is_wet,
            group_index=group_index,
            spells_can_span_groups=spells_can_span_groups,
        )

        result = max_cwd.astype(np.float32, copy=False)
        if spells_can_span_groups:
            all_wet_in_group = self._aggregate_by_group(is_wet, group_index, np.all)
            result = np.where(all_wet_in_group & ~has_spell_end, np.nan, result)

        logger.debug(f"Computed CWD with shape {result.shape}")

        return result


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

    @check_supported_compute_frequencies
    def r95p(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Sum of precipitation on very wet days (R95p).

        Total precipitation on days with daily precipitation > 95th percentile.
        """
        # Identify days exceeding threshold
        exceed_mask = pr_data > threshold_array[np.newaxis, ...]

        # Sum precipitation on wet days
        r95p = self._aggregate_by_group(pr_data * exceed_mask, group_index, np.sum)

        logger.debug(f"Computed R95p with shape {r95p.shape}")
        return r95p

    @check_supported_compute_frequencies
    def r99p(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Sum of precipitation on extremely wet days (R99p).

        Total precipitation on days with daily precipitation > 99th percentile.
        """
        # Identify days exceeding threshold
        exceed_mask = pr_data > threshold_array[np.newaxis, ...]

        # Sum precipitation on wet days
        r99p = self._aggregate_by_group(pr_data * exceed_mask, group_index, np.sum)

        logger.debug(f"Computed R99p with shape {r99p.shape}")
        return r99p

    @check_supported_compute_frequencies
    def r95p_tot(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
        fixed_threshold:    float,
    ):
        """Percentage contribution of very wet days to total
        precipitation on wet days(R95pTOT).

        Contribution: 100 × (R95p / PRCPTOT)

        Parameters
        ----------
        threshold_array : np.ndarray
            Pre-computed 95th percentile threshold from QuantileThresholdIndex.
        """
        # Compute R95p (sum on wet days)
        exceed_mask = pr_data > threshold_array[np.newaxis, ...]
        r95p = self._aggregate_by_group(pr_data * exceed_mask, group_index, np.sum)

        # Compute PRCPTOT (total precipitation on wet days, >= 1mm)
        wet_mask = pr_data >= fixed_threshold
        prcptot = self._aggregate_by_group(pr_data * wet_mask, group_index, np.sum)

        # Avoid division by zero
        r95p_tot = np.full(prcptot.shape, np.nan, dtype=float)
        np.divide(
            100.0 * r95p,
            prcptot,
            out=r95p_tot,
            where=prcptot > 0,
        )

        logger.debug(f"Computed R95pTOT with shape {r95p_tot.shape}")
        return r95p_tot

    @check_supported_compute_frequencies
    def r99p_tot(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
        fixed_threshold:    float,
    ):
        """Percentage contribution of extremely wet days to total
        precipitation on wet days (R99pTOT).

        Contribution: 100 × (R99p / PRCPTOT)

        Parameters
        ----------
        threshold_array : np.ndarray
            Pre-computed 99th percentile threshold from QuantileThresholdIndex.
        """
        # Compute R99p (sum on wet days)
        exceed_mask = pr_data > threshold_array[np.newaxis, ...]
        r99p = self._aggregate_by_group(pr_data * exceed_mask, group_index, np.sum)

        # Compute PRCPTOT (total precipitation on wet days, >= 1mm)
        wet_mask = pr_data >= fixed_threshold
        prcptot = self._aggregate_by_group(pr_data * wet_mask, group_index, np.sum)

        # Avoid division by zero
        r99p_tot = np.full(prcptot.shape, np.nan, dtype=float)
        np.divide(
            100.0 * r99p,
            prcptot,
            out=r99p_tot,
            where=prcptot > 0,
        )

        logger.debug(f"Computed R99pTOT with shape {r99p_tot.shape}")
        return r99p_tot


# compute_backends/python_backend.py
from functools import wraps
from typing import Callable

import numpy as np
from netCDF4 import num2date
from scipy.signal import convolve # use signal convolve for n-dim arrays

from pyClimExtremes.logging.setup_logging import get_logger
from pyClimExtremes.compute_backend.backend_registry import register_backend

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
            out[i] = reducer(data[group_index == i], axis=0)

        return out


    def _count_consecutive_days(
        self,
        bool_array: np.ndarray,
        group_index: np.ndarray | None = None,
        spells_can_span_groups: bool = False,
    ) -> np.ndarray:
        """
        Count consecutive True values across time dimension.

        Parameters:
        -----------
        bool_array : np.ndarray
            Boolean array with shape (time, ...)
        group_index : np.ndarray | None, default None
            Optional array mapping each time index to a group. If None,
            no group boundary handling is applied.
        spells_can_span_groups : bool, default False  # ← Note: Different default from R
            If False, reset counters at group boundaries. Has no effect
            when group_index is None.

        Returns:
        --------
        cumulative : np.ndarray
            Array of cumulative counts with same shape as bool_array.
        """
        cumulative = np.zeros_like(bool_array, dtype=np.int32)
        current_run = np.zeros(bool_array.shape[1:], dtype=np.int32)

        for t in range(bool_array.shape[0]):
            is_event = bool_array[t]

            # Check for group boundary (skip first timestep)
            if t > 0 and group_index is not None and not spells_can_span_groups:
                group_changed = group_index[t] != group_index[t - 1]
                # Broadcast group_changed to match spatial dims if needed
                if group_changed.ndim == 0:  # scalar
                    current_run = np.where(group_changed, 0, current_run)
                elif group_changed.shape != current_run.shape:
                    # Broadcast to spatial dimensions
                    if current_run.ndim == 2:  # (spatial_x, spatial_y)
                        current_run = np.where(group_changed[..., np.newaxis], 0, current_run)
                    else:
                        current_run = np.where(group_changed, 0, current_run)
                else:
                    current_run = np.where(group_changed, 0, current_run)

            # Update run length
            current_run = (current_run + 1) * is_event
            cumulative[t] = current_run

        return cumulative

    def _max_spell_length_by_group(
        self,
        bool_array: np.ndarray,
        group_index: np.ndarray,
        spells_can_span_groups: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the longest spell ending in each group.

        When `spells_can_span_groups` is True, spells are assigned to the group
        containing the spell end date.

        Parameters
        ----------
        bool_array : np.ndarray
            Boolean array with shape (time, ...)
        group_index : np.ndarray
            Array mapping each time index to a group.
        spells_can_span_groups : bool
            If True, spells can span group boundaries.

        Returns
        -------
        max_spell : np.ndarray
            Array of maximum spell lengths for each group.
        has_spell_end : np.ndarray
            Boolean array indicating if a spell ended in each group.
        """
        n_groups = int(group_index.max()) + 1 if group_index.size else 0
        out_shape = (n_groups,) + tuple(bool_array.shape[1:])
        max_spell = np.zeros(out_shape, dtype=np.int32)
        has_spell_end = np.zeros(out_shape, dtype=bool)
        current_run = np.zeros(bool_array.shape[1:], dtype=np.int32)

        for t in range(bool_array.shape[0]):
            current_group = int(group_index[t])
            in_spell = bool_array[t]

            if (
                t > 0
                and not spells_can_span_groups
                and group_index[t] != group_index[t - 1]
            ):
                current_run[...] = 0

            current_run = np.where(in_spell, current_run + 1, 0)

            is_last_timestep = t == bool_array.shape[0] - 1
            next_group_changes = (
                False if is_last_timestep else group_index[t + 1] != group_index[t]
            )
            next_is_false = False if is_last_timestep else ~bool_array[t + 1]

            if spells_can_span_groups:
                spell_ends = in_spell & (is_last_timestep | next_is_false)
            else:
                spell_ends = in_spell & (is_last_timestep | next_group_changes | next_is_false)

            if np.any(spell_ends):
                ended_lengths = np.where(spell_ends, current_run, 0)
                max_spell[current_group] = np.maximum(
                    max_spell[current_group],
                    ended_lengths,
                )
                has_spell_end[current_group] |= spell_ends

        return max_spell, has_spell_end

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

    def _first_run_start(
        self,
        cumulative_runs: np.ndarray,
        start_date_idx: int,
        end_date_idx: int,
        min_run_length: int = 6,
    ) -> np.ndarray:
        """
        Find the start index of the first run with length >= min_run_length
        in cumulative_runs[start_date_idx:end_date_idx].

        Parameters
        ----------
        cumulative_runs : np.ndarray
            Shape (time, lat, lon) or (time, ...) — cumulative run lengths
        start_date_idx : int
            Start index in time dimension
        end_date_idx : int
            End index in time dimension (exclusive)
        min_run_length : int, optional
            Minimum run length to search for, by default 6

        Returns
        -------
        np.ndarray
            Shape (...) — indices of first run start, or -1 if not found
        """
        if start_date_idx >= end_date_idx:
            return np.full(cumulative_runs.shape[1:], -1, dtype=int)

        window = cumulative_runs[start_date_idx+1:end_date_idx+1] >= min_run_length
        has_run = np.any(window, axis=0)
        first_end_rel = np.argmax(window, axis=0)
        first_end = (start_date_idx + 1) + first_end_rel
        first_start = first_end - (min_run_length - 1)
        return np.where(has_run, first_start, -1)

    def _growing_season_length(
        self,
        tas_data: np.ndarray,
        group_index: np.ndarray,
        dates,
        fixed_threshold: float,
        run_len: int = 6,
        first_half_months: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    ) -> np.ndarray:
        warm = tas_data >= fixed_threshold
        cold = tas_data < fixed_threshold

        warm_runs = self._count_consecutive_days(warm)
        cold_runs = self._count_consecutive_days(cold)

        n_groups = int(group_index.max()) + 1 if group_index.size else 0
        out = np.full(
            (n_groups,) + tas_data.shape[1:],
            0.0,
            dtype=np.float32
        )

        months = np.fromiter((d.month for d in dates), dtype=int, count=len(dates))
        first_half_months_set = set(first_half_months)

        for g in range(n_groups):
            idx = np.where(group_index == g)[0]
            if idx.size == 0:
                continue

            group_months = months[idx]

            is_first_half = np.isin(group_months, list(first_half_months_set))
            first_half = idx[is_first_half]
            second_half = idx[~is_first_half]

            if first_half.size == 0 or second_half.size == 0:
                continue

            start_idx = self._first_run_start(
                warm_runs,
                int(first_half[0]),
                int(first_half[-1]) + 1,
                min_run_length=run_len,
            )

            end_idx = self._first_run_start(
                cold_runs,
                int(second_half[0]),
                int(second_half[-1]) + 1,
                min_run_length=run_len,
            )

            if first_half_months == (1, 2, 3, 4, 5, 6):
                year_end_idx = idx[-1] if idx.size > 0 else -1
            else:
                year_end_idx = first_half[-1] if first_half.size > 0 else -1

            both_found = (start_idx >= 0) & (end_idx >= 0) & (end_idx > start_idx)
            warm_only = (start_idx >= 0) & (end_idx < 0)

            gsl_standard = end_idx - start_idx

            if year_end_idx >= 0:
                gsl_year_round = year_end_idx - start_idx
            else:
                gsl_year_round = 0.0

            gsl = np.where(both_found, gsl_standard,
                    np.where(warm_only, gsl_year_round, 0.0))

            out[g] = gsl.astype(np.float32)

        return out


    @check_supported_compute_frequencies
    def gsl(
        self,
        compute_fq: str,
        tas_data: np.ndarray,
        group_index: np.ndarray,
        fixed_threshold: float,
        time_array: np.ndarray = None,
        time_units: str = None,
        calendar: str = None,
        lat: np.ndarray = None,
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

        dates = num2date(time_array, units=time_units, calendar=calendar)
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


    def _build_day_of_year_sample(
        self,
        calendar_day: int,
        data: np.ndarray,
        time_array: np.ndarray,
        time_units: str,
        calendar: str,
        base_years: np.ndarray | None = None,
        exclude_year: int | None = None,
    ) -> np.ndarray:
        """Extract all values for a given calendar day across specified years.

        Parameters
        ----------
        calendar_day : int
            Day of year (0-364 or 0-365)
        data : np.ndarray
            Shape (time, lat, lon) — daily data
        time_array : np.ndarray
            Time coordinate values
        time_units : str
            Time units string
        calendar : str
            Calendar type
        base_years : np.ndarray | None
            Array of year values to include; if None, use all years
        exclude_year : int | None
            Year to exclude (for leave-one-out bootstrap)

        Returns
        -------
        np.ndarray
            Stacked values from all qualifying years for the calendar day
        """
        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.fromiter((d.year for d in dates), dtype=int, count=len(dates))
        day_of_years = np.fromiter((d.timetuple().tm_yday - 1 for d in dates), dtype=int, count=len(dates))

        # Identify indices matching both calendar day and allowed year range
        day_match = day_of_years == calendar_day
        if base_years is not None:
            year_match = np.isin(years, base_years)
        else:
            year_match = np.ones_like(years, dtype=bool)

        if exclude_year is not None:
            year_match = year_match & (years != exclude_year)

        idx = np.where(day_match & year_match)[0]

        if idx.size == 0:
            return np.array([])

        return data[idx]

    def _compute_percentile_from_window(
        self,
        calendar_day: int,
        data: np.ndarray,
        time_array: np.ndarray,
        time_units: str,
        calendar: str,
        quantile: float,
        window_size: int = 5,
        base_years: np.ndarray | None = None,
        exclude_year: int | None = None,
    ) -> np.ndarray:
        """Compute percentile threshold for a calendar day using rolling window.

        For a given calendar day, extracts samples from a rolling window of days
        (e.g., day ± 2 for window_size=5 centered) across all available years,
        then computes the empirical quantile.

        Parameters
        ----------
        calendar_day : int
            Center day of year (0-364 or 0-365)
        data : np.ndarray
            Shape (time, lat, lon) — daily data
        time_array : np.ndarray
            Time coordinate values
        time_units : str
            Time units string
        calendar : str
            Calendar type
        quantile : float
            Quantile to compute (0 to 1)
        window_size : int
            Size of rolling window (default 5)
        base_years : np.ndarray | None
            Years to include
        exclude_year : int | None
            Year to exclude (for leave-one-out)

        Returns
        -------
        np.ndarray
            Shape (lat, lon) — quantile threshold per grid point
        """
        n_doy = 366  # Max days in year
        half_window = window_size // 2

        # Collect data from window of days
        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.fromiter((d.year for d in dates), dtype=int, count=len(dates))
        day_of_years = np.fromiter((d.timetuple().tm_yday - 1 for d in dates), dtype=int, count=len(dates))

        # Build window [center - half_window, ..., center + half_window]
        window_days = set()
        for offset in range(-half_window, half_window + 1):
            window_day = (calendar_day + offset) % n_doy
            window_days.add(window_day)

        day_match = np.isin(day_of_years, list(window_days))
        if base_years is not None:
            year_match = np.isin(years, base_years)
        else:
            year_match = np.ones_like(years, dtype=bool)

        if exclude_year is not None:
            year_match = year_match & (years != exclude_year)

        idx = np.where(day_match & year_match)[0]

        if idx.size == 0:
            # No data in window; return NaN
            return np.full(data.shape[1:], np.nan, dtype=data.dtype)

        window_data = data[idx]  # Shape (n_samples, lat, lon)

        # Compute quantile per grid point
        return np.quantile(window_data, quantile, axis=0, method='linear')

    def _temperature_quantiles_estimation(
        self,
        temp_data: np.ndarray,
        base_period_mask: np.ndarray,
        time_array: np.ndarray,
        time_units: str,
        calendar: str,
        quantile: float,
        window_size: int = 5,
        bootstrap_samples: int = 1000,
        random_seed: int | None = None,
    ) -> dict:
        """Estimate daily quantile thresholds and compute yearly exceedance frequencies.

        Parameters
        ----------
        temp_data : np.ndarray
            Shape (time, lat, lon) — daily temperature data
        base_period_mask : np.ndarray
            Shape (num_years,) — boolean mask of base-period years
        time_array : np.ndarray
            Time coordinate
        time_units : str
            Time units string
        calendar : str
            Calendar type
        quantile : float
            Quantile to compute (0.1 for 10th percentile, etc.)
        window_size : int
            Rolling window size for threshold estimation
        bootstrap_samples : int
            Number of bootstrap resamples for base-period years
        random_seed : int | None
            Random seed for reproducibility

        Returns
        -------
        dict
            {'result': yearly_exceedances (shape: num_years × lat × lon),
             'thresholds': daily_thresholds (shape: 366 × lat × lon)}
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        dates = num2date(time_array, units=time_units, calendar=calendar)
        years = np.asarray([d.year for d in dates], dtype=int)
        day_of_years = np.asarray([d.timetuple().tm_yday - 1 for d in dates], dtype=int)

        unique_years = np.unique(years)
        n_years = len(unique_years)
        year_to_idx = {int(y): i for i, y in enumerate(unique_years)}

        # Validate base_period_mask
        if base_period_mask.size != n_years:
            err_msg = (
                f"base_period_mask size ({base_period_mask.size}) does not match "
                f"number of years in data ({n_years})"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        base_years = unique_years[base_period_mask.astype(bool)]

        # --- Step 1: Compute daily thresholds from base period ---
        n_doy = 366  # Handle leap years
        thresholds_by_doy = np.full((n_doy,) + temp_data.shape[1:], np.nan, dtype=temp_data.dtype)

        for doy in range(n_doy):
            thresholds_by_doy[doy] = self._compute_percentile_from_window(
                doy, temp_data, time_array, time_units, calendar,
                quantile, window_size, base_years, exclude_year=None
            )

        # --- Step 2: Compute yearly exceedance frequencies ---
        exceedance_rates = np.full((n_years,) + temp_data.shape[1:], np.nan, dtype=np.float32)

        for i, year in enumerate(unique_years):
            year_idx = np.where(years == year)[0]
            year_doys = day_of_years[year_idx]
            year_data = temp_data[year_idx]

            is_base_year = base_period_mask[i]

            if not is_base_year:
                # For years outside base period: use fixed thresholds
                exceed_count = 0
                valid_count = 0

                for t, doy in enumerate(year_doys):
                    doy_threshold = thresholds_by_doy[doy]
                    temp_val = year_data[t]

                    valid_mask = ~np.isnan(temp_val) & ~np.isnan(doy_threshold)
                    valid_count += np.sum(valid_mask)

                    # For lower-tail quantiles (e.g., 10th percentile), use <
                    # For upper-tail quantiles (e.g., 90th percentile), use >
                    if quantile <= 0.5:
                        exceed_mask = temp_val < doy_threshold
                    else:
                        exceed_mask = temp_val > doy_threshold

                    exceed_count += np.sum(exceed_mask & valid_mask)

                exceedance_rates[i] = np.where(
                    valid_count > 0,
                    exceed_count / valid_count,
                    np.nan
                )
            else:
                # For years in base period: bootstrap resampling
                boot_rates = []

                for b in range(bootstrap_samples):
                    # Build leave-one-year-out sample and bootstrap-resample
                    exceed_count_b = 0
                    valid_count_b = 0

                    for t, doy in enumerate(year_doys):
                        # Compute threshold from bootstrap sample of leave-one-year-out base
                        boot_threshold = self._compute_percentile_from_window(
                            doy, temp_data, time_array, time_units, calendar,
                            quantile, window_size, base_years, exclude_year=int(year)
                        )

                        # Don't bootstrap resample if computing from full base - just apply threshold
                        # This implements the leave-one-year-out thresholds directly
                        temp_val = year_data[t]

                        valid_mask = ~np.isnan(temp_val) & ~np.isnan(boot_threshold)
                        valid_count_b += np.sum(valid_mask)

                        if quantile <= 0.5:
                            exceed_mask = temp_val < boot_threshold
                        else:
                            exceed_mask = temp_val > boot_threshold

                        exceed_count_b += np.sum(exceed_mask & valid_mask)

                    boot_rate = np.where(
                        valid_count_b > 0,
                        exceed_count_b / valid_count_b,
                        np.nan
                    )
                    boot_rates.append(boot_rate)

                # Average across bootstrap repetitions
                boot_rates_arr = np.asarray(boot_rates)
                exceedance_rates[i] = np.nanmean(boot_rates_arr, axis=0)

        return {
            'result': exceedance_rates,
            'thresholds': thresholds_by_doy
        }

    @check_supported_compute_frequencies
    def tx90p(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        time_array:         np.ndarray,
        time_units:         str,
        calendar:           str,
        window_size:        int = 5,
        bootstrap_samples:  int = 1000,
        random_seed:        int | None = None,
    ) -> dict:
        """90th percentile of daily maximum temperature (TX90p).

        Returns yearly count of days when TX > 90th percentile threshold.
        For base-period years, uses bootstrap resampling with leave-one-year-out.
        """
        return self._temperature_quantiles_estimation(
            tasmax_data, base_period_mask, time_array, time_units, calendar,
            quantile, window_size, bootstrap_samples, random_seed
        )

    @check_supported_compute_frequencies
    def tn90p(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        time_array:         np.ndarray,
        time_units:         str,
        calendar:           str,
        window_size:        int = 5,
        bootstrap_samples:  int = 1000,
        random_seed:        int | None = None,
    ) -> dict:
        """90th percentile of daily minimum temperature (TN90p).

        Returns yearly count of days when TN > 90th percentile threshold.
        For base-period years, uses bootstrap resampling with leave-one-year-out.
        """
        return self._temperature_quantiles_estimation(
            tasmin_data, base_period_mask, time_array, time_units, calendar,
            quantile, window_size, bootstrap_samples, random_seed
        )

    @check_supported_compute_frequencies
    def tx10p(
        self,
        compute_fq:         str,
        tasmax_data:        np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        time_array:         np.ndarray,
        time_units:         str,
        calendar:           str,
        window_size:        int = 5,
        bootstrap_samples:  int = 1000,
        random_seed:        int | None = None,
    ) -> dict:
        """10th percentile of daily maximum temperature (TX10p).

        Returns yearly count of days when TX < 10th percentile threshold.
        For base-period years, uses bootstrap resampling with leave-one-year-out.
        """
        return self._temperature_quantiles_estimation(
            tasmax_data, base_period_mask, time_array, time_units, calendar,
            quantile, window_size, bootstrap_samples, random_seed
        )

    @check_supported_compute_frequencies
    def tn10p(
        self,
        compute_fq:         str,
        tasmin_data:        np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        base_period_mask:   np.ndarray,
        time_array:         np.ndarray,
        time_units:         str,
        calendar:           str,
        window_size:        int = 5,
        bootstrap_samples:  int = 1000,
        random_seed:        int | None = None,
    ) -> dict:
        """10th percentile of daily minimum temperature (TN10p).

        Returns yearly count of days when TN < 10th percentile threshold.
        For base-period years, uses bootstrap resampling with leave-one-year-out.
        """
        return self._temperature_quantiles_estimation(
            tasmin_data, base_period_mask, time_array, time_units, calendar,
            quantile, window_size, bootstrap_samples, random_seed
        )

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


    def _compute_precipitation_quantile_threshold(
        self,
        pr_data: np.ndarray,
        quantile: float,
        base_period_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute global perceptile threshold for precipitation data.

        Parameters
        ----------
        pr_data : np.ndarray
            Shape (time, lat, lon) — daily precipitation
        quantile : float
            Quantile level (0 to 1)
        base_period_mask : np.ndarray
            Shape (num_years,) — boolean mask of base-period years to
            include in threshold calculation

        Returns
        -------
        np.ndarray
            Threshold value per grid point
        """
        valid_data = pr_data[~np.isnan(pr_data)]
        valid_data = valid_data[base_period_mask]

        if valid_data.size == 0:
            return np.full(pr_data.shape[1:], np.nan, dtype=pr_data.dtype)

        percentile_value = np.percentile(valid_data, quantile * 100)
        return np.full(pr_data.shape[1:], percentile_value, dtype=pr_data.dtype)

    @check_supported_compute_frequencies
    def r95p(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        quantile:           float,
        threshold_array:    np.ndarray | None = None,
    ):
        """Sum of precipitation on very wet days (R95p).

        Total precipitation on days with daily precipitation > 95th percentile.

        Parameters
        ----------
        threshold_array : np.ndarray | None, optional
            Pre-computed threshold (e.g., from QuantileThresholdIndex).
            If None, computed from quantile.
        """
        # Use provided threshold or compute from quantile
        if threshold_array is None:
            threshold = self._compute_precipitation_quantile_threshold(pr_data, quantile, base_period_mask)
        else:
            threshold = threshold_array

        # Identify days exceeding threshold
        exceed_mask = pr_data > threshold[np.newaxis, ...]

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
        quantile:           float,
        threshold_array:    np.ndarray | None = None,
    ):
        """Sum of precipitation on extremely wet days (R99p).

        Total precipitation on days with daily precipitation > 99th percentile.

        Parameters
        ----------
        threshold_array : np.ndarray | None, optional
            Pre-computed threshold (e.g., from QuantileThresholdIndex).
            If None, computed from quantile.
        """
        # Use provided threshold or compute from quantile
        if threshold_array is None:
            threshold = self._compute_precipitation_quantile_threshold(pr_data, quantile)
        else:
            threshold = threshold_array

        # Identify days exceeding threshold
        exceed_mask = pr_data > threshold[np.newaxis, ...]

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
    ):
        """Percentage contribution of very wet days to total precipitation (R95pTOT).

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
        wet_mask = pr_data >= 1.0  # 1 mm threshold
        prcptot = self._aggregate_by_group(pr_data * wet_mask, group_index, np.sum)

        # Avoid division by zero
        r95p_tot = np.where(prcptot > 0, 100.0 * r95p / prcptot, np.nan)

        logger.debug(f"Computed R95pTOT with shape {r95p_tot.shape}")
        return r95p_tot

    @check_supported_compute_frequencies
    def r99p_tot(
        self,
        compute_fq:         str,
        pr_data:            np.ndarray,
        group_index:        np.ndarray,
        threshold_array:    np.ndarray,
    ):
        """Percentage contribution of extremely wet days to total precipitation (R99pTOT).

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
        wet_mask = pr_data >= 1.0  # 1 mm threshold
        prcptot = self._aggregate_by_group(pr_data * wet_mask, group_index, np.sum)

        # Avoid division by zero
        r99p_tot = np.where(prcptot > 0, 100.0 * r99p / prcptot, np.nan)

        logger.debug(f"Computed R99pTOT with shape {r99p_tot.shape}")
        return r99p_tot


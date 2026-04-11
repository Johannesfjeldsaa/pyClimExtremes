# ==================================================================== #
# === imports and setup ============================================== #
# ==================================================================== #

import numpy as np
from numba import njit, cuda
from netCDF4 import num2date

from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)

# ==================================================================== #
# === Spell duration calculation ===================================== #
# ==================================================================== #

def _count_consecutive_days(
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


def _first_run_start(
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

def growing_season_length(
    tas_data: np.ndarray,
    group_index: np.ndarray,
    dates,
    fixed_threshold: float,
    run_len: int = 6,
    first_half_months: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> np.ndarray:
    warm = tas_data >= fixed_threshold
    cold = tas_data < fixed_threshold

    warm_runs = _count_consecutive_days(warm)
    cold_runs = _count_consecutive_days(cold)

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

        start_idx = _first_run_start(
            warm_runs,
            int(first_half[0]),
            int(first_half[-1]) + 1,
            min_run_length=run_len,
        )

        end_idx = _first_run_start(
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

def max_spell_length_by_group(
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

# ==================================================================== #
# === Quantile estimation ============================================ #
# ==================================================================== #

def _verify_base_period_mask(
    data: np.ndarray,
    base_period_mask: np.ndarray,
    group_index: np.ndarray | None = None
) -> np.ndarray:
    """Validate and expand base_period_mask to per-timestep boolean mask."""

    base_period_mask = np.asarray(base_period_mask, dtype=bool)
    if base_period_mask.ndim != 1:
        err_msg = (f"base_period_mask must be 1D, got {base_period_mask.shape}")
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    n_time = data.shape[0]

    if base_period_mask.size == n_time:
        time_mask = base_period_mask
    else:
        # Per-year mask, expand to per-timestep
        if group_index is None:
            raise ValueError("group_index required for per-year base_period_mask")

        group_index = np.asarray(group_index)
        if group_index.shape != (n_time,):
            err_msg = f"group_index must be (time,), got {group_index.shape}"
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        unique_groups = np.unique(group_index)
        if base_period_mask.size != unique_groups.size:
            err_msg = (
                f"Mask size mismatch: mask={base_period_mask.size}, "
                f"time={n_time}, unique_groups={unique_groups.size}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        time_mask = base_period_mask[group_index.astype(int)]

    return time_mask

# ==================================================================== #
# === Quantile estimation for temperature indices ==================== #
# ==================================================================== #

def _build_day_of_year_sample(
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

def temperature_quantiles_estimation(
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
        thresholds_by_doy[doy] = _compute_percentile_from_window(
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
                    boot_threshold = _compute_percentile_from_window(
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


# ==================================================================== #
# === Quantile estimation for precipitation indices ================== #
# ==================================================================== #

def precipitation_quantiles_estimation(
    pr_data: np.ndarray,
    quantile: float,
    base_period_mask: np.ndarray,
    group_index: np.ndarray,
    wet_day_threshold: float,
) -> np.ndarray:
    """
    Compute precipitation quantile threshold per grid point.
    Uses quantile type 8 (alphap=0.375, betap=0.375) as used
    by climdex.pcic.

    Parameters
    ----------
    pr_data : np.ndarray
        Daily precipitation data with shape (time, lat, lon).
    quantile : float
        Quantile level between 0 and 1.
    base_period_mask : np.ndarray
        1D boolean array indicating which time steps belong to
        the base period. Can follow either of two formats:
        * Per-timestep mask of shape (time,). Used if base_period_mask is
        same length as time dimension of pr_data.
        * Per-year mask of shape (num_years,). Used if base_period_mask
        is shorter than time dimension, in which case it will be expanded
        to match the time dimension using the group_index.
    group_index : np.ndarray
        1D integer array of same length as time dimension of pr_data,
        indicating group membership (e.g., year) for each time step.
        Required if base_period_mask is a per-year mask.
    wet_day_threshold : float
        Minimum precipitation to consider a day "wet" and include in
        quantile calculation. Typically 1 mm or 1/86400 kg m-2.
        Can be found by cls.get_wet_day_threshold(`some_pr_unit`)
        if cls is `QuantileIndex` subclass.


    Returns
    -------
    np.ndarray
        2D array (lat, lon) of the computed quantile threshold.
    """
    if pr_data.ndim != 3:
        raise ValueError(
            f"Expected shape (time, lat, lon), got {pr_data.shape}"
        )

    time_mask = _verify_base_period_mask(
        pr_data, base_period_mask, group_index
    )

    baseline_data = pr_data[time_mask, ...]
    if baseline_data.size == 0:
        warn_msg = "No data in baseline period after applying mask; returning NaN"
        logger.warning(warn_msg, stack_info=True)
        return np.full(pr_data.shape[1:], np.nan, dtype=pr_data.dtype)

    is_wet = (~np.isnan(baseline_data)) & (baseline_data >= wet_day_threshold)
    wet_day_data = np.where(is_wet, baseline_data, np.nan)

    if np.isnan(wet_day_data).all():
        warn_msg = "No wet days in baseline period; returning NaN"
        logger.warning(warn_msg, stack_info=True)
        return np.full(pr_data.shape[1:], np.nan, dtype=pr_data.dtype)

    n_time_base, n_lat, n_lon = wet_day_data.shape
    wet_day_flat = wet_day_data.reshape(n_time_base, -1)  # (time, lat*lon)

    # Support single or multiple quantiles
    if not isinstance(quantile, float):
        quantile = float(quantile)

    # NumPy's 'median_unbiased' method matches Hyndman-Fan type 8,
    # which is the same quantile convention used by climdex here:
    # https://numpy.org/doc/2.0/reference/generated/numpy.quantile.html
    percentile_values = np.nanquantile(
        wet_day_flat,
        quantile,
        axis=0,
        method="median_unbiased",
    )

    return percentile_values.reshape(n_lat, n_lon)
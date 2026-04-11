# ==================================================================== #
# === imports and setup ============================================== #
# ==================================================================== #

import math
from typing import Any
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

# ==================================================================== #
# === Quantile estimation for temperature indices ==================== #
# ==================================================================== #


def _normalize_quantiles(
    quantile: float | list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize scalar or vector quantiles to a 1D float array."""
    quantiles = np.atleast_1d(np.asarray(quantile, dtype=np.float64))
    if quantiles.ndim != 1:
        err_msg = (
            "quantile must be a scalar or 1D sequence, "
            f"got shape {quantiles.shape}"
        )
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        err_msg = f"quantile values must be strictly between 0 and 1, got {quantiles}"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    return quantiles, np.isscalar(quantile)


def _format_quantile_label(quantiles: np.ndarray) -> str:
    """Format one or more quantiles for progress/debug messages."""
    return ", ".join(f"{q * 100:g}" for q in quantiles)


def _verify_base_period_mask(
    data: np.ndarray,
    base_period_mask: np.ndarray,
    group_index: np.ndarray | None = None,
) -> np.ndarray:
    """Validate and expand a base-period mask to the time axis."""
    base_period_mask = np.asarray(base_period_mask, dtype=bool)
    if base_period_mask.ndim != 1:
        err_msg = f"base_period_mask must be 1D, got {base_period_mask.shape}"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    n_time = data.shape[0]
    if base_period_mask.size == n_time:
        return base_period_mask

    if group_index is None:
        err_msg = "group_index required for per-year base_period_mask"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

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

    return base_period_mask[group_index.astype(int)]

def _get_day_of_year_array(
    time_array: np.ndarray,
    time_units: str,
    calendar: str,
) -> np.ndarray:
    """Convert CF-style times to a 1D day-of-year integer array."""
    dates_raw = num2date(time_array, units=time_units, calendar=calendar)
    if isinstance(dates_raw, np.ndarray):
        dates_iter = dates_raw.flat
    else:
        dates_iter = [dates_raw]
    return np.array(
        [getattr(getattr(d, "timetuple")(), "tm_yday") for d in dates_iter],
        dtype=np.int32,
    )


def _get_days_per_year(calendar: str, doy: np.ndarray) -> int:
    """Infer the working day-of-year size from calendar metadata."""
    if calendar in [
        "gregorian", "proleptic_gregorian", "standard", "julian", "all_leap"
    ]:
        has_leap = np.any(doy == 366)
        return 366 if has_leap else 365
    return 365


def _prepare_temperature_quantile_inputs(
    temp_data: np.ndarray,
    quantile: float | list[float] | tuple[float, ...] | np.ndarray,
    base_period_mask: np.ndarray,
    group_index: np.ndarray,
    time_array: np.ndarray,
    time_units: str,
    calendar: str,
    window_size: int,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray, int, int, int]:
    """Validate inputs and extract base-period arrays for the njit kernel."""
    if temp_data.ndim != 3:
        err_msg = f"Expected temp_data with shape (time, lat, lon), got {temp_data.shape}"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)
    if window_size % 2 == 0:
        err_msg = f"window_size must be odd, got {window_size}"
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    quantiles, is_scalar = _normalize_quantiles(quantile)
    _, n_lat, n_lon = temp_data.shape

    daily_mask = _verify_base_period_mask(temp_data, base_period_mask, group_index)
    doy = _get_day_of_year_array(time_array, time_units, calendar)
    days_per_year = _get_days_per_year(calendar, doy)

    base_time_idx = np.where(daily_mask)[0]
    if base_time_idx.size == 0:
        err_msg = "No data available in base period after applying mask."
        logger.error(err_msg, stack_info=True)
        raise ValueError(err_msg)

    base_data = np.ascontiguousarray(temp_data[base_time_idx], dtype=np.float64)
    base_doy = np.ascontiguousarray(doy[base_time_idx], dtype=np.int32)
    return quantiles, is_scalar, base_data, base_doy, days_per_year, n_lat, n_lon


@njit(cache=True, fastmath={"contract": True},)
def _temperature_quantiles_loop_cpu(
    base_data: np.ndarray,
    base_doy: np.ndarray,
    quantiles: np.ndarray,
    window_size: int,
    days_per_year: int,
    n_lat: int,
    n_lon: int,
) -> np.ndarray:
    """njit DOY-loop kernel for temperature quantile thresholds."""
    half_win = window_size // 2
    n_quantiles = quantiles.shape[0]
    n_base = base_data.shape[0]
    thresholds = np.full((n_quantiles, days_per_year, n_lat, n_lon), np.nan)

    for target_doy in range(1, days_per_year + 1):
        # collect indices whose DOY falls in the circular window
        count = 0
        for t in range(n_base):
            d = base_doy[t]
            delta = (d - target_doy + days_per_year) % days_per_year
            if delta <= half_win or delta >= days_per_year - half_win:
                count += 1

        if count == 0:
            continue

        window_data = np.empty((count, n_lat, n_lon))
        idx = 0
        for t in range(n_base):
            d = base_doy[t]
            delta = (d - target_doy + days_per_year) % days_per_year
            if delta <= half_win or delta >= days_per_year - half_win:
                window_data[idx] = base_data[t]
                idx += 1

        # sort each grid point and apply Hyndman-Fan type 8
        for i in range(n_lat):
            for j in range(n_lon):
                col = window_data[:, i, j].copy()
                col.sort()
                n = col.shape[0]
                for q_idx in range(n_quantiles):
                    q = quantiles[q_idx]
                    h = (n + 1.0 / 3.0) * q + 1.0 / 3.0
                    if h <= 1.0:
                        thresholds[q_idx, target_doy - 1, i, j] = col[0]
                    elif h >= n:
                        thresholds[q_idx, target_doy - 1, i, j] = col[n - 1]
                    else:
                        lo = int(np.floor(h)) - 1
                        frac = h - np.floor(h)
                        thresholds[q_idx, target_doy - 1, i, j] = (
                            col[lo] + frac * (col[lo + 1] - col[lo])
                        )

    return thresholds


@cuda.jit
def _temperature_quantiles_loop_gpu_kernel(
    base_data,
    base_doy,
    quantiles,
    half_win,
    days_per_year,
    thresholds,
):
    """CUDA kernel: one thread per (lat, lon) point, loops over all DOYs."""
    flat_ij = cuda.grid(1)
    n_lat = thresholds.shape[2]
    n_lon = thresholds.shape[3]
    if flat_ij >= n_lat * n_lon:
        return

    i = flat_ij // n_lon
    j = flat_ij % n_lon
    n_base = base_data.shape[0]
    n_quantiles = quantiles.shape[0]

    col: Any = cuda.local.array(1024, np.float64)  # type: ignore[arg-type]  # max practical window count

    for target_doy in range(1, days_per_year + 1):
        count = 0
        for t in range(n_base):
            d = base_doy[t]
            delta = (d - target_doy + days_per_year) % days_per_year
            if delta <= half_win or delta >= days_per_year - half_win:
                col[count] = base_data[t, i, j]
                count += 1

        if count == 0:
            continue

        # insertion sort
        for s in range(1, count):
            key = col[s]
            k = s - 1
            while k >= 0 and col[k] > key:
                col[k + 1] = col[k]
                k -= 1
            col[k + 1] = key

        # Hyndman-Fan type 8
        for q_idx in range(n_quantiles):
            q = quantiles[q_idx]
            h = (count + 1.0 / 3.0) * q + 1.0 / 3.0
            if h <= 1.0:
                thresholds[q_idx, target_doy - 1, i, j] = col[0]
            elif h >= count:
                thresholds[q_idx, target_doy - 1, i, j] = col[count - 1]
            else:
                lo = int(math.floor(h)) - 1
                frac = h - math.floor(h)
                thresholds[q_idx, target_doy - 1, i, j] = (
                    col[lo] + frac * (col[lo + 1] - col[lo])
                )


def _temperature_quantiles_loop_gpu(
    base_data: np.ndarray,
    base_doy: np.ndarray,
    quantiles: np.ndarray,
    window_size: int,
    days_per_year: int,
    n_lat: int,
    n_lon: int,
) -> np.ndarray:
    """GPU wrapper: transfers data, launches kernel, returns result on host."""
    half_win = window_size // 2
    n_quantiles = quantiles.shape[0]
    thresholds_host = np.full(
        (n_quantiles, days_per_year, n_lat, n_lon), np.nan, dtype=np.float64
    )
    d_base_data = cuda.to_device(base_data)
    d_base_doy = cuda.to_device(base_doy)
    d_quantiles = cuda.to_device(quantiles)
    d_thresholds = cuda.to_device(thresholds_host)

    n_spatial = n_lat * n_lon
    threads_per_block = 64
    blocks_per_grid = (n_spatial + threads_per_block - 1) // threads_per_block
    _temperature_quantiles_loop_gpu_kernel[blocks_per_grid, threads_per_block](  # type: ignore[index]
        d_base_data, d_base_doy, d_quantiles, half_win, days_per_year, d_thresholds
    )
    return d_thresholds.copy_to_host()


def temperature_quantiles_estimation(
    temp_data: np.ndarray,
    quantile: float | list[float] | tuple[float, ...] | np.ndarray,
    base_period_mask: np.ndarray,
    group_index: np.ndarray,
    time_array: np.ndarray,
    time_units: str,
    calendar: str,
    window_size: int = 5,
    bootstrap_samples: int | None = None,
    random_seed: int | None = None,
    use_cuda: bool = False,
) -> np.ndarray:
    """Compute day-of-year dependent temperature quantile thresholds.

    Parameters
    ----------
    temp_data : np.ndarray
        Daily temperature data, shape (time, lat, lon).
    quantile : float or sequence of float
        Quantile level(s) strictly between 0 and 1.
    base_period_mask : np.ndarray
        Boolean mask for base period (per-timestep or per-year).
    group_index : np.ndarray
        Integer array mapping time steps to year groups.
    time_array : np.ndarray
        CF-convention time coordinate values.
    time_units : str
        CF-convention time units string.
    calendar : str
        Calendar type.
    window_size : int, optional
        Size of day-of-year window (must be odd). Default 5.
    bootstrap_samples : int | None, optional
        Reserved for future bootstrap implementations.
    random_seed : int | None, optional
        Reserved for reproducibility.

    Returns
    -------
    np.ndarray
        Float64 array of shape (days_per_year, lat, lon) or
        (n_quantiles, days_per_year, lat, lon) for vector quantile input.
    """
    quantiles, is_scalar, base_data, base_doy, days_per_year, n_lat, n_lon = (
        _prepare_temperature_quantile_inputs(
            temp_data=temp_data,
            quantile=quantile,
            base_period_mask=base_period_mask,
            group_index=group_index,
            time_array=time_array,
            time_units=time_units,
            calendar=calendar,
            window_size=window_size,
        )
    )
    logger.debug(
        "Computing temperature quantiles [%s] over %d base timesteps, %d DOYs",
        _format_quantile_label(quantiles), base_data.shape[0], days_per_year,
    )
    if use_cuda:
        thresholds = _temperature_quantiles_loop_gpu(
            base_data, base_doy, quantiles, window_size, days_per_year, n_lat, n_lon
        )
    else:
        thresholds = _temperature_quantiles_loop_cpu(
            base_data, base_doy, quantiles, window_size, days_per_year, n_lat, n_lon
        )
    return thresholds[0] if is_scalar else thresholds

# ==================================================================== #
# === Quantile estimation for precipitation indices ================== #
# ==================================================================== #

def precipitation_quantiles_estimation(
    pr_data: np.ndarray,
    quantile: float | list[float] | tuple[float, ...] | np.ndarray,
    base_period_mask: np.ndarray,
    group_index: np.ndarray,
    wet_day_threshold: float,
) -> np.ndarray:
    """
    Compute precipitation quantile threshold per grid point.
    Uses Hyndman-Fan type 8 quantile (alphap=0.375, betap=0.375) as
    used by climdex.pcic.

    Parameters
    ----------
    pr_data : np.ndarray
        Daily precipitation data with shape (time, lat, lon).
    quantile : float | list[float] | tuple[float, ...] | np.ndarray
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
        Float64 array with shape (lat, lon) containing the computed
        quantile threshold.
    """
    if pr_data.ndim != 3:
        raise ValueError(f"Expected shape (time, lat, lon), got {pr_data.shape}")

    quantiles, is_scalar = _normalize_quantiles(quantile)
    time_mask = _verify_base_period_mask(pr_data, base_period_mask, group_index)
    baseline_data = pr_data[time_mask]

    if baseline_data.size == 0:
        logger.warning("No data in baseline period after applying mask; returning NaN")
        return np.full(pr_data.shape[1:], np.nan, dtype=np.float64)

    wet_day_data = np.where(
        (~np.isnan(baseline_data)) & (baseline_data >= wet_day_threshold),
        baseline_data,
        np.nan,
    )
    # NumPy's 'median_unbiased' method matches Hyndman-Fan type 8,
    # which is the same quantile convention used by climdex here:
    # https://numpy.org/doc/2.0/reference/generated/numpy.quantile.html
    result = np.nanquantile(wet_day_data, quantiles, axis=0, method="median_unbiased")
    return result[0] if is_scalar else result

from pathlib import Path
import timeit
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset, num2date

from pyClimExtremes.indices.base_index import QuantileIndex
from pyClimExtremes.indices.quantile_indices import build_runtime_quantile_class
from pyClimExtremes.indices.units_utils import validate_input_units
from pyClimExtremes.io.data_wrapping import prepare_inputs_and_meta, prepare_time_groupings
from pyClimExtremes.io.save_utils import check_filepath
from pyClimExtremes.logging.setup_logging import get_logger

logger = get_logger(__name__)


def _normalize_percentile_value(raw_value: float | int) -> tuple[float, float]:
	"""Normalize user-provided percentile values to percent and fraction."""
	percentile = float(raw_value)
	if percentile <= 0:
		err_msg = f"Percentile must be > 0, got {raw_value}."
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)
	if percentile < 1.0:
		percentile *= 100.0
	if percentile >= 100.0:
		err_msg = f"Percentile must be < 100, got {raw_value}."
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)
	return percentile, percentile / 100.0


def _resolve_quantile_requests(
	quantiles: dict[str, float | int | list[float] | tuple[float, ...]],
) -> list[type[QuantileIndex]]:
	"""Resolve dict-based generic qXXp requests into runtime quantile classes."""
	quantile_index_list: list[type[QuantileIndex]] = []
	for family_id, values in quantiles.items():
		if isinstance(values, (list, tuple, np.ndarray)):
			iter_values = values
		else:
			iter_values = [values]
		for value in iter_values:
			percentile, _ = _normalize_percentile_value(value)
			quantile_index_list.append(
				build_runtime_quantile_class(family_id, percentile)
			)
	return quantile_index_list


def _build_threshold_filename(index_class: type, meta: dict[str, Any]) -> str:
	"""Build a stable filename for threshold outputs."""
	source_id = str(meta.get("source_id", "unknown"))
	experiment_id = str(meta.get("experiment_id", "unknown"))
	variant_label = str(meta.get("variant_label", "unknown"))
	baseline_period = index_class.baseline_period

	return (
		f"{index_class.index_id}_threshold_"
		f"{source_id}_{experiment_id}_{variant_label}_"
		f"{baseline_period[0]}-{baseline_period[1]}.nc"
	)


def _write_threshold_netcdf(
	threshold_array: np.ndarray,
	index_class: type,
	output_path: Path,
	metadata: dict[str, Any],
) -> None:
	"""Write threshold array to NetCDF.

	Temperature thresholds are expected as (day_of_year, lat, lon), while
	precipitation thresholds are expected as (lat, lon).
	"""
	lat = metadata.get("lat")
	lon = metadata.get("lon")
	time_units = metadata.get("time_units")
	calendar = metadata.get("calendar")

	if lat is None or lon is None:
		err_msg = (
			"Missing 'lat' or 'lon' in metadata; cannot write threshold file."
		)
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)

	with Dataset(output_path, "w") as ds:
		ds.createDimension("lat", len(lat))
		ds.createDimension("lon", len(lon))

		lat_var = ds.createVariable("lat", "f4", ("lat",))
		lat_var[:] = lat
		lat_var.standard_name = "latitude"
		lat_var.units = "degrees_north"

		lon_var = ds.createVariable("lon", "f4", ("lon",))
		lon_var[:] = lon
		lon_var.standard_name = "longitude"
		lon_var.units = "degrees_east"

		if threshold_array.ndim == 3:
			ds.createDimension("day_of_year", threshold_array.shape[0])
			doy_var = ds.createVariable("day_of_year", "i4", ("day_of_year",))
			doy_var[:] = np.arange(1, threshold_array.shape[0] + 1, dtype=np.int32)
			threshold_var = ds.createVariable(
				"threshold",
				"f4",
				("day_of_year", "lat", "lon"),
				zlib=True,
			)
		elif threshold_array.ndim == 2:
			threshold_var = ds.createVariable(
				"threshold",
				"f4",
				("lat", "lon"),
				zlib=True,
			)
		else:
			err_msg = (
				"Threshold array must be 2D (lat, lon) or "
				"3D (day_of_year, lat, lon)."
			)
			logger.error(err_msg, stack_info=True)
			raise ValueError(err_msg)

		threshold_var[:] = threshold_array
		threshold_var.long_name = (
			f"Threshold array for index {index_class.index_id} "
			f"(quantile={getattr(index_class, 'quantile', 'unknown')})"
		)

		# Threshold units follow the input variable units.
		if index_class.index_type == "temperature_quantile":
			threshold_var.units = "deg_C_or_K"
		else:
			threshold_var.units = "mm d-1_or_kg m-2 s-1"

		ds.index_id = index_class.index_id
		ds.index_long_name = index_class.index_long_name
		ds.index_type = index_class.index_type
		ds.quantile = getattr(index_class, "quantile", np.nan)
		ds.baseline_period_start = getattr(index_class, "baseline_period", (None, None))[0]
		ds.baseline_period_end = getattr(index_class, "baseline_period", (None, None))[1]
		ds.time_units = "" if time_units is None else str(time_units)
		ds.calendar = "" if calendar is None else str(calendar)

		parent_global = metadata.get("parent_global_attrs", {})
		for attr_name, attr_value in parent_global.items():
			if not hasattr(ds, attr_name):
				setattr(ds, attr_name, attr_value)


def _build_base_period_mask(
	*,
	time_array: np.ndarray,
	time_units: str,
	calendar: str,
	baseline_period: tuple[int, int],
) -> np.ndarray:
	"""Build per-year boolean mask for baseline period selection."""
	dates = num2date(time_array, units=time_units, calendar=calendar)
	years = np.asarray([d.year for d in dates], dtype=int)
	unique_years = np.unique(years)

	start_year, end_year = baseline_period
	mask = (unique_years >= start_year) & (unique_years <= end_year)

	if not np.any(mask):
		err_msg = (
			"No years from baseline period "
			f"{baseline_period} found in input data years "
			f"[{unique_years.min()}, {unique_years.max()}]."
		)
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)

	return mask


def compute_threshold_array(
	*,
	index_class: type[QuantileIndex],
	index_obj: QuantileIndex,
	arrays: dict[str, np.ndarray],
	units: Mapping[str, str],
	meta: dict[str, Any],
	time_groupings: dict[str, dict[str, np.ndarray]],
	window_size: int = 5,
	bootstrap_samples: int = 1000,
	random_seed: int | None = None,
) -> np.ndarray:
	"""Compute the threshold array for a quantile index without writing it.

	This is the shared threshold-generation path used both by
	``compute_thresholds`` and by ``compute_indices`` when a
	``QuantileThresholdIndex`` is requested without a saved threshold file.
	"""
	time_array = meta.get("time")
	time_units = meta.get("time_units")
	calendar = meta.get("calendar")

	if time_array is None or time_units is None or calendar is None:
		err_msg = (
			"Missing time metadata required for quantile threshold "
			f"index '{index_class.index_id}'."
		)
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)

	baseline_period = getattr(index_class, "baseline_period", (1981, 2010))
	base_period_mask = _build_base_period_mask(
		time_array=time_array,
		time_units=time_units,
		calendar=calendar,
		baseline_period=baseline_period,
	)

	if index_class.index_type == "temperature_quantile":
		compute_kwargs = {
			"compute_fq": "yr",
			"data_array": arrays,
			"group_index": time_groupings["yr"]["group_index"],
			"time_array": time_array,
			"time_units": time_units,
			"calendar": calendar,
			"base_period_mask": base_period_mask,
			"window_size": window_size,
			"bootstrap_samples": bootstrap_samples,
		}
		if random_seed is not None:
			compute_kwargs["random_seed"] = random_seed

		index_obj.compute(**compute_kwargs)
		threshold_array = index_obj.thresholds_by_doy
	elif index_class.index_type == "precipitation":
		wet_day_threshold = index_class.get_wet_day_threshold(units["pr"])
		if wet_day_threshold is None or isinstance(wet_day_threshold, dict):
			err_msg = (
				"Precipitation quantile threshold computation requires a "
				f"wet_day_threshold for index '{index_class.index_id}'."
			)
			logger.error(err_msg, stack_info=True)
			raise ValueError(err_msg)

		threshold_array = index_obj.compute(
			compute_fq="yr",
			data_array=arrays,
			group_index=time_groupings["yr"]["group_index"],
			base_period_mask=base_period_mask,
			wet_day_threshold=float(wet_day_threshold),
		)
		index_obj.thresholds_by_doy = np.asarray(threshold_array)
	else:
		err_msg = (
			f"Unsupported index_type '{index_class.index_type}' for "
			f"index '{index_class.index_id}'."
		)
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)

	if threshold_array is None:
		err_msg = (
			f"Threshold computation returned None for index "
			f"'{index_class.index_id}'."
		)
		logger.error(err_msg, stack_info=True)
		raise RuntimeError(err_msg)

	return np.asarray(threshold_array)


def compute_thresholds(
	indices: str | list[str],
	compute_backend: str,
	output_dir: Path,
	tasmax: Path | None = None,
	tasmin: Path | None = None,
	pr: Path | None = None,
	overwrite: bool = False,
	window_size: int = 5,
	bootstrap_samples: int = 1000,
	random_seed: int | None = None,
	**kwargs,
) -> tuple[list[Path], list[Path]]:
	"""Compute and write threshold arrays for quantile-based indices.

	Supported indices are subclasses of QuantileIndex. For temperature quantile
	indices, this computes daily thresholds (day_of_year, lat, lon) using the
	index baseline period and stores them in NetCDF. For precipitation quantile
	indices (e.g., R95p/R99p), this computes a per-grid-point threshold
	(lat, lon).

	Parameters
	----------
	indices : str | list[str]
		Requested indices or index groups (e.g., 'all', 'temperature').
	compute_backend : str
		Compute backend name. Currently expected: 'python'.
	output_dir : Path
		Directory where threshold NetCDF files are saved.
	tasmax, tasmin, pr : Path | None
		Input files used by the selected indices.
	overwrite : bool, optional
		Whether existing output files should be overwritten.
	window_size : int, optional
		Temperature threshold rolling window size (default 5).
	bootstrap_samples : int, optional
		Number of bootstrap samples for base-period years.
	random_seed : int | None, optional
		Seed for reproducible bootstrapping.
	**kwargs : dict
		Optional keys:
		- backend_kwargs: dict passed to index constructors.

	Returns
	-------
	tuple[list[Path], list[Path]]
		(new_files, files_created_previously)
	"""
	start_time_checks = timeit.default_timer()
	backend_kwargs = kwargs.get("backend_kwargs", {})
	if not isinstance(quantiles, dict):
		err_msg = (
			"compute_thresholds requires dict-based quantile requests, for example "
			"{'pr_qXXp': [95, 99]} or {'tn_qXXp': [10, 50, 90]}."
		)
		logger.error(err_msg, stack_info=True)
		raise TypeError(err_msg)

	quantile_index_list = _resolve_quantile_requests(quantiles)
	new_files: list[Path] = []
	files_created_previously: list[Path] = []

	if not quantile_index_list:
		logger.warning("No quantile selected. Nothing to do.")
		return [], []

	required_inputs: list[str] = []
	for idx_class in quantile_index_list:
		required_inputs.extend(idx_class.required_vars)
	required_inputs = sorted(list(set(required_inputs)))

	input_args = {
		"tasmax": tasmax,
		"tasmin": tasmin,
		"pr": pr,
	}

	input_files_to_use: dict[str, Path] = {}
	for var in required_inputs:
		path = input_args.get(var)
		if path is None:
			err_msg = (
				f"Input variable '{var}' is required for selected quantile "
				"indices but no file was provided."
			)
			logger.error(err_msg, stack_info=True)
			raise ValueError(err_msg)
		input_files_to_use[var] = Path(path)

	wrappers, meta = prepare_inputs_and_meta(**input_files_to_use)
	time_groupings = prepare_time_groupings(
		fq_list=["yr"],
		compute_backend=compute_backend,
		metadata=meta,
	)

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	new_files: list[Path] = []
	files_created_previously: list[Path] = []

	try:
		for index_class in quantile_index_list:
			index_obj = index_class(compute_backend, **backend_kwargs)

			arrays = {
				var: wrappers[var].load_ndarray(var)
				for var in index_class.required_vars
			}
			units = {
				var: wrappers[var].get_units(var)
				for var in index_class.required_vars
			}

			for var, unit in units.items():
				if unit is None:
					err_msg = (
						f"Input variable '{var}' is missing units for "
						f"index '{index_class.index_id}'."
					)
					logger.error(err_msg, stack_info=True)
					raise ValueError(err_msg)
				if not validate_input_units(var, unit):
					err_msg = (
						f"Invalid input units '{unit}' for variable '{var}' "
						f"used by index '{index_class.index_id}'."
					)
					logger.error(err_msg, stack_info=True)
					raise ValueError(err_msg)

			logger.info("Computing thresholds for %s", index_class.index_id)
			threshold_array = compute_threshold_array(
				index_class=index_class,
				index_obj=index_obj,
				arrays=arrays,
				units=units,
				meta=meta,
				time_groupings=time_groupings,
				window_size=window_size,
				bootstrap_samples=bootstrap_samples,
				random_seed=random_seed,
			)

			output_filename = _build_threshold_filename(index_class, meta)
			output_path = output_dir.joinpath(output_filename)

			if not check_filepath(output_path, overwrite):
				files_created_previously.append(output_path)
				continue

			_write_threshold_netcdf(
				threshold_array=np.asarray(threshold_array),
				index_class=index_class,
				output_path=output_path,
				metadata=meta,
			)
			new_files.append(output_path)

		thresholds_comp_time = timeit.default_timer() - start_time_thresholds_comp

	finally:
		for wrapper in wrappers.values():
			wrapper.close()

	total_time = timeit.default_timer() - start_time_checks
	avg_get_arrays_time = (
		sum(get_arrays_elapsed_time) / len(get_arrays_elapsed_time)
		if get_arrays_elapsed_time else 0.0
	)
	avg_get_units_time = (
		sum(get_units_elapsed_time) / len(get_units_elapsed_time)
		if get_units_elapsed_time else 0.0
	)
	avg_compute_time = (
		sum(compute_elapsed_time) / len(compute_elapsed_time)
		if compute_elapsed_time else 0.0
	)
	avg_write_time = (
		sum(write_elapsed_time) / len(write_elapsed_time)
		if write_elapsed_time else 0.0
	)

	timing_summary = (
		f"Time taken for input checks and loading: {checks_time:.2f} secs.\n"
		f"Time taken for threshold computations: {thresholds_comp_time:.2f} secs.\n"
		f" - Average get_arrays time: {avg_get_arrays_time:.2f} secs.\n"
		f" - Average get_units time: {avg_get_units_time:.2f} secs.\n"
		f" - Average compute time: {avg_compute_time:.2f} secs.\n"
		f" - Average write time: {avg_write_time:.2f} secs.\n"
		f"Total time taken: {total_time:.2f} secs.\n"
		"Detailed timing per quantile:\n"
	)
	if threshold_timing_map:
		for index_id, duration in threshold_timing_map.items():
			timing_summary += (
				f" - Quantile '{index_id}': {duration:.2f} secs.\n"
			)

	summary_msg = (
		"Threshold computation completed. "
		f"Total quantiles requested: {len(quantile_index_list)} with timing details:\n"
		f"{timing_summary}"
	)
	summary_msg += (
		f"New files created: {len(new_files)}." if new_files else
		"No new files created."
	)
	summary_msg += (
		f"Files skipped (already existed): {len(files_created_previously)}."
		if files_created_previously else ""
	)
	logger.info(summary_msg)
	return new_files, files_created_previously

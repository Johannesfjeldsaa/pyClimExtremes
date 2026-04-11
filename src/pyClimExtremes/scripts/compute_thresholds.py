from pathlib import Path
import timeit
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset, num2date

from pyClimExtremes.indices.base_index import QuantileIndex
from pyClimExtremes.indices.quantile_indices import build_runtime_quantile_class
from pyClimExtremes.indices.units_utils import validate_input_units
from pyClimExtremes.io.data_wrapping import (
	DataWrapper,
	prepare_inputs_and_meta,
	prepare_time_groupings,
)
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


def _select_metadata_input_path(
	input_args: Mapping[str, Path | None],
) -> Path:
	"""Choose one available input file for metadata extraction."""
	for key in ("tasmax", "tasmin", "pr"):
		path = input_args.get(key)
		if path is not None:
			return Path(path)

	err_msg = "At least one input file must be provided to compute thresholds."
	logger.error(err_msg, stack_info=True)
	raise ValueError(err_msg)


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
	dates_raw = num2date(time_array, units=time_units, calendar=calendar)
	if isinstance(dates_raw, np.ndarray):
		dates_iter = dates_raw.flat
	else:
		dates_iter = [dates_raw]
	years = np.asarray([getattr(d, "year") for d in dates_iter], dtype=int)
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

		threshold_array = index_obj.compute(**compute_kwargs)
		if index_obj.thresholds_by_doy is not None:
			threshold_array = index_obj.thresholds_by_doy
	elif index_class.index_type == "precipitation_quantile":
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


def _build_precipitation_batch_key(
	index_class: type[QuantileIndex],
	units: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[int, int], float]:
	"""Build a batch key for precipitation quantiles that can share one compute call."""
	wet_day_threshold = index_class.get_wet_day_threshold(units["pr"])
	if wet_day_threshold is None or isinstance(wet_day_threshold, dict):
		err_msg = (
			"Precipitation quantile threshold computation requires a "
			f"wet_day_threshold for index '{index_class.index_id}'."
		)
		logger.error(err_msg, stack_info=True)
		raise ValueError(err_msg)

	return (
		tuple(index_class.required_vars),
		getattr(index_class, "baseline_period", (1981, 2010)),
		float(wet_day_threshold),
	)


def _compute_precipitation_threshold_batch(
	*,
	batch_items: list[tuple[type[QuantileIndex], Path]],
	compute_backend: str,
	backend_kwargs: Mapping[str, Any],
	arrays_cache: Mapping[str, np.ndarray],
	units_cache: Mapping[str, str],
	meta: dict[str, Any],
	time_groupings: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
	"""Compute multiple precipitation quantiles in one backend call."""
	if not batch_items:
		return {}

	index_classes = [index_class for index_class, _ in batch_items]
	reference_index = index_classes[0](compute_backend, **dict(backend_kwargs))
	pr_units = units_cache["pr"]
	batch_key = _build_precipitation_batch_key(index_classes[0], {"pr": pr_units})
	_, baseline_period, wet_day_threshold = batch_key
	base_period_mask = _build_base_period_mask(
		time_array=meta["time"],
		time_units=meta["time_units"],
		calendar=meta["calendar"],
		baseline_period=baseline_period,
	)
	quantiles = np.asarray([index_class.quantile for index_class in index_classes], dtype=np.float64)
	backend_method = getattr(reference_index.compute_backend, "pr_qXXp")
	threshold_array = backend_method(
		compute_fq="yr",
		pr_data=arrays_cache["pr"],
		group_index=time_groupings["yr"]["group_index"],
		quantile=quantiles,
		base_period_mask=base_period_mask,
		wet_day_threshold=wet_day_threshold,
	)
	threshold_array = np.asarray(threshold_array)
	if threshold_array.ndim == 2:
		threshold_array = threshold_array[np.newaxis, ...]

	return {
		index_class.index_id: np.asarray(threshold_array[idx])
		for idx, index_class in enumerate(index_classes)
	}


def compute_thresholds(
	quantiles: dict[str, float | int | list[float] | tuple[float, ...]],
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

	Supported quantiles are subclasses of QuantileIndex. For temperature quantiles
	this computes daily thresholds (day_of_year, lat, lon) using the
	index baseline period and stores them in NetCDF. For precipitation quantiles
 	this computes a per-grid-point threshold (lat, lon).

	Parameters
	----------
	quantiles : dict[str, float | int | list[float] | tuple[float, ...]]
		Requested quantiles must be provided as a dict whose keys are generic
		quantile families such as 'tn_qXXp', 'tx_qXXp', or 'pr_qXXp', and whose
		values are one or more percentile values, e.g.
		{'tn_qXXp': [10, 50, 90], 'pr_qXXp': [95, 99]}.
	compute_backend : str
		Compute backend name. Currently expected: 'python'.
	output_dir : Path
		Directory where threshold NetCDF files are saved.
	tasmax, tasmin, pr : Path | None
		Input files used by the selected quantile(s).
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

	input_args = {
		"tasmax": tasmax,
		"tasmin": tasmin,
		"pr": pr,
	}
	metadata_path = _select_metadata_input_path(input_args)
	metadata_wrapper = DataWrapper(metadata_path)
	try:
		_, meta = prepare_inputs_and_meta(wrappers={"meta": metadata_wrapper})
		time_groupings = prepare_time_groupings(
			fq_list=["yr"],
			compute_backend=compute_backend,
			metadata=meta,
		)
	except Exception:
		metadata_wrapper.close()
		raise

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	pending_outputs: list[tuple[type[QuantileIndex], Path]] = []
	try:
		for index_class in quantile_index_list:
			output_filename = _build_threshold_filename(index_class, meta)
			output_path = output_dir.joinpath(output_filename)
			if not check_filepath(output_path, overwrite):
				files_created_previously.append(output_path)
				continue
			pending_outputs.append((index_class, output_path))
	finally:
		metadata_wrapper.close()

	if not pending_outputs:
		checks_time = timeit.default_timer() - start_time_checks
		thresholds_comp_time = 0.0
		total_time = timeit.default_timer() - start_time_checks
		timing_summary = (
			f"Time taken for input checks and loading: {checks_time:.2f} secs.\n"
			f"Time taken for threshold computations: {thresholds_comp_time:.2f} secs.\n"
			" - Average get_arrays time: 0.00 secs.\n"
			" - Average get_units time: 0.00 secs.\n"
			" - Average compute time: 0.00 secs.\n"
			" - Average write time: 0.00 secs.\n"
			f"Total time taken: {total_time:.2f} secs.\n"
			"Detailed timing per quantile:\n"
		)
		logger.info(
			"Threshold computation completed. Total quantiles requested: %d with timing details:\n%sNo new files created.%s",
			len(quantile_index_list),
			timing_summary,
			(
				f"Files skipped (already existed): {len(files_created_previously)}."
				if files_created_previously else ""
			),
		)
		return new_files, files_created_previously

	required_inputs: list[str] = []
	for idx_class, _ in pending_outputs:
		required_inputs.extend(idx_class.required_vars)
	required_inputs = sorted(list(set(required_inputs)))

	input_files_to_use: dict[str, Path] = {}
	for var in required_inputs:
		path = input_args.get(var)
		if path is None:
			err_msg = (
				f"Input variable '{var}' is required for selected quantile "
				"but no file was provided."
			)
			logger.error(err_msg, stack_info=True)
			raise ValueError(err_msg)
		input_files_to_use[var] = Path(path)

	wrappers: dict[str, DataWrapper] = {
		var: DataWrapper(path)
		for var, path in input_files_to_use.items()
	}
	checks_time = timeit.default_timer() - start_time_checks

	threshold_timing_map: dict[str, float] = {}
	get_arrays_elapsed_time: list[float] = []
	get_units_elapsed_time: list[float] = []
	compute_elapsed_time: list[float] = []
	write_elapsed_time: list[float] = []
	arrays_cache: dict[str, np.ndarray] = {}
	units_cache: dict[str, str] = {}
	thresholds_comp_time = 0.0

	try:
		for var in required_inputs:
			get_arrays_timer = timeit.default_timer()

			arrays_cache[var] = wrappers[var].load_ndarray(var)
			get_arrays_elapsed_time.append(timeit.default_timer() - get_arrays_timer)

			get_units_timer = timeit.default_timer()
			unit = wrappers[var].get_units(var)
			get_units_elapsed_time.append(timeit.default_timer() - get_units_timer)
			if unit is None:
				err_msg = f"Input variable '{var}' is missing units."
				logger.error(err_msg, stack_info=True)
				raise ValueError(err_msg)
			if not validate_input_units(var, unit):
				err_msg = f"Invalid input units '{unit}' for variable '{var}'."
				logger.error(err_msg, stack_info=True)
				raise ValueError(err_msg)
			units_cache[var] = unit

		precipitation_batches: dict[
			tuple[tuple[str, ...], tuple[int, int], float],
			list[tuple[type[QuantileIndex], Path]],
		] = {}
		work_items: list[tuple[str, Any]] = []
		for item in pending_outputs:
			index_class, _ = item
			if index_class.index_type != "precipitation_quantile":
				work_items.append(("single", item))
				continue

			batch_key = _build_precipitation_batch_key(index_class, {"pr": units_cache["pr"]})
			if batch_key not in precipitation_batches:
				precipitation_batches[batch_key] = []
				work_items.append(("precipitation_batch", batch_key))
			precipitation_batches[batch_key].append(item)

		start_time_thresholds_comp = timeit.default_timer()
		for work_type, work_item in work_items:
			if work_type == "single":
				index_class, output_path = work_item
				index_obj = index_class(compute_backend, **backend_kwargs)
				arrays = {
					var: arrays_cache[var]
					for var in index_class.required_vars
				}
				units = {
					var: units_cache[var]
					for var in index_class.required_vars
				}

				logger.info("Computing thresholds for %s", index_class.index_id)

				compute_timer = timeit.default_timer()
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
				compute_time = timeit.default_timer() - compute_timer
				compute_elapsed_time.append(compute_time)

				write_timer = timeit.default_timer()
				_write_threshold_netcdf(
					threshold_array=np.asarray(threshold_array),
					index_class=index_class,
					output_path=output_path,
					metadata=meta,
				)
				write_time = timeit.default_timer() - write_timer
				write_elapsed_time.append(write_time)
				threshold_timing_map[index_class.index_id] = compute_time + write_time
				new_files.append(output_path)
				continue

			batch_key = work_item
			batch_items = precipitation_batches[batch_key]
			for index_class, _ in batch_items:
				logger.info("Computing thresholds for %s", index_class.index_id)

			compute_timer = timeit.default_timer()
			batch_thresholds = _compute_precipitation_threshold_batch(
				batch_items=batch_items,
				compute_backend=compute_backend,
				backend_kwargs=backend_kwargs,
				arrays_cache=arrays_cache,
				units_cache=units_cache,
				meta=meta,
				time_groupings=time_groupings,
			)
			compute_time = timeit.default_timer() - compute_timer
			per_quantile_compute_time = compute_time / len(batch_items)
			compute_elapsed_time.extend([per_quantile_compute_time] * len(batch_items))

			for index_class, output_path in batch_items:
				write_timer = timeit.default_timer()
				_write_threshold_netcdf(
					threshold_array=np.asarray(batch_thresholds[index_class.index_id]),
					index_class=index_class,
					output_path=output_path,
					metadata=meta,
				)
				write_time = timeit.default_timer() - write_timer
				write_elapsed_time.append(write_time)
				threshold_timing_map[index_class.index_id] = per_quantile_compute_time + write_time
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

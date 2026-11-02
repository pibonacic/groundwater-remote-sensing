"""Step 1: load the three sources, apply quality control, write the clean daily table."""
import config as cfg
from functions.data import (
    load_piezometer, load_nbar, load_chirps, normalize_piezometer, apply_nbar_qa,
    align_sources, mask_reflectance_outliers, quality_report, interpolate_nbar, save_table)
from functions.plots import quality_overview_plot

SITE = "sdh1"

site = cfg.SITES[SITE]
out_dir = cfg.OUTPUTS_DIR / SITE / "01_quality"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Read the raw files. The piezometer is converted to cm (negative = below surface).
piezo = normalize_piezometer(
    load_piezometer(site["piezo_path"], site["piezo_column"]), site["piezo_unit"], site["piezo_zero_cm"])

# 2. VIIRS NBAR: values are kept as they are; qa_low only marks days with a flag other than the good ones.
nbar = apply_nbar_qa(load_nbar(site["nbar_path"]), site["nbar_qa"])
chirps = load_chirps(site["chirps_path"])

# 3. Join the three sources on one daily index. Gaps stay as gaps (no interpolation).
daily = align_sources(piezo, nbar, chirps)

# 4. Mask reflectance outliers: NBAR values with |z-score| above max_reflectance_z become gaps.
#    Done before the report, so the report counts them as gaps.
bands = nbar.columns.drop("qa_low")
daily, masked = mask_reflectance_outliers(daily, bands, site["max_reflectance_z"])

# 5. Diagnose: coverage, gaps, low-quality days, sudden jumps and out-of-range levels.
#    Nothing else is removed here; you decide from the report.
report = quality_report(daily, "gw_cm", site["physical_range_cm"], site["max_jump_cm"])
report["masked_reflectance"] = masked

# 6. Fill the gaps inside the NBAR band series (time-weighted linear interpolation).
#    Done after the report, so the report describes the original gaps.
filled = interpolate_nbar(daily, bands)

# 7. Save the clean table (input of step 2), the report tables and an overview figure
#    (the figure uses the series before interpolation, so the gaps stay visible).
save_table(filled, cfg.INTERIM_DIR / f"{SITE}_daily.csv")
for name, table in report.items():
    save_table(table, out_dir / f"{name}.csv")
quality_overview_plot(daily, "gw_cm", "I2", f"{SITE}: quality overview").savefig(out_dir / "overview.png", dpi=150)

# 8. Short summary to decide whether the unit, zero and quality rules are right.
print(f"{SITE}: {daily.index.min().date()} to {daily.index.max().date()}, {len(daily)} days")
print(report["coverage"][["first", "last", "n_valid", "n_missing_inside", "longest_gap_days"]].loc[["gw_cm", "I2", "M11", "precip_mm"]])
print("qa_low days by year:\n", report["qa_by_year"])
print(f"jumps > {site['max_jump_cm']} cm: {len(report['jumps'])}; "
      f"outside {site['physical_range_cm']}: {len(report['out_of_range'])}")
print(f"reflectance values masked (|z| > {site['max_reflectance_z']}):\n", masked["band"].value_counts().sort_index().to_string())

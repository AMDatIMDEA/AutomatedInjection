"""
Tensile test post-processing: 0.2 % offset yield stress from raw force /
displacement data, replicating Instron's own calculation.

Per specimen:
    1. Parse the raw force/displacement block from the Instron CSV export.
    2. Instron-style search domain (2 % of max force -> zero-slope region or
       max force), split into 6 equal-stress bands; pick the steepest band
       (highest adjacent-pair slope sum).
    3. Fix the slope to Instron's reported modulus and solve only the
       strain-axis intercept (toe compensation) from that band.
       NO-REFERENCE MODE: if no Instron modulus is available for a specimen
       (or the file has no results table at all), the band's own
       least-squares slope and intercept are used instead.
    4. Offset line  sigma = E * (strain - intercept - 0.002); the yield is
       where the curve crosses DOWN through that line (last downward
       crossing -- see yield_stress_from_modulus for why).

Validation: 16 of 16 specimens for which Instron reports a yield agree
within 0.05 MPa (max error 0.016 MPa) in fixed-slope mode (Instron modulus
used). The Instron-implied intercept is printed next to ours as a check.
The no-reference (self-fit) mode is NOT validated by default: run
main(..., use_reference_modulus=False) on specimens that do have Instron
values to measure how it compares (see METHOD.md section 6).

Confirmed inputs (from the Instron test method / measurements, not guessed):
    cross_section_area = 4.65 * 1.65 mm^2   (measured, post-molding)
    zero-slope search  : n_regions=100, overlap_pct=50, yield_slope_criteria=3
    modulus-region split: 6 regions (Instron's documented default)
Still to confirm: gauge_length (50 vs 60 mm). It DOES affect yield stress,
because the 0.2 % offset is defined in strain units.

Full description of the method, validation and limitations: METHOD.md
"""

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Marker checked by the notebook cell: this version supports modulus=NaN (self-fit mode).
SELF_FIT_SUPPORTED = True

# =============================================================================
# CSV parsing
# =============================================================================

def parse_instron_csv(file_path, max_file_size):
    """
    Split an Instron multi-specimen export into:
      - total_df: the summary "Results Table 1" (one row per specimen,
        with columns like Modulus, Yield stress, etc.)
      - series_blocks: raw per-specimen time/displacement/force blocks

    Parameters
    ----------
    file_path : str
    max_file_size : int
        Max number of specimens expected in the file (used to detect the
        "N,Time..." block-header lines).

    Returns
    -------
    total_df : pd.DataFrame
    series_blocks : list[list[str]]
        Each entry is the raw lines for one specimen (including its own
        2-line header), skipping series_blocks[0] which is pre-data noise.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    block_header_prefixes = tuple(f"{i},Time" for i in range(1, max_file_size + 1))
    series_blocks, current_block = [], []
    for line in lines:
        if line.strip().startswith(block_header_prefixes):
            if current_block:
                series_blocks.append(current_block)
            current_block = []
        current_block.append(line)
    if current_block:
        series_blocks.append(current_block)

    # series_blocks[0] is meant to be the pre-data preamble. A raw-only file may
    # start directly with specimen 1's header; keep an empty placeholder so that
    # specimen is not mistaken for the preamble and silently dropped.
    if series_blocks and series_blocks[0][0].strip().startswith(block_header_prefixes):
        series_blocks.insert(0, [])

    n_specimens = len(series_blocks) - 1

    results_line = next((i for i, line in enumerate(lines) if line.strip() == "Results Table 1"), None)
    if results_line is None:
        # Raw-data-only export: no summary table, so no Instron reference values.
        ids = [int(block[0].strip().split(",")[0]) for block in series_blocks[1:]]
        return pd.DataFrame({"Series_ID": ids}), series_blocks

    start = results_line + 3
    header = lines[start - 2].split(",")
    header[0] = "Series_ID"
    table_lines = lines[start: start + n_specimens]
    total_df = pd.read_csv(StringIO("".join(table_lines)), names=header)

    return total_df, series_blocks


def parse_specimen_block(block):
    """Extract (series_id, df[Time, Displacement, Force]) from one raw block."""
    series_id = block[0].strip().split(",")[0]
    data = "".join(block[2:])
    df = pd.read_csv(StringIO(data), header=None, names=["_", "Time", "Displacement", "Force"])
    df = df.drop(columns=["_"])
    df["Series"] = series_id
    return series_id, df


# =============================================================================
# Instron algorithms: zero-slope search + modulus-region intercept
# =============================================================================

def find_zero_slope_region(displacement, force, n_regions=100, overlap_pct=50,
                            yield_slope_criteria=3, upper_limit_idx=None):
    """
    Replicates Instron's "Zero slope yield" calculation: split displacement
    into equal-size (optionally overlapping) regions, least-squares fit the
    slope of each, and find the first run of `yield_slope_criteria`
    consecutive regions with slope <= 0. The candidate yield index is the
    point of highest force within that qualifying run.

    Defaults (n_regions=100, overlap_pct=50, yield_slope_criteria=3) are
    the CONFIRMED values from this specimen's actual Instron test method
    configuration, not guessed defaults.

    Returns
    -------
    dict with 'yield_idx' (or None if criteria not met), 'region_slopes',
    'region_bounds', 'qualifying_region_start'.
    """
    displacement = np.asarray(displacement, dtype=float)
    force = np.asarray(force, dtype=float)
    n = len(displacement)
    upper_limit_idx = n if upper_limit_idx is None else upper_limit_idx

    disp_search = displacement[:upper_limit_idx]
    force_search = force[:upper_limit_idx]

    d_min, d_max = disp_search.min(), disp_search.max()
    region_width = (d_max - d_min) / n_regions
    step = region_width * (1 - overlap_pct / 100.0)

    region_bounds = []
    start = d_min
    while start + region_width <= d_max + 1e-9:
        region_bounds.append((start, start + region_width))
        start += step
        if step <= 0:
            break

    region_slopes, region_indices = [], []
    for lo, hi in region_bounds:
        idxs = np.where((disp_search >= lo) & (disp_search <= hi))[0]
        if len(idxs) < 2:
            region_slopes.append(np.nan)
            region_indices.append(idxs)
            continue
        slope = np.polyfit(disp_search[idxs], force_search[idxs], 1)[0]
        region_slopes.append(slope)
        region_indices.append(idxs)
    region_slopes = np.array(region_slopes)

    consecutive_count, qualifying_start = 0, None
    for i, s in enumerate(region_slopes):
        if np.isnan(s):
            consecutive_count, qualifying_start = 0, None
            continue
        if s <= 0:
            consecutive_count += 1
            if consecutive_count == 1:
                qualifying_start = i
            if consecutive_count >= yield_slope_criteria:
                break
        else:
            consecutive_count, qualifying_start = 0, None
    else:
        qualifying_start = None

    if qualifying_start is None:
        return {"yield_idx": None, "region_slopes": region_slopes, "region_bounds": region_bounds}

    qualifying_range = range(qualifying_start, qualifying_start + yield_slope_criteria)
    candidate_idxs = np.concatenate([region_indices[i] for i in qualifying_range
                                      if len(region_indices[i]) > 0])
    yield_idx = candidate_idxs[np.argmax(force_search[candidate_idxs])]

    return {
        "yield_idx": yield_idx,
        "region_slopes": region_slopes,
        "region_bounds": region_bounds,
        "qualifying_region_start": qualifying_start,
    }


def find_instron_style_intercept(force, displacement, gauge_length, cross_section_area,
                                  known_modulus=None, force_to_N=1000.0, n_regions=6,
                                  zero_slope_kwargs=None, verbose=False):
    """
    Approximates Instron's modulus-region search to derive a strain-axis
    intercept (toe compensation).

    known_modulus : float or None
        Instron's reported modulus. If given, it is used as a FIXED slope
        (only the intercept is solved for) so the result stays consistent
        with Instron's own value. If None/NaN (no reference available), the
        selected region's own least-squares slope and intercept are used.

    Parameters
    ----------
    zero_slope_kwargs : dict, optional
        Passed through to find_zero_slope_region (n_regions, overlap_pct,
        yield_slope_criteria). Uses that function's defaults if omitted
        -- which are now the CONFIRMED Instron values (100/50/3).
    verbose : bool, default False
        Print region slopes/selection diagnostics.

    Returns
    -------
    dict: strain_intercept, region_bounds_stress, region_index,
    region_strain, region_stress, self_fit_modulus, search_domain_end_idx,
    zero_slope_idx, max_force_idx
    """
    strain = np.asarray(displacement, dtype=float) / gauge_length
    stress = (np.asarray(force, dtype=float) * force_to_N) / cross_section_area
    force_N = np.asarray(force, dtype=float) * force_to_N

    order = np.argsort(strain)
    strain, stress, force_N = strain[order], stress[order], force_N[order]
    displacement_sorted = np.asarray(displacement, dtype=float)[order]

    # --- Search domain start: 2% of max force ---
    max_force = force_N.max()
    start_idx = np.argmax(force_N >= 0.02 * max_force)
    max_force_idx = np.argmax(force_N)

    # --- Search domain end: zero-slope region point, or max force, whichever first ---
    zs_kwargs = zero_slope_kwargs or {}
    zs_result = find_zero_slope_region(displacement_sorted, force_N, **zs_kwargs)
    zero_slope_idx = zs_result["yield_idx"]
    end_idx = max_force_idx if zero_slope_idx is None else min(zero_slope_idx, max_force_idx)

    if end_idx <= start_idx:
        raise ValueError("Computed end index occurs before start — check data/region settings")

    seg_strain = strain[start_idx:end_idx + 1]
    seg_stress = stress[start_idx:end_idx + 1]

    # --- Split into n_regions equal-stress regions, least-squares fit each ---
    stress_bounds = np.linspace(seg_stress.min(), seg_stress.max(), n_regions + 1)
    region_slopes, region_data = [], []
    for i in range(n_regions):
        lo, hi = stress_bounds[i], stress_bounds[i + 1]
        mask = (seg_stress >= lo) & (seg_stress <= hi)
        if mask.sum() < 2:
            region_slopes.append(np.nan)
            region_data.append((None, None))
            continue
        slope, _ = np.polyfit(seg_strain[mask], seg_stress[mask], 1)
        region_slopes.append(slope)
        region_data.append((seg_strain[mask], seg_stress[mask]))
    region_slopes = np.array(region_slopes)

    # --- Highest slope-sum consecutive pair, then the higher-slope region within it ---
    pair_sums = np.array([region_slopes[i] + region_slopes[i + 1] for i in range(n_regions - 1)])
    if np.all(np.isnan(pair_sums)):
        raise ValueError("Could not compute valid slopes in any region pair")
    best_pair_idx = np.nanargmax(pair_sums)
    idx_a, idx_b = best_pair_idx, best_pair_idx + 1
    selected_region = idx_a if region_slopes[idx_a] >= region_slopes[idx_b] else idx_b

    sel_strain, sel_stress = region_data[selected_region]
    if sel_strain is None:
        raise ValueError("Selected region has insufficient data")

    # --- Fixed slope (Instron modulus) if available, else the band's own fit ---
    self_slope, self_icpt = np.polyfit(sel_strain, sel_stress, 1)
    if known_modulus is None or not np.isfinite(known_modulus):
        modulus_used = self_slope
        strain_intercept = -self_icpt / self_slope
        modulus_source = "self-fit"
    else:
        modulus_used = float(known_modulus)
        b = np.mean(sel_stress - modulus_used * sel_strain)
        strain_intercept = -b / modulus_used
        modulus_source = "Instron"

    if verbose:
        print("Region slopes:", np.round(region_slopes, 2))
        print("Pair sums:     ", np.round(pair_sums, 2))
        print(f"Selected region: {selected_region} (pair: {best_pair_idx},{best_pair_idx + 1})")
        print(f"Search domain ended at index: {end_idx} "
              f"(zero-slope: {zero_slope_idx}, max force: {max_force_idx})")
        print(f"Self-fit modulus in selected region: {self_slope:.2f} "
              f"(modulus used: {modulus_used:.2f}, {modulus_source})")
        print(f"Strain intercept: {strain_intercept:.6f}")

    return {
        "strain_intercept": strain_intercept,
        "region_bounds_stress": stress_bounds,
        "region_index": selected_region,
        "region_strain": sel_strain,
        "region_stress": sel_stress,
        "self_fit_modulus": self_slope,
        "modulus_used": modulus_used,
        "modulus_source": modulus_source,
        "search_domain_end_idx": end_idx,
        "zero_slope_idx": zero_slope_idx,
        "max_force_idx": max_force_idx,
    }


# =============================================================================
# 0.2 % offset yield calculation
# =============================================================================

def yield_stress_from_modulus(force, displacement, modulus, strain_intercept,
                               gauge_length, cross_section_area, offset=0.002,
                               force_to_N=1000.0, return_details=False):
    """
    0.2 % offset yield stress, given Instron's modulus and the strain-axis
    intercept (toe compensation) derived to be consistent with it.

    The offset line is  sigma = modulus * (strain - strain_intercept - offset).
    The yield point is where the curve crosses DOWN through this line.

    The crossing is taken as the LAST point of the curve that lies above the
    line (interpolated to the next point). Reasons:
      * After true yielding the curve never climbs back above the line (its
        slope is far below E), so this equals the first downward crossing on
        clean data.
      * It ignores UPWARD crossings. When the strain intercept is negative
        (specimen already loaded at displacement zero: series 12/14/15) the
        offset line starts ABOVE the curve and the curve first climbs up
        through it. An earlier version counted that as yield and returned
        0.3-0.9 MPa instead of 2.5-5 MPa.
      * It is not fooled by a slip (brief dip below the line, then recovery)
        or by noise chattering around an upward crossing.
    """
    strain = np.asarray(displacement, dtype=float) / gauge_length
    stress = (np.asarray(force, dtype=float) * force_to_N) / cross_section_area

    order = np.argsort(strain)
    strain, stress = strain[order], stress[order]
    strain_corrected = strain - strain_intercept

    offset_line = modulus * (strain_corrected - offset)
    diff = stress - offset_line

    start_idx = np.searchsorted(strain_corrected, offset)
    above = np.where(diff[start_idx:] > 0)[0]
    if above.size == 0:
        raise ValueError("Curve never rises above the offset line -- check modulus/intercept.")
    idx = start_idx + int(above[-1])
    if idx + 1 >= len(diff):
        raise ValueError("No intersection found between offset line and curve.")

    x0, x1 = strain_corrected[idx], strain_corrected[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]                    # d0 > 0 >= d1
    yield_strain = x0 - d0 * (x1 - x0) / (d1 - d0)
    yield_stress = np.interp(yield_strain, strain_corrected, stress)

    if not return_details:
        return yield_stress

    return {
        "yield_stress": yield_stress,
        "yield_strain": yield_strain,
        "strain": strain_corrected,
        "stress": stress,
        "modulus": modulus,
        "offset": offset,
        "strain_intercept": strain_intercept,
    }


def instron_implied_intercept(force, displacement, modulus, instron_yield,
                              gauge_length, cross_section_area, offset=0.002,
                              force_to_N=1000.0):
    """
    Validation helper: the strain intercept Instron MUST have used to report
    `instron_yield` with `modulus`. The yield point (eps_y, sigma_y) lies on
    the offset line, so  eps_int = eps_y - offset - sigma_y / E.
    Evaluated on the loading part of the curve (up to max force) in time
    order. Returns NaN if Instron gave no yield or the curve never reaches it.
    """
    d = np.asarray(displacement, dtype=float)
    f = np.asarray(force, dtype=float) * force_to_N
    n = int(np.argmax(f))
    eps, sig = d[:n + 1] / gauge_length, f[:n + 1] / cross_section_area
    if not (np.isfinite(instron_yield) and np.isfinite(modulus)) or sig.max() < instron_yield:
        return np.nan
    i = int(np.argmax(sig >= instron_yield))
    if i == 0 or sig[i] == sig[i - 1]:
        eps_y = eps[i]
    else:
        eps_y = eps[i - 1] + (instron_yield - sig[i - 1]) / (sig[i] - sig[i - 1]) * (eps[i] - eps[i - 1])
    return float(eps_y - offset - instron_yield / modulus)


def plot_yield_offset(result, ax=None, units=("mm/mm", "MPa"), title="0.2% Offset Yield Strength"):
    """Plot the stress-strain curve, offset line, and yield point from yield_stress_from_modulus(..., return_details=True)."""
    strain, stress = result["strain"], result["stress"]
    modulus, offset = result["modulus"], result["offset"]
    yield_strain, yield_stress = result["yield_strain"], result["yield_stress"]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    strain_unit, stress_unit = units

    ax.plot(strain, stress, color="tab:blue", lw=1.5, label="Stress-strain curve")

    x_max_plot = min(strain.max(), yield_strain * 1.5 if yield_strain > 0 else strain.max())
    x_offset_line = np.linspace(offset, x_max_plot, 100)
    y_offset_line = modulus * (x_offset_line - offset)
    valid = y_offset_line >= 0
    ax.plot(x_offset_line[valid], y_offset_line[valid], color="tab:green", lw=1.5, ls="--",
            label=f"{offset * 100:.1f}% offset line (E = {modulus:.3g} {stress_unit})")

    diag = result.get("intercept_diagnostics")
    if diag is not None and diag.get("region_strain") is not None:
        ax.plot(np.asarray(diag["region_strain"]) - result["strain_intercept"],
                diag["region_stress"], color="tab:orange", lw=3.5, alpha=0.8,
                label="Region used for intercept")

    ax.scatter([yield_strain], [yield_stress], color="red", zorder=5, s=60,
               label=f"Yield point ({yield_strain:.4f}, {yield_stress:.1f} {stress_unit})")
    ax.axhline(yield_stress, color="red", lw=0.6, ls=":")
    ax.axvline(yield_strain, color="red", lw=0.6, ls=":")

    ax.set_xlabel(f"Strain [{strain_unit}]" if strain_unit else "Strain")
    ax.set_ylabel(f"Stress [{stress_unit}]" if stress_unit else "Stress")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, stress.max() * 1.05)
    ax.set_xlim(0, strain.max() * 1.05)
    return ax


# =============================================================================
# Per-specimen orchestration
# =============================================================================

def process_specimen(df, modulus, gauge_length, cross_section_area, offset=0.002,
                      intercept_kwargs=None, verbose=False):
    """
    Full intercept + yield pipeline for one specimen's raw df (columns:
    Displacement, Force). `modulus` is Instron's modulus, or None/NaN when no
    reference is available (then the selected region's own fitted slope is
    used). Returns the yield_stress_from_modulus detail dict with the
    intercept diagnostics merged in under 'intercept_diagnostics', plus
    'modulus_source' ("Instron" or "self-fit").
    """
    intercept_result = find_instron_style_intercept(
        force=df.Force, displacement=df.Displacement,
        gauge_length=gauge_length, cross_section_area=cross_section_area,
        known_modulus=modulus, verbose=verbose,
        **(intercept_kwargs or {}),
    )
    yield_result = yield_stress_from_modulus(
        force=df.Force, displacement=df.Displacement,
        modulus=intercept_result["modulus_used"],
        strain_intercept=intercept_result["strain_intercept"],
        gauge_length=gauge_length, cross_section_area=cross_section_area,
        offset=offset, return_details=True,
    )
    yield_result["intercept_diagnostics"] = intercept_result
    yield_result["modulus_source"] = intercept_result["modulus_source"]
    return yield_result


# =============================================================================
# Main pipeline
# =============================================================================

def _fmt(v, nd=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _table_value(row, column):
    """Value from the Instron results-table row, or NaN if the column/row is absent or '---'."""
    if row.empty or column not in row.columns:
        return np.nan
    return pd.to_numeric(row[column].iloc[0], errors="coerce")


MODULUS_COL = "Modulus (Automatic Young's)"
YIELD_COL = "Tensile stress at Yield (Offset 0.2 %)"


def main(files=None,
         guide_file="Versaloid_TPU_190_guide.csv",
         max_file_size=19,
         gauge_length=50.0,                 # mm -- confirm (50 vs 60 used interchangeably before)
         cross_section_area=4.65 * 1.65,    # mm^2 -- measured specimen dimensions
         plot_mode="flagged",               # "all" | "flagged" | "none"
         use_reference_modulus=True):
    """
    Works with or without Instron reference values.

    * Instron modulus available (and use_reference_modulus=True): fixed-slope
      mode -- the validated configuration.
    * No Instron modulus for a specimen (or no results table in the file):
      the selected region's own fitted slope/intercept are used ("self-fit").
    * Instron yield is only ever used for comparison, never for the calculation.

    use_reference_modulus=False is a BLIND TEST: Instron's modulus is ignored
    so every specimen is calculated in self-fit mode, while Instron's yield is
    still shown for comparison. Run it on your validated data to see how much
    self-fit mode differs from Instron before trusting it on data with no
    reference.

    plot_mode="flagged" plots only specimens that need a look: the calculation
    failed, our value differs from Instron's by > 0.1 MPa, Instron gave no
    yield for a specimen while it did for others in the file, the intercept is
    negative, or (fixed-slope mode) the self-fit modulus is > 5 % off Instron's.
    """
    files = files or ["Versaloid_TPU_190_new.csv", "Versaloid_TPU_190_new_2.csv"]
    zero_slope_kwargs = {"n_regions": 100, "overlap_pct": 50, "yield_slope_criteria": 3}

    if not use_reference_modulus:
        print("BLIND TEST: Instron modulus ignored, all specimens use self-fit mode.\n")

    all_dfs, comparison = [], []
    for file_path in files:
        total_df, series_blocks = parse_instron_csv(file_path, max_file_size)
        records = {}

        # Does this file carry reference values at all?
        file_has_yield_ref = (YIELD_COL in total_df.columns
                              and pd.to_numeric(total_df[YIELD_COL], errors="coerce").notna().any())
        file_has_modulus_ref = (MODULUS_COL in total_df.columns
                                and pd.to_numeric(total_df[MODULUS_COL], errors="coerce").notna().any())
        if not file_has_yield_ref and not file_has_modulus_ref:
            print(f"[{file_path}] no Instron reference values found -> self-fit mode, no comparison.")

        for block in series_blocks[1:]:
            series_id, df = parse_specimen_block(block)
            sid = int(series_id)
            row = total_df.loc[total_df["Series_ID"] == sid]
            modulus_ref = _table_value(row, MODULUS_COL)
            instron_yield = _table_value(row, YIELD_COL)
            modulus_in = modulus_ref if use_reference_modulus else np.nan

            df["Clean_displacement"] = np.where(df["Force"] > 0.01, df["Displacement"], 0)

            result, error = None, None
            try:
                result = process_specimen(
                    df, modulus_in, gauge_length, cross_section_area,
                    intercept_kwargs={"zero_slope_kwargs": zero_slope_kwargs})
            except ValueError as e:
                error = str(e)

            my_yield = result["yield_stress"] if result else np.nan
            intercept = result["strain_intercept"] if result else np.nan
            modulus_used = result["modulus"] if result else np.nan
            source = result["modulus_source"] if result else "n/a"
            implied = instron_implied_intercept(
                df.Force, df.Displacement, modulus_ref, instron_yield, gauge_length, cross_section_area)

            # ---- console report + flags ----
            notes = []
            if error:
                notes.append(f"FAILED: {error}")
            if result:
                if source == "Instron":
                    self_fit = result["intercept_diagnostics"]["self_fit_modulus"]
                    if abs(self_fit - modulus_ref) / modulus_ref > 0.05:
                        notes.append(f"self-fit modulus {self_fit:.1f} differs >5 % from Instron's {modulus_ref:.1f}")
                elif use_reference_modulus and file_has_modulus_ref:
                    notes.append("no Instron modulus for this specimen -> self-fit modulus used")
                if intercept < 0:
                    notes.append("negative strain intercept")
            if np.isfinite(instron_yield) and np.isfinite(my_yield) and abs(my_yield - instron_yield) > 0.1:
                notes.append(f"differs from Instron by {my_yield - instron_yield:+.2f} MPa")
            if file_has_yield_ref and not np.isfinite(instron_yield):
                notes.append("Instron reported no yield")

            line = f"Series {sid:>2} ({file_path}): yield {_fmt(my_yield)} MPa"
            if np.isfinite(instron_yield):
                line += f" (Instron {_fmt(instron_yield)})"
            line += f" | E {_fmt(modulus_used, 1)} ({source}) | intercept {_fmt(intercept, 5)}"
            if np.isfinite(implied):
                line += f" (Instron-implied {_fmt(implied, 5)})"
            print(line)
            for n in notes:
                print(f"      note: {n}")

            if result and (plot_mode == "all" or (plot_mode == "flagged" and notes)):
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_yield_offset(result, ax=ax, title=f"Series {sid}" + (f" [{'; '.join(notes)}]" if notes else ""))
                plt.tight_layout()
                plt.show()

            records[sid] = {
                "Elongation at break (%)": df["Clean_displacement"].max() * 100 / gauge_length,
                "Maximum force (kN)": df["Force"].max(),
                "Yield stress recalculated (MPa)": my_yield,
                "Modulus used (MPa)": modulus_used,
                "Modulus source": source,
                "Strain intercept": intercept,
                "Instron-implied intercept": implied,
            }
            comparison.append((my_yield, instron_yield, source))

        extra = pd.DataFrame.from_dict(records, orient="index")
        for col in extra.columns:
            total_df[col] = total_df["Series_ID"].map(extra[col])
        all_dfs.append(total_df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    cols = [c for c in (YIELD_COL, MODULUS_COL) if c in final_df.columns]
    final_df[cols] = final_df[cols].apply(pd.to_numeric, errors="coerce")

    # ---- summary ----
    comp = pd.DataFrame(comparison, columns=["mine", "instron", "source"])
    n_ok = int(comp["mine"].notna().sum())
    print("\n" + "=" * 70)
    print(f"{n_ok} of {len(comp)} specimens calculated "
          f"({int((comp['source'] == 'Instron').sum())} fixed-slope, "
          f"{int((comp['source'] == 'self-fit').sum())} self-fit).")
    ref = comp[comp["instron"].notna() & comp["mine"].notna()]
    if len(ref):
        err = (ref["mine"] - ref["instron"]).abs()
        print(f"Agreement with Instron on {len(ref)} specimens it calculated: "
              f"{int((err <= 0.05).sum())} within 0.05 MPa | "
              f"median |err| {_fmt(err.median(), 3)} | max |err| {_fmt(err.max(), 3)}")
    else:
        print("No Instron yield values to compare against.")
    print("=" * 70)

    if guide_file:
        final_df = pd.concat([final_df, pd.read_csv(guide_file)], axis=1)
    return final_df


if __name__ == "__main__":
    final_df = main()





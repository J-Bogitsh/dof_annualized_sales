"""
ingestion.py
------------
Pulls and consolidates the three NYC DOF source datasets into clean parquet
files for downstream analysis:

  1. NYC Citywide Annualized Calendar Sales Update (Socrata: w2pb-icbu)
  2. NYC Calendar Sales Archive 2003-2015 (local CSVs already compiled)
  3. Property Valuation and Assessment Data, Tax Classes 1-4 (Socrata: 8y4t-faws)

Run sections independently by toggling the flags in `main()` — the
valuation pull is the slow one (11.7M rows) so we don't want to
re-run it every time we're just testing the sales/archive logic.

Requires: polars, requests
    pip install polars requests
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Paste NYC Open Data app token here, or set the NYC_OPEN_DATA_APP_TOKEN
# environment variable instead (preferred — keeps the token out of the file).
# Get one at: https://data.cityofnewyork.us/profile/app_tokens
APP_TOKEN = os.environ.get("NYC_OPEN_DATA_APP_TOKEN")

SOCRATA_DOMAIN = "data.cityofnewyork.us"
CALENDAR_SALES_ID = "w2pb-icbu"       # NYC Citywide Annualized Calendar Sales Update
PROPERTY_VALUATION_ID = "8y4t-faws"   # Property Valuation and Assessment Data
PLUTO_ID = "64uk-42ks"                # Primary Land Use Tax Lot Output (PLUTO)

OUTPUT_DIR = Path("./data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Local archive files already compiled.
# ARCHIVE_PATHS = {local_path}

# Confirmed column names, taken directly from the official DOF data
# dictionary. Socrata usually lowercases column names when it turns them
# into API field names (e.g. YEAR -> year), but double check this against
# data/processed/valuation_data_dictionary.csv
# the first time we run fetch_socrata_metadata below — if any of these
# don't match, that CSV has the real ones.
CANDIDATE_VALUATION_COLUMNS = [
    # identifiers -- used to match a valuation record to a sale
    "parid", "boro", "block", "lot", "easement", "year", "rectype",

    # property characteristics -- used to fill in gaps in the sales data
    "bldg_class", "yrbuilt", "land_area", "gross_sqft",
    "units", "coop_apts", "num_bldgs",
    "lot_frt", "lot_dep", "bld_frt", "bld_dep",

    # market value fields. FIN = Final Roll, the official settled value
    # for that tax year -- this is the number to compare sale price against.
    "fintaxclass", "finmktland", "finmkttot",

    # "period" tells you how far that year's roll had progressed when this
    # row was recorded (0 = just started, 3 = final roll published) --
    # useful for knowing whether finmkttot is even populated yet for that row
    "period",
]

# PLUTO columns, confirmed against the real DCP data dictionary
# Note: PLUTO does NOT have a BIN or NTA field
# those two will stay blank for the archive sales rows.
CANDIDATE_PLUTO_COLUMNS = [
    # identifiers -- used to match a PLUTO record to a sale
    "bbl", "borough", "block", "lot", "address",

    # geographic fields -- the whole reason we're pulling PLUTO
    "latitude", "longitude", "cd", "council", "zipcode", "bct2020",

    # property characteristics -- useful as a backup/cross-check alongside
    # the valuation data for filling in missing sales fields
    "bldgclass", "yearbuilt", "lotarea", "bldgarea",
    "unitsres", "unitstotal", "numbldgs",
]

REQUEST_TIMEOUT = 60
PAGE_SIZE_SMALL = 50_000    # for the ~846K-row sales dataset
PAGE_SIZE_LARGE = 50_000    # for the ~11.7M-row valuation dataset


# ---------------------------------------------------------------------------
# SOCRATA HELPERS
# ---------------------------------------------------------------------------

def _socrata_headers() -> dict:
    if APP_TOKEN in ("", "PASTE_YOUR_TOKEN_HERE"):
        print("WARNING: no Socrata app token set — you'll hit strict rate limits.")
        return {}
    return {"X-App-Token": APP_TOKEN}


def fetch_socrata_metadata(dataset_id: str, out_name: str) -> pl.DataFrame:
    """Pull the column-level metadata for a dataset and write it as a
    data dictionary CSV. Use this to confirm real field names before
    a large pull."""
    url = f"https://{SOCRATA_DOMAIN}/api/views/{dataset_id}.json"
    resp = requests.get(url, headers=_socrata_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    cols = resp.json().get("columns", [])
    df = pl.DataFrame([
        {
            "name": c.get("name"),
            "fieldName": c.get("fieldName"),
            "dataTypeName": c.get("dataTypeName"),
            "description": c.get("description"),
        }
        for c in cols
    ])
    out_path = OUTPUT_DIR / f"{out_name}_data_dictionary.csv"
    df.write_csv(out_path)
    print(f"Wrote data dictionary ({len(df)} columns) -> {out_path}")
    return df


def fetch_socrata_dataset(
    dataset_id: str,
    select_cols: list[str] | None = None,
    page_size: int = PAGE_SIZE_SMALL,
    max_retries: int = 3,
) -> pl.DataFrame:
    """Paginate through a Socrata dataset via SoQL and return a single
    polars DataFrame. Set select_cols to limit payload size on large tables."""
    base_url = f"https://{SOCRATA_DOMAIN}/resource/{dataset_id}.json"
    headers = _socrata_headers()

    frames = []
    offset = 0
    while True:
        params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
        if select_cols:
            params["$select"] = ",".join(select_cols)

        for attempt in range(max_retries):
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  retrying after error: {e}")
                time.sleep(2 ** attempt)

        batch = resp.json()
        if not batch:
            break

        frames.append(pl.DataFrame(batch, infer_schema_length=len(batch)))
        offset += page_size
        print(f"  fetched {offset:,} rows so far...")

        if len(batch) < page_size:
            break

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# STEP 1 — CURRENT / ANNUALIZED CALENDAR SALES (Socrata, ~846K rows)
# ---------------------------------------------------------------------------

def pull_calendar_sales_annualized() -> pl.DataFrame:
    print("Pulling NYC Citywide Annualized Calendar Sales Update...")
    df = fetch_socrata_dataset(CALENDAR_SALES_ID, page_size=PAGE_SIZE_SMALL)
    out_path = OUTPUT_DIR / "sales_annualized_raw.parquet"
    df.write_parquet(out_path)
    print(f"  {len(df):,} rows -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — ARCHIVE SALES (local CSVs, 2003-2015)
# ---------------------------------------------------------------------------

def load_and_standardize_archive() -> pl.DataFrame:
    print("Loading and standardizing archive sales files...")

    def _standardize(path: Path) -> pl.DataFrame:
        df = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)
        # normalize header text: strip, lowercase, spaces/hyphens -> underscore
        rename_map = {
            c: c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        }
        return df.rename(rename_map)

    archive_frames = {}
    for label, path in ARCHIVE_PATHS.items():
        if not path.exists():
            print(f"  SKIPPING {label}: file not found at {path}")
            continue
        archive_frames[label] = _standardize(path)
        print(f"  {label}: {archive_frames[label].shape[0]:,} rows, "
              f"{archive_frames[label].shape[1]} columns")

    if not archive_frames:
        raise FileNotFoundError("No archive files found — check ARCHIVE_PATHS.")

    # schema check before concatenating
    col_sets = {label: set(df.columns) for label, df in archive_frames.items()}
    all_cols = set.union(*col_sets.values())
    for label, cols in col_sets.items():
        missing = all_cols - cols
        if missing:
            print(f"  NOTE: {label} is missing columns present elsewhere: {missing}")

    combined = pl.concat(list(archive_frames.values()), how="diagonal_relaxed")
    out_path = OUTPUT_DIR / "sales_archive_raw.parquet"
    combined.write_parquet(out_path)
    print(f"  combined archive: {len(combined):,} rows -> {out_path}")
    return combined


# ---------------------------------------------------------------------------
# STEP 3 — MASTER SALES FILE (archive + annualized, schema-aligned)
# ---------------------------------------------------------------------------

def build_master_sales_file() -> pl.DataFrame:
    print("Building master sales file (archive + annualized)...")
    archive = pl.read_parquet(OUTPUT_DIR / "sales_archive_raw.parquet")
    current = pl.read_parquet(OUTPUT_DIR / "sales_annualized_raw.parquet")

    # The archive and current datasets use different names for a few columns
    # that mean the same thing -- either DOF relabeled them over the years,
    # or Socrata truncated a long name. Rename both sides to one shared,
    # readable name before combining, so they land in the same column
    # instead of showing up as two separate half-empty columns.
    archive_rename = {
        "tax_class_at_present": "tax_class_as_of_final_roll",
        "building_class_at_present": "building_class_as_of_final",
    }
    current_rename = {
        "building_class_at_time_of": "building_class_at_time_of_sale",
    }
    archive = archive.rename(archive_rename)
    current = current.rename(current_rename)

    archive_cols, current_cols = set(archive.columns), set(current.columns)
    only_in_archive = archive_cols - current_cols
    only_in_current = current_cols - archive_cols
    if only_in_archive:
        print(f"  columns only in archive: {only_in_archive}")
    if only_in_current:
        print(f"  columns only in annualized: {only_in_current}")

    master = pl.concat([archive, current], how="diagonal_relaxed")
    out_path = OUTPUT_DIR / "sales_master.parquet"
    master.write_parquet(out_path)
    print(f"  master sales file: {len(master):,} rows -> {out_path}")
    return master


# ---------------------------------------------------------------------------
# STEP 4 — PROPERTY VALUATION (Socrata, ~11.7M rows — streamed to parquet)
# ---------------------------------------------------------------------------

def check_valuation_year_coverage(year_col: str = "year") -> None:
    """Cheap single-request check of how many distinct roll years this
    endpoint actually covers, and the row count per year. Run this BEFORE
    the full pull — if it only covers recent years, DOF likely publishes
    an older assessment roll as a separate archived dataset, the same way
    the sales data was split into archive vs. current."""
    print(f"Checking year coverage on {PROPERTY_VALUATION_ID} (column: '{year_col}')...")
    base_url = f"https://{SOCRATA_DOMAIN}/resource/{PROPERTY_VALUATION_ID}.json"
    params = {
        "$select": f"{year_col}, count(*) as n",
        "$group": year_col,
        "$order": year_col,
    }
    resp = requests.get(base_url, params=params, headers=_socrata_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        print(f"  No results — '{year_col}' may not be the right field name. "
              f"Check valuation_data_dictionary.csv for the actual year/roll-year column.")
        return
    for r in rows:
        print(f"  {r.get(year_col)}: {r.get('n')} rows")
    years = sorted(r.get(year_col) for r in rows if r.get(year_col))
    print(f"  Coverage: {years[0]} to {years[-1]} ({len(years)} distinct years)")


def check_valuation_columns(select_cols: list[str] = CANDIDATE_VALUATION_COLUMNS) -> list[str]:
    """Compares the columns we're trying to pull against the real list of
    valid field names in valuation_data_dictionary.csv (written by
    fetch_socrata_metadata). Prints anything that doesn't actually exist,
    with its best guess at the correct name. Run this whenever a pull
    fails with a 400 Bad Request -- that error means one of the requested
    column names is wrong."""
    dict_path = OUTPUT_DIR / "valuation_data_dictionary.csv"
    if not dict_path.exists():
        print(f"  No data dictionary found at {dict_path}. "
              f"Run fetch_socrata_metadata(PROPERTY_VALUATION_ID, 'valuation') first.")
        return select_cols

    real_dict = pl.read_csv(dict_path)
    real_names = set(real_dict["fieldName"].to_list())

    bad_cols = [c for c in select_cols if c not in real_names]
    if not bad_cols:
        print("  All candidate columns are valid — the 400 error must be something else.")
        return select_cols

    print(f"  Found {len(bad_cols)} column name(s) that don't exist in the real dataset:")
    for bad in bad_cols:
        # simple guess: any real column name that contains our guess, or vice versa
        close_matches = [n for n in real_names if bad in n or n in bad]
        if close_matches:
            print(f"    '{bad}' is not valid. Possible matches: {close_matches}")
        else:
            print(f"    '{bad}' is not valid, and no similar name was found. "
                  f"Check valuation_data_dictionary.csv manually for this one.")
    return bad_cols


def pull_property_valuation(select_cols: list[str] = CANDIDATE_VALUATION_COLUMNS) -> None:
    """Streams the valuation pull directly to parquet in chunks rather than
    holding 11.7M rows in memory at once."""
    print("Pulling Property Valuation and Assessment Data (this will take a while)...")
    base_url = f"https://{SOCRATA_DOMAIN}/resource/{PROPERTY_VALUATION_ID}.json"
    headers = _socrata_headers()

    offset = 0
    chunk_idx = 0
    chunk_dir = OUTPUT_DIR / "valuation_chunks"
    chunk_dir.mkdir(exist_ok=True)

    while True:
        params = {
            "$select": ",".join(select_cols),
            "$limit": PAGE_SIZE_LARGE,
            "$offset": offset,
            "$order": ":id",
        }
        resp = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        chunk_df = pl.DataFrame(batch, infer_schema_length=len(batch))
        chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.parquet"
        chunk_df.write_parquet(chunk_path)

        offset += PAGE_SIZE_LARGE
        chunk_idx += 1
        print(f"  {offset:,} rows fetched...")

        if len(batch) < PAGE_SIZE_LARGE:
            break

    stitch_valuation_chunks()


def stitch_valuation_chunks() -> None:
    """Combines all the small downloaded chunk files into one parquet file.
    Separated out from pull_property_valuation so you can re-run just this
    step (fast) without re-downloading everything (slow) if stitching fails.

    missing_columns='insert' handles a real quirk of this data source: if a
    column is empty for every row in a given 50,000-row batch, that batch's
    file won't have the column at all. This tells polars to just fill it in
    with blanks for that file instead of erroring out."""
    chunk_dir = OUTPUT_DIR / "valuation_chunks"
    lazy = pl.scan_parquet(chunk_dir / "*.parquet", missing_columns="insert")
    out_path = OUTPUT_DIR / "valuation_master.parquet"
    lazy.sink_parquet(out_path)
    print(f"  valuation pull complete -> {out_path}")


def check_pluto_columns(select_cols: list[str] = CANDIDATE_PLUTO_COLUMNS) -> list[str]:
    """Same idea as check_valuation_columns: compares our guessed PLUTO
    column names against the real list in pluto_data_dictionary.csv, and
    prints out anything that's wrong along with a best guess at the fix."""
    dict_path = OUTPUT_DIR / "pluto_data_dictionary.csv"
    if not dict_path.exists():
        print(f"  No data dictionary found at {dict_path}. "
              f"Run fetch_socrata_metadata(PLUTO_ID, 'pluto') first.")
        return select_cols

    real_dict = pl.read_csv(dict_path)
    real_names = set(real_dict["fieldName"].to_list())

    bad_cols = [c for c in select_cols if c not in real_names]
    if not bad_cols:
        print("  All candidate PLUTO columns are valid.")
        return select_cols

    print(f"  Found {len(bad_cols)} column name(s) that don't exist in the real dataset:")
    for bad in bad_cols:
        close_matches = [n for n in real_names if bad in n or n in bad]
        if close_matches:
            print(f"    '{bad}' is not valid. Possible matches: {close_matches}")
        else:
            print(f"    '{bad}' is not valid, and no similar name was found. "
                  f"Check pluto_data_dictionary.csv manually for this one.")
    return bad_cols


def pull_pluto(select_cols: list[str] = CANDIDATE_PLUTO_COLUMNS) -> pl.DataFrame:
    """Pulls PLUTO -- a single current snapshot (not per-year like
    valuation), ~859K rows, one row per tax lot. Small enough to pull in
    one go like the sales data, no chunking needed."""
    print("Pulling PLUTO (Primary Land Use Tax Lot Output)...")
    df = fetch_socrata_dataset(PLUTO_ID, select_cols=select_cols, page_size=PAGE_SIZE_SMALL)
    out_path = OUTPUT_DIR / "pluto_raw.parquet"
    df.write_parquet(out_path)
    print(f"  {len(df):,} rows -> {out_path}")
    return df


def build_bbl(df: pl.DataFrame, boro_col="boro", block_col="block", lot_col="lot") -> pl.DataFrame:
    """Constructs the standard 10-digit BBL (borough + 5-digit block +
    4-digit lot) for joining valuation data to sales data, which already
    has a native `bbl` field."""
    return df.with_columns(
        (
            pl.col(boro_col).cast(pl.Utf8)
            + pl.col(block_col).cast(pl.Utf8).str.zfill(5)
            + pl.col(lot_col).cast(pl.Utf8).str.zfill(4)
        ).alias("bbl_constructed")
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Toggle these as needed — the valuation pull is the expensive one.
    RUN_METADATA_PULL = True
    RUN_SALES_ANNUALIZED = True
    RUN_ARCHIVE = True
    RUN_MASTER_SALES = True
    RUN_VALUATION = False  # flip to True once confirmed CANDIDATE_VALUATION_COLUMNS
    RUN_YEAR_CHECK = True  # run this BEFORE flipping RUN_VALUATION on
    RUN_PLUTO = False  # flip to True once confirmed CANDIDATE_PLUTO_COLUMNS

    if RUN_METADATA_PULL:
        fetch_socrata_metadata(CALENDAR_SALES_ID, "sales")
        fetch_socrata_metadata(PROPERTY_VALUATION_ID, "valuation")
        fetch_socrata_metadata(PLUTO_ID, "pluto")

    if RUN_YEAR_CHECK:
        # after seeing valuation_data_dictionary.csv, confirm this is the
        # right field name (it may be "yrbuilt", "avroll", "period", etc.)
        check_valuation_year_coverage(year_col="year")

    if RUN_SALES_ANNUALIZED:
        pull_calendar_sales_annualized()

    if RUN_ARCHIVE:
        load_and_standardize_archive()

    if RUN_MASTER_SALES:
        build_master_sales_file()

    if RUN_VALUATION:
        pull_property_valuation()

    if RUN_PLUTO:
        pull_pluto()


if __name__ == "__main__":
    main()

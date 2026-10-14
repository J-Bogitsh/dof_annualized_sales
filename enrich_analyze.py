"""
enrich_analyze.py
------------------
Takes the four files produced by ingestion.py and combines them into one
enriched sales file, ready for the actual analysis metrics.

Four steps, each its own function, each saving its own checkpoint file --
run them one at a time in your notebook, same pattern as ingestion.py, so
you can look at the result after each one before moving to the next.

  Step 1: build a BBL for the archive sales rows (they don't have one
          natively) and turn the two different date text formats into one
          real date column.
  Step 2: for properties that sold more than once, tag each sale with its
          order, days since the property's last sale, and price change
          since last sale.
  Step 3: join in the valuation data, matched on BBL + year (only
          possible for sales in 2023 or later, since that's all the
          valuation source covers).
  Step 4: join in PLUTO (matched on BBL alone, since it's a single
          current snapshot), then fill in gaps in the sales data
          (units, square footage, year built) using PLUTO or valuation
          wherever the sales record itself is missing that value.

A note on data types: everything coming back from the Socrata API gets
read in as text (even numbers), and the two sales sources don't always
agree on formatting (e.g. "4,522" vs 4522). This script casts things to
actual numbers where needed, tolerating anything that doesn't convert
cleanly (turning it into a blank rather than crashing).
"""

from pathlib import Path

import polars as pl

DATA_DIR = Path("./data/processed")


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def build_bbl(df: pl.DataFrame, boro_col: str, block_col: str, lot_col: str,
              out_col: str = "bbl_constructed") -> pl.DataFrame:
    """Builds the standard 10-digit BBL (borough + 5-digit block +
    4-digit lot) from separate columns -- the same formula NYC uses
    everywhere. Use this whenever a dataset has borough/block/lot as
    separate fields but no ready-made BBL."""
    return df.with_columns(
        (
            pl.col(boro_col).cast(pl.Utf8).str.strip_chars()
            + pl.col(block_col).cast(pl.Utf8).str.strip_chars().str.zfill(5)
            + pl.col(lot_col).cast(pl.Utf8).str.strip_chars().str.zfill(4)
        ).alias(out_col)
    )


def to_number(col: str) -> pl.Expr:
    """Cleans up a text column that should be a number: strips commas
    (e.g. "4,522"), then converts to a number, turning anything that
    still doesn't convert into a blank instead of erroring."""
    return (
        pl.col(col).cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
    )


# ---------------------------------------------------------------------------
# STEP 1 -- BBL FOR ARCHIVE ROWS + PARSED SALE DATES
# ---------------------------------------------------------------------------

def step1_add_bbl_and_dates() -> pl.DataFrame:
    print("Step 1: adding BBL to archive rows, parsing sale dates...")
    sales = pl.read_parquet(DATA_DIR / "sales_master.parquet")

    before = len(sales)
    sales = sales.filter(pl.col("borough").is_not_null())
    dropped = before - len(sales)
    if dropped:
        print(f"  dropped {dropped:,} completely blank rows (leftover artifacts from "
              f"the archive CSVs, not real sale records).")

    sales = build_bbl(sales, "borough", "block", "lot", out_col="bbl_constructed")
    sales = sales.with_columns(
        pl.coalesce(["bbl", "bbl_constructed"]).alias("bbl")
    ).drop("bbl_constructed")

    # a real BBL is always exactly 10 characters (1 borough + 5 block +
    # 4 lot). anything else is not usable -- blank it out rather than
    # let a malformed value silently mismatch later.
    bad_length = sales.filter(
        pl.col("bbl").is_not_null() & (pl.col("bbl").str.len_chars() != 10)
    ).height
    if bad_length:
        print(f"  found {bad_length:,} rows with a malformed BBL (not 10 characters) -- "
              f"blanking these out so they don't cause bad matches later.")
        sales = sales.with_columns(
            pl.when(pl.col("bbl").str.len_chars() == 10)
            .then(pl.col("bbl"))
            .otherwise(None)
            .alias("bbl")
        )

    # archive dates look like "1/22/2003"; annualized dates look like
    # "2019-06-21T00:00:00.000" -- try both, whichever one matches wins
    sales = sales.with_columns(
        pl.coalesce(
            pl.col("sale_date").str.to_date("%m/%d/%Y", strict=False),
            pl.col("sale_date").str.to_date("%Y-%m-%dT%H:%M:%S%.f", strict=False),
        ).alias("sale_date_parsed")
    )

    missing_bbl = sales.filter(pl.col("bbl").is_null()).height
    missing_date = sales.filter(pl.col("sale_date_parsed").is_null()).height
    print(f"  {len(sales):,} rows. Missing BBL: {missing_bbl:,}. Dates that didn't parse: {missing_date:,}.")

    out_path = DATA_DIR / "sales_with_bbl.parquet"
    sales.write_parquet(out_path)
    print(f"  -> {out_path}")
    return sales


# ---------------------------------------------------------------------------
# STEP 2 -- FLAG REPEAT SALES
# ---------------------------------------------------------------------------

def step2_flag_repeat_sales() -> pl.DataFrame:
    print("Step 2: flagging properties that sold more than once...")
    sales = pl.read_parquet(DATA_DIR / "sales_with_bbl.parquet")

    sales = sales.sort(["bbl", "sale_date_parsed"])
    sales = sales.with_columns([
        # 1st sale for this bbl = 1, 2nd = 2, etc. -- but if we don't know
        # the bbl at all, we can't know the sequence either, so leave
        # those rows blank rather than defaulting to a misleading 0
        pl.when(pl.col("bbl").is_not_null())
        .then(pl.col("bbl").cum_count().over("bbl"))
        .otherwise(None)
        .alias("sale_sequence"),
        pl.when(pl.col("bbl").is_not_null())
        .then(pl.col("sale_date_parsed").diff().over("bbl").dt.total_days())
        .otherwise(None)
        .alias("days_since_last_sale"),
        pl.when(pl.col("bbl").is_not_null())
        .then(to_number("sale_price").diff().over("bbl"))
        .otherwise(None)
        .alias("price_change_since_last_sale"),
        # DOF's own definition: a $0 sale price means ownership transferred
        # without a cash payment (gift, inheritance, family transfer, etc.)
        # -- not a real market transaction. Flag it so price-trend analysis
        # can exclude these while turnover analysis can still count them.
        (to_number("sale_price") == 0).alias("non_arms_length_sale"),
    ])

    repeat_rows = sales.filter(pl.col("sale_sequence") > 1).height
    print(f"  {repeat_rows:,} rows are a repeat sale of a property that sold before.")

    out_path = DATA_DIR / "sales_with_repeats.parquet"
    sales.write_parquet(out_path)
    print(f"  -> {out_path}")
    return sales


# ---------------------------------------------------------------------------
# STEP 3 -- JOIN VALUATION (BBL + YEAR, 2023+ ONLY)
# ---------------------------------------------------------------------------

def step3_join_valuation() -> pl.DataFrame:
    print("Step 3: joining valuation data (matched by BBL + year)...")
    sales = pl.read_parquet(DATA_DIR / "sales_with_repeats.parquet")
    valuation = pl.read_parquet(DATA_DIR / "valuation_master.parquet")

    valuation = build_bbl(valuation, "boro", "block", "lot", out_col="bbl")

    # rectype 1 = ordinary real estate -- this is the standard property
    # record. rectype 2/3 are special utility-company records that would
    # otherwise create duplicate bbl+year matches.
    valuation = valuation.filter(pl.col("rectype").cast(pl.Utf8) == "1")

    # DOF publishes one row PER STAGE of the tax year (period 0 = start,
    # 1 = tentative, 2 = change-by-notice, 3 = final roll, 4 = remission),
    # not one row per property-year -- so the same bbl+year shows up
    # multiple times, with early-stage rows holding 0 as a placeholder
    # rather than a real value. Keep whichever row reached the highest
    # (most advanced/complete) period for each property-year.
    valuation = valuation.with_columns(pl.col("period").cast(pl.Int32, strict=False))
    valuation = valuation.sort("period", descending=True)

    before_dedup = len(valuation)
    dupes = valuation.group_by(["bbl", "year"]).len().filter(pl.col("len") > 1).height
    if dupes:
        print(f"  {dupes:,} bbl+year combinations had more than one stage recorded -- "
              f"keeping only the most advanced stage (highest period) for each.")
    valuation = valuation.unique(subset=["bbl", "year"], keep="first")
    print(f"  kept {len(valuation):,} of {before_dedup:,} valuation rows after "
          f"picking one stage per property-year.")

    sales = sales.with_columns(pl.col("sale_date_parsed").dt.year().alias("sale_year"))

    valuation_slim = valuation.select([
        "bbl",
        pl.col("year").cast(pl.Int32, strict=False),
        pl.col("finmktland").alias("val_land_value"),
        pl.col("finmkttot").alias("val_market_value"),
        pl.col("fintaxclass").alias("val_tax_class"),
        pl.col("bldg_class").alias("val_bldg_class"),
        pl.col("yrbuilt").alias("val_year_built"),
        pl.col("land_area").alias("val_land_area"),
        pl.col("gross_sqft").alias("val_gross_sqft"),
        pl.col("units").alias("val_units"),
    ])

    # a $0 valuation isn't a real number to compare against a sale price --
    # whether it's an unpopulated placeholder or a property type (condo
    # sub-units, utility company property) that this field doesn't apply
    # to normally, treat it as missing rather than a real value
    valuation_slim = valuation_slim.with_columns([
        pl.when(pl.col("val_market_value") == "0").then(None)
        .otherwise(pl.col("val_market_value")).alias("val_market_value"),
        pl.when(pl.col("val_land_value") == "0").then(None)
        .otherwise(pl.col("val_land_value")).alias("val_land_value"),
    ])

    merged = sales.join(
        valuation_slim,
        left_on=["bbl", "sale_year"],
        right_on=["bbl", "year"],
        how="left",
    )
    matched = merged.filter(pl.col("val_market_value").is_not_null()).height
    print(f"  {matched:,} of {len(merged):,} sales matched to a valuation record "
          f"(only possible for sales from 2023 onward).")

    out_path = DATA_DIR / "sales_with_valuation.parquet"
    merged.write_parquet(out_path)
    print(f"  -> {out_path}")
    return merged


# ---------------------------------------------------------------------------
# STEP 4 -- JOIN PLUTO, FILL GAPS
# ---------------------------------------------------------------------------

def step4_join_pluto_and_fill_gaps() -> pl.DataFrame:
    print("Step 4: joining PLUTO, filling gaps in the sales data...")
    sales = pl.read_parquet(DATA_DIR / "sales_with_valuation.parquet")
    pluto = pl.read_parquet(DATA_DIR / "pluto_raw.parquet")

    pluto_slim = pluto.select([
        # PLUTO's bbl comes back as a decimal-looking string like
        # "4087860042.00000000" -- convert through a number to get the
        # clean "4087860042" form that matches the sales data's bbl
        pl.col("bbl").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).cast(pl.Utf8).alias("bbl"),
        pl.col("latitude").alias("pluto_latitude"),
        pl.col("longitude").alias("pluto_longitude"),
        pl.col("cd").alias("pluto_community_district"),
        pl.col("council").alias("pluto_council_district"),
        pl.col("bct2020").alias("pluto_census_tract_2020"),
        pl.col("bldgclass").alias("pluto_bldg_class"),
        pl.col("yearbuilt").alias("pluto_year_built"),
        pl.col("lotarea").alias("pluto_land_area"),
        pl.col("bldgarea").alias("pluto_gross_sqft"),
        pl.col("unitsres").alias("pluto_residential_units"),
        pl.col("unitstotal").alias("pluto_total_units"),
    ]).unique(subset=["bbl"], keep="first")

    merged = sales.join(pluto_slim, on="bbl", how="left")

    # fill gaps: keep the sales record's own value if it has one,
    # otherwise use PLUTO's, otherwise valuation's (2023+ only)
    merged = merged.with_columns([
        pl.coalesce([
            to_number("residential_units"),
            to_number("pluto_residential_units"),
        ]).alias("residential_units_filled"),
        pl.coalesce([
            to_number("total_units"),
            to_number("pluto_total_units"),
            to_number("val_units"),
        ]).alias("total_units_filled"),
        pl.coalesce([
            to_number("land_square_feet"),
            to_number("pluto_land_area"),
            to_number("val_land_area"),
        ]).alias("land_square_feet_filled"),
        pl.coalesce([
            to_number("gross_square_feet"),
            to_number("pluto_gross_sqft"),
            to_number("val_gross_sqft"),
        ]).alias("gross_square_feet_filled"),
        pl.coalesce([
            to_number("year_built"),
            to_number("pluto_year_built"),
            to_number("val_year_built"),
        ]).alias("year_built_filled"),
    ])

    filled = merged.filter(
        to_number("total_units").is_null() & pl.col("total_units_filled").is_not_null()
    ).height
    print(f"  filled in total_units for {filled:,} rows that were previously missing it.")

    out_path = DATA_DIR / "sales_enriched.parquet"
    merged.write_parquet(out_path)
    print(f"  {len(merged):,} rows, {len(merged.columns)} columns -> {out_path}")
    return merged


def main():
    step1_add_bbl_and_dates()
    step2_flag_repeat_sales()
    step3_join_valuation()
    step4_join_pluto_and_fill_gaps()


if __name__ == "__main__":
    main()

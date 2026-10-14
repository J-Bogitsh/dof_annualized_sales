"""
analyze_metrics.py
-------------------
Builds the Tier 1 analysis tables from sales_enriched.parquet. Each
function computes one group of metrics and saves a small output file --
small enough that the eventual dashboard can read these directly instead
of touching the full ~2.1 million row sales table on every interaction.

Run each function one at a time in your notebook, same pattern as the
last two scripts.

  price_trends()       -- median price and spread over time, sliced a
                           few different ways
  turnover_metrics()    -- how often properties change hands
  sales_activity()      -- sales volume by building type, and how long
                           people hold a property before reselling
  class_transitions()   -- properties whose building class or tax class
                           has changed since they sold
  assessment_equity()   -- sale price vs. assessed value: the "tax
                           equity" ratio study (2023+ only)

A note on scope: every metric here excludes non-arm's-length sales
($0 or otherwise not a real market transaction) -- these would distort
a median badly if left in. Anywhere valuation data is used, it's
limited to 2023 onward, since that's all the source covers.
"""

from pathlib import Path

import polars as pl

DATA_DIR = Path("./data/processed")
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def to_number(col: str) -> pl.Expr:
    """Cleans up a text column that should be a number."""
    return (
        pl.col(col).cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
    )


def load_sales() -> pl.DataFrame:
    """Loads the enriched sales file and adds a few columns every metric
    below needs: a real numeric sale price, the sale month, and a
    "current" building/tax class that falls back to PLUTO or valuation
    data when the sales record's own current-class field is blank
    (which happens for every archive-era row, since that field only
    ever existed in the annualized/current dataset)."""
    sales = pl.read_parquet(DATA_DIR / "sales_enriched.parquet")
    sales = sales.with_columns([
        to_number("sale_price").alias("sale_price_num"),
        pl.col("sale_date_parsed").dt.month().alias("sale_month"),
        pl.coalesce([
            pl.col("building_class_as_of_final"),
            pl.col("pluto_bldg_class"),
            pl.col("val_bldg_class"),
        ]).alias("current_building_class"),
        pl.coalesce([
            pl.col("tax_class_as_of_final_roll"),
            pl.col("val_tax_class"),
        ]).alias("current_tax_class"),
    ])
    # a known DOF quirk: a bulk/portfolio transaction (e.g. a sponsor
    # selling several co-op units in one deal) sometimes gets logged as
    # one row PER UNIT, with the FULL deal price repeated on every row,
    # not split per unit. Flag any bbl+date+price combination repeated
    # 3+ times as a likely bulk sale -- one real transaction, not several
    # units each genuinely worth that much.
    sales = sales.with_columns(
        (pl.len().over(["bbl", "sale_date_parsed", "sale_price_num"]) >= 3)
        .alias("likely_bulk_transaction")
    )
    return sales


MIN_ARMS_LENGTH_PRICE = 1000  # sales under this are treated as non-arm's-length,
                               # same as a $0 sale -- $1/$10 "sales" turned out to
                               # be real data (parking spots, nominal transfers)
                               # that would otherwise wreck any ratio or average
                               # they're part of


def usable_sales(sales: pl.DataFrame) -> pl.DataFrame:
    """The filter every price-based metric should start from: a real
    BBL, a real (arm's-length) sale price, and a parsed date."""
    return sales.filter(
        pl.col("bbl").is_not_null()
        & ~pl.col("non_arms_length_sale")
        & ~pl.col("likely_bulk_transaction")
        & pl.col("sale_price_num").is_not_null()
        & (pl.col("sale_price_num") >= MIN_ARMS_LENGTH_PRICE)
        & pl.col("sale_date_parsed").is_not_null()
    )


# ---------------------------------------------------------------------------
# METRIC 1 -- PRICE TRENDS
# ---------------------------------------------------------------------------

def price_trends() -> None:
    print("Building price trend tables...")
    sales = usable_sales(load_sales())
    print(f"  {len(sales):,} usable (arm's-length) sales going into these tables.")

    def summarize(group_cols: list[str]) -> pl.DataFrame:
        return (
            sales.group_by(group_cols)
            .agg([
                pl.col("sale_price_num").median().alias("median_price"),
                pl.col("sale_price_num").quantile(0.25).alias("p25_price"),
                pl.col("sale_price_num").quantile(0.75).alias("p75_price"),
                pl.len().alias("sale_count"),
            ])
            .sort(group_cols)
        )

    citywide_monthly = summarize(["borough", "sale_year", "sale_month"])
    by_neighborhood = summarize(["borough", "neighborhood", "sale_year"])
    by_building_class = summarize(["building_class_category", "sale_year"])
    by_tax_class = summarize(["tax_class_at_time_of_sale", "sale_year"])

    citywide_monthly.write_parquet(METRICS_DIR / "price_trend_citywide_monthly.parquet")
    by_neighborhood.write_parquet(METRICS_DIR / "price_trend_by_neighborhood_annual.parquet")
    by_building_class.write_parquet(METRICS_DIR / "price_trend_by_building_class_annual.parquet")
    by_tax_class.write_parquet(METRICS_DIR / "price_trend_by_tax_class_annual.parquet")

    print(f"  -> {METRICS_DIR}/price_trend_citywide_monthly.parquet ({len(citywide_monthly):,} rows)")
    print(f"  -> {METRICS_DIR}/price_trend_by_neighborhood_annual.parquet ({len(by_neighborhood):,} rows)")
    print(f"  -> {METRICS_DIR}/price_trend_by_building_class_annual.parquet ({len(by_building_class):,} rows)")
    print(f"  -> {METRICS_DIR}/price_trend_by_tax_class_annual.parquet ({len(by_tax_class):,} rows)")


# ---------------------------------------------------------------------------
# METRIC 2 -- TURNOVER
# ---------------------------------------------------------------------------

def turnover_metrics() -> None:
    print("Building turnover rate tables...")
    sales = usable_sales(load_sales())
    pluto = pl.read_parquet(DATA_DIR / "pluto_raw.parquet")

    # PLUTO stores borough as letter codes (MN/BX/BK/QN/SI); the sales
    # data uses the numeric codes (1-5). Translate PLUTO's to match,
    # or the join below silently matches nothing.
    borough_code_map = {"MN": "1", "BX": "2", "BK": "3", "QN": "4", "SI": "5"}
    pluto = pluto.with_columns(
        pl.col("borough").replace(borough_code_map).alias("borough")
    )

    # A co-op building files every individual apartment sale under ONE
    # shared BBL (a co-op apartment is legally a share of stock, not its
    # own tax lot -- unlike a condo). Counting "distinct properties sold"
    # flattened 40 real co-op apartment sales in one building down to
    # "1 property turned over," badly understating turnover anywhere
    # co-ops are common (especially Manhattan). Fix: count every real
    # transaction (the numerator), and use total HOUSING UNITS as the
    # denominator (not total lots), so a 200-unit co-op building
    # contributes 200 to the denominator, matching how its sales can
    # contribute up to 200 to the numerator. Non-residential properties
    # (commercial, vacant land) still count as 1 slot each rather than
    # being dropped from the denominator entirely.
    pluto = pluto.with_columns(
        pl.max_horizontal([to_number("unitstotal"), pl.lit(1.0)]).alias("units_or_one")
    )
    inventory_by_borough = (
        pluto.group_by("borough").agg(pl.col("units_or_one").sum().alias("total_units"))
    )
    transactions_by_borough_year = (
        sales.group_by(["borough", "sale_year"])
        .agg(pl.len().alias("transactions"))
    )
    turnover_borough = transactions_by_borough_year.join(inventory_by_borough, on="borough", how="left")
    turnover_borough = turnover_borough.with_columns(
        (pl.col("transactions") / pl.col("total_units") * 100).alias("turnover_rate_pct")
    ).sort(["borough", "sale_year"])
    turnover_borough.write_parquet(METRICS_DIR / "turnover_borough_annual.parquet")
    print(f"  -> {METRICS_DIR}/turnover_borough_annual.parquet ({len(turnover_borough):,} rows)")

    # NOTE on the neighborhood version below: PLUTO doesn't have a
    # "neighborhood" field (only borough/community district), so there's
    # no clean way to get a true unit-count denominator at the
    # neighborhood level from our current data sources. This is an
    # APPROXIMATION using "how many distinct properties in this
    # neighborhood have EVER sold across the full 2003-2027 history" as
    # a stand-in denominator, against total transactions as the
    # numerator -- which means this can legitimately exceed 100% in
    # co-op-heavy neighborhoods (many sales against relatively few
    # distinct BBLs). Treat this as a churn-intensity indicator, not a
    # literal turnover percentage -- the borough-level number above is
    # the trustworthy, properly-scaled one.
    neighborhood_universe = (
        sales.group_by("neighborhood").agg(pl.col("bbl").n_unique().alias("approx_property_universe"))
    )
    transactions_by_neighborhood_year = (
        sales.group_by(["neighborhood", "sale_year"])
        .agg(pl.len().alias("transactions"))
    )
    turnover_neighborhood = transactions_by_neighborhood_year.join(
        neighborhood_universe, on="neighborhood", how="left"
    )
    turnover_neighborhood = turnover_neighborhood.with_columns(
        (pl.col("transactions") / pl.col("approx_property_universe") * 100).alias("approx_turnover_rate_pct")
    ).sort(["neighborhood", "sale_year"])
    turnover_neighborhood.write_parquet(METRICS_DIR / "turnover_neighborhood_approx_annual.parquet")
    print(f"  -> {METRICS_DIR}/turnover_neighborhood_approx_annual.parquet ({len(turnover_neighborhood):,} rows) "
          f"-- APPROXIMATE, see comment in the code")


# ---------------------------------------------------------------------------
# METRIC 3 -- SALES VOLUME BY BUILDING TYPE, HOLDING PERIODS
# ---------------------------------------------------------------------------

def sales_activity() -> None:
    print("Building sales volume and holding period tables...")
    sales = usable_sales(load_sales())

    volume = (
        sales.group_by(["building_class_category", "sale_year"])
        .agg(pl.len().alias("sale_count"))
        .sort(["building_class_category", "sale_year"])
    )
    volume.write_parquet(METRICS_DIR / "sales_volume_by_building_class.parquet")
    print(f"  -> {METRICS_DIR}/sales_volume_by_building_class.parquet ({len(volume):,} rows)")

    # holding periods only make sense for a property's 2nd+ sale (there's
    # nothing to measure "since" for a first-ever recorded sale)
    repeat_sales = sales.filter((pl.col("sale_sequence") > 1) & pl.col("days_since_last_sale").is_not_null())
    holding = (
        repeat_sales.group_by("building_class_category")
        .agg([
            pl.col("days_since_last_sale").median().alias("median_days_held"),
            (pl.col("days_since_last_sale").median() / 365.25).alias("median_years_held"),
            pl.len().alias("repeat_sale_count"),
        ])
        .sort("median_years_held", descending=True)
    )
    holding.write_parquet(METRICS_DIR / "holding_periods_by_building_class.parquet")
    print(f"  -> {METRICS_DIR}/holding_periods_by_building_class.parquet ({len(holding):,} rows)")


# ---------------------------------------------------------------------------
# METRIC 4 -- BUILDING CLASS / TAX CLASS TRANSITIONS
# ---------------------------------------------------------------------------

def class_transitions() -> None:
    print("Building class transition tables...")
    sales = usable_sales(load_sales())

    bldg = sales.filter(
        pl.col("building_class_at_time_of_sale").is_not_null()
        & pl.col("current_building_class").is_not_null()
        & (pl.col("building_class_at_time_of_sale").str.strip_chars() != "")
        & (pl.col("current_building_class").str.strip_chars() != "")
        & (pl.col("building_class_at_time_of_sale") != pl.col("current_building_class"))
    )
    bldg_transitions = (
        bldg.group_by(["building_class_at_time_of_sale", "current_building_class"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    bldg_transitions.write_parquet(METRICS_DIR / "building_class_transitions.parquet")
    print(f"  -> {METRICS_DIR}/building_class_transitions.parquet "
          f"({len(bldg_transitions):,} distinct transition pairs, "
          f"{bldg.height:,} properties involved)")

    tax = sales.filter(
        pl.col("tax_class_at_time_of_sale").is_not_null()
        & pl.col("current_tax_class").is_not_null()
        & (pl.col("tax_class_at_time_of_sale").str.strip_chars() != "")
        & (pl.col("current_tax_class").str.strip_chars() != "")
        & (pl.col("tax_class_at_time_of_sale") != pl.col("current_tax_class"))
    )
    tax_transitions = (
        tax.group_by(["tax_class_at_time_of_sale", "current_tax_class"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    tax_transitions.write_parquet(METRICS_DIR / "tax_class_transitions.parquet")
    print(f"  -> {METRICS_DIR}/tax_class_transitions.parquet "
          f"({len(tax_transitions):,} distinct transition pairs, "
          f"{tax.height:,} properties involved)")


# ---------------------------------------------------------------------------
# METRIC 5 -- ASSESSMENT EQUITY (SALE PRICE VS. VALUATION)
# ---------------------------------------------------------------------------

def assessment_equity() -> None:
    """The tax-equity ratio study: assessed market value / sale price,
    the same way official assessment ratio studies (and Tax Equity Now's
    analysis) measure it. Computed within tax class, since Class 1, 2,
    and 4 are assessed under different methods and shouldn't be pooled
    together. 2023+ only."""
    print("Building assessment equity (ratio study) tables...")
    sales = usable_sales(load_sales())

    ratio_data = sales.filter(pl.col("val_market_value").is_not_null()).with_columns(
        (to_number("val_market_value") / pl.col("sale_price_num")).alias("ratio")
    )
    print(f"  {len(ratio_data):,} sales have a usable valuation match to compute a ratio from.")

    # standard ratio-study practice: trim ratios outside a reasonable
    # range before computing COD/PRD. A handful of sales with a genuinely
    # bizarre ratio (distressed sale, partial-interest transfer, a data
    # entry error) can otherwise dominate these statistics even after
    # the price-floor filter above.
    before_trim = len(ratio_data)
    ratio_data = ratio_data.filter((pl.col("ratio") >= 0.1) & (pl.col("ratio") <= 5.0))
    trimmed = before_trim - len(ratio_data)
    print(f"  trimmed {trimmed:,} sales with a ratio outside the 0.1-5.0 range "
          f"(not representative for a ratio study).")

    equity = (
        ratio_data.group_by(["val_tax_class", "borough", "sale_year"])
        .agg([
            pl.col("ratio").median().alias("median_ratio"),
            pl.col("ratio").mean().alias("mean_ratio"),
            # Coefficient of Dispersion: average distance from the group's
            # median ratio, as a percent of that median -- measures how
            # consistently similar properties are being assessed
            ((pl.col("ratio") - pl.col("ratio").median()).abs().mean()
             / pl.col("ratio").median() * 100).alias("cod"),
            # Price-Related Differential: simple mean ratio divided by the
            # value-weighted mean ratio. Above ~1.03 conventionally
            # indicates regressivity -- lower-value properties assessed
            # relatively higher than higher-value ones
            (pl.col("ratio").mean()
             / (to_number("val_market_value").sum() / pl.col("sale_price_num").sum())).alias("prd"),
            pl.len().alias("n_sales"),
        ])
        .sort(["val_tax_class", "borough", "sale_year"])
    )
    equity.write_parquet(METRICS_DIR / "assessment_equity.parquet")
    print(f"  -> {METRICS_DIR}/assessment_equity.parquet ({len(equity):,} rows)")


def main():
    price_trends()
    turnover_metrics()
    sales_activity()
    class_transitions()
    assessment_equity()


if __name__ == "__main__":
    main()

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

from src.config import (
    NUMERIC_COLS, CATEGORICAL_COLS, LABEL_CLF,
    DEW_POINT_DEPRESSION_THRESHOLD,
)


def create_rain_label(df):
    """
    Tao label Rain (0/1) tu Dew Point Depression.
    Khi Temperature - DewPoint < threshold => xac suat mua cao.
    """
    threshold = DEW_POINT_DEPRESSION_THRESHOLD
    df = df.withColumn(
        "DewPointDepression",
        F.col("Temperature") - F.col("DewPoint"),
    )
    df = df.withColumn(
        "Rain",
        F.when(
            F.col("DewPointDepression") < threshold,
            1.0,
        ).otherwise(0.0),
    )
    rain_count = df.filter(F.col("Rain") == 1.0).count()
    total = df.count()
    pct = rain_count / total * 100 if total > 0 else 0
    print(f"  Rain label (DewPoint Depression < {threshold}): "
          f"{rain_count:,}/{total:,} ({pct:.1f}%) co mua")
    return df


def create_targets(df):
    """Tao target cho classification (Rain_next) va regression (Temp_next)."""
    w = Window.partitionBy("StationID").orderBy("datetime")
    df = df.withColumn("Rain_next", F.lead("Rain", 1).over(w))
    df = df.withColumn("Temp_next", F.lead("Temperature", 1).over(w))
    df = df.withColumn(LABEL_CLF, F.col("Rain_next"))
    print("  Targets: Rain_next (clf), Temp_next (reg)")
    return df


def parse_datetime(df):
    df = df.withColumn("datetime_ts", F.to_timestamp("datetime"))
    df = df.filter(F.col("datetime_ts").isNotNull())
    print("  Parsed datetime")
    return df


def drop_target_nulls(df):
    before = df.count()
    df = df.dropna(subset=["Rain_next", "Temp_next"])
    after = df.count()
    print(f"  Drop null targets: {before:,} -> {after:,} (giam {before - after:,})")
    return df


def fill_numeric_missing(df, cols=None):
    if cols is None:
        cols = NUMERIC_COLS
    for col_name in cols:
        median_val = df.approxQuantile(col_name, [0.5], 0.05)
        if median_val:
            df = df.fillna({col_name: median_val[0]})
    print(f"  Filled {len(cols)} numeric cols with median")
    return df


def get_string_indexers(cols=None):
    if cols is None:
        cols = CATEGORICAL_COLS
    return [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cols
    ]


def preprocess(df):
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    df = parse_datetime(df)
    df = create_rain_label(df)
    df = create_targets(df)
    df = drop_target_nulls(df)
    df = fill_numeric_missing(df)

    print(f"  Result: {df.count():,} dong, {len(df.columns)} cot")
    print("Preprocessing done.\n")
    return df

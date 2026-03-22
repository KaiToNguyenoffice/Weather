from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import OneHotEncoder

from src.config import CATEGORICAL_COLS


def add_time_features(df):
    df = df.withColumn("Hour", F.hour("datetime_ts"))
    df = df.withColumn("Month", F.month("datetime_ts"))
    df = df.withColumn("DayOfWeek", F.dayofweek("datetime_ts"))
    df = df.withColumn("DayOfYear", F.dayofyear("datetime_ts"))

    df = df.withColumn("Hour_sin", F.sin(2 * 3.14159 * F.col("Hour") / 24))
    df = df.withColumn("Hour_cos", F.cos(2 * 3.14159 * F.col("Hour") / 24))
    df = df.withColumn("Month_sin", F.sin(2 * 3.14159 * F.col("Month") / 12))
    df = df.withColumn("Month_cos", F.cos(2 * 3.14159 * F.col("Month") / 12))

    return df


def add_lag_features(df):
    w = Window.partitionBy("StationID").orderBy("datetime_ts")

    df = df.withColumn("Temp_lag1", F.lag("Temperature", 1).over(w))
    df = df.withColumn("Temp_lag3", F.lag("Temperature", 3).over(w))
    df = df.withColumn("Temp_lag6", F.lag("Temperature", 6).over(w))
    df = df.withColumn("Temp_lag24", F.lag("Temperature", 24).over(w))

    df = df.withColumn("DewPoint_lag1", F.lag("DewPoint", 1).over(w))
    df = df.withColumn("Pressure_lag1", F.lag("Pressure", 1).over(w))
    df = df.withColumn("WindSpeed_lag1", F.lag("WindSpeed", 1).over(w))
    df = df.withColumn("Rain_lag1", F.lag("Rain", 1).over(w))
    df = df.withColumn("Rain_lag3", F.lag("Rain", 3).over(w))

    return df


def add_rolling_features(df):
    w6 = Window.partitionBy("StationID").orderBy("datetime_ts").rowsBetween(-6, -1)
    w24 = Window.partitionBy("StationID").orderBy("datetime_ts").rowsBetween(-24, -1)

    df = df.withColumn("Temp_roll6h", F.avg("Temperature").over(w6))
    df = df.withColumn("Temp_roll24h", F.avg("Temperature").over(w24))
    df = df.withColumn("DewPoint_roll6h", F.avg("DewPoint").over(w6))
    df = df.withColumn("Pressure_roll6h", F.avg("Pressure").over(w6))

    df = df.withColumn("Rain_sum6h", F.sum("Rain").over(w6))
    df = df.withColumn("Rain_sum24h", F.sum("Rain").over(w24))

    return df


def add_derived_features(df):
    df = df.withColumn("TempChange_1h",
                       F.col("Temperature") - F.col("Temp_lag1"))
    df = df.withColumn("DewPointChange_1h",
                       F.col("DewPoint") - F.col("DewPoint_lag1"))
    df = df.withColumn("PressureChange_1h",
                       F.col("Pressure") - F.col("Pressure_lag1"))
    df = df.withColumn("TempDewPoint",
                       F.col("Temperature") - F.col("DewPoint"))
    return df


def get_ohe_encoders(cols=None):
    if cols is None:
        cols = CATEGORICAL_COLS
    return [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec", handleInvalid="keep")
        for c in cols
    ]


def engineer_features(df):
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = add_time_features(df)
    print("  + Time: Hour, Month, DayOfWeek, cyclic sin/cos")

    df = add_lag_features(df)
    print("  + Lag: Temp 1/3/6/24h, DewPoint/Pressure/Wind 1h, Rain 1/3h")

    df = add_rolling_features(df)
    print("  + Rolling: Temp/DewPoint/Pressure 6h/24h, Rain_sum 6h/24h")

    df = add_derived_features(df)
    print("  + Derived: TempChange, DewPointChange, PressureChange, TempDewPoint")

    before = df.count()
    df = df.dropna()
    after = df.count()
    print(f"  Drop NaN from lag/rolling: {before:,} -> {after:,}")
    print(f"  Total columns: {len(df.columns)}")
    print("Feature engineering done.\n")
    return df

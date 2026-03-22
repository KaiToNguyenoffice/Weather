import os
import json
import random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
)

from src.config import (
    STATIONS_DIR, META_INFO_FILE,
    NUM_STATIONS_SAMPLE, MIN_VALID_PERCENT, RANDOM_SEED,
    SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)


def create_spark_session(app_name=None, master=None, driver_memory=None):
    if app_name is None:
        app_name = SPARK_APP_NAME
    if master is None:
        master = SPARK_MASTER
    if driver_memory is None:
        driver_memory = SPARK_DRIVER_MEMORY

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.network.timeout", "600s")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _load_meta_info(meta_path=None):
    """Load meta_info.json va loc tram co du lieu hop le."""
    if meta_path is None:
        meta_path = META_INFO_FILE
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def _select_stations(meta, num_stations=None, min_valid=None, seed=None):
    """Loc va random sample tram tu metadata."""
    if num_stations is None:
        num_stations = NUM_STATIONS_SAMPLE
    if min_valid is None:
        min_valid = MIN_VALID_PERCENT
    if seed is None:
        seed = RANDOM_SEED

    # Loc tram co valid_percent >= min_valid
    valid_stations = [
        fname for fname, info in meta.items()
        if info.get("valid_percent", 0) >= min_valid
    ]
    print(f"  Tram co valid_percent >= {min_valid}: {len(valid_stations)}/{len(meta)}")

    # Sample
    if num_stations < len(valid_stations):
        random.seed(seed)
        selected = random.sample(valid_stations, num_stations)
    else:
        selected = valid_stations

    print(f"  Selected: {len(selected)} tram")
    return selected


def load_weather5k(spark, stations_dir=None, num_stations=None):
    """
    Load WEATHER-5K dataset:
    1. Doc meta_info.json, loc tram hop le
    2. Sample num_stations tram
    3. Load moi tram CSV, them StationID
    4. Union tat ca
    5. Rename columns, loc MASK, drop columns khong can
    """
    if stations_dir is None:
        stations_dir = STATIONS_DIR

    print("=" * 60)
    print("LOADING WEATHER-5K")
    print("=" * 60)

    # 1. Meta info
    meta = _load_meta_info()
    selected_files = _select_stations(meta, num_stations)

    # 2. Load tung file CSV va union
    schema = StructType([
        StructField("DATE", StringType(), True),
        StructField("LONGITUDE", DoubleType(), True),
        StructField("LATITUDE", DoubleType(), True),
        StructField("TMP", DoubleType(), True),
        StructField("DEW", DoubleType(), True),
        StructField("WND_ANGLE", DoubleType(), True),
        StructField("WND_RATE", DoubleType(), True),
        StructField("SLP", DoubleType(), True),
        StructField("MASK", StringType(), True),
        StructField("TIME_DIFF", DoubleType(), True),
    ])

    dfs = []
    loaded = 0
    for fname in selected_files:
        fpath = os.path.join(stations_dir, fname)
        if not os.path.exists(fpath):
            continue
        station_id = fname.replace(".csv", "")
        df = (
            spark.read.csv(fpath, header=True, schema=schema)
            .withColumn("StationID", F.lit(station_id))
        )
        dfs.append(df)
        loaded += 1

    print(f"  Loaded: {loaded} files")

    if not dfs:
        raise ValueError("Khong load duoc file nao tu WEATHER-5K!")

    # Union all
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.unionByName(df)

    # 3. Loc du lieu hop le: MASK = [1 1 1 1 1]
    before = combined.count()
    combined = combined.filter(F.col("MASK") == "[1 1 1 1 1]")
    after = combined.count()
    print(f"  MASK filter: {before:,} -> {after:,} (loai {before - after:,} invalid)")

    # 4. Rename columns
    combined = (
        combined
        .withColumnRenamed("DATE", "datetime")
        .withColumnRenamed("TMP", "Temperature")
        .withColumnRenamed("DEW", "DewPoint")
        .withColumnRenamed("WND_ANGLE", "WindDirection")
        .withColumnRenamed("WND_RATE", "WindSpeed")
        .withColumnRenamed("SLP", "Pressure")
    )

    # 5. Drop columns khong can
    combined = combined.drop("MASK", "TIME_DIFF", "LONGITUDE", "LATITUDE")

    total = combined.count()
    print(f"  Result: {total:,} dong x {len(combined.columns)} cot")
    print(f"  Columns: {combined.columns}")
    print("Loading done.\n")
    return combined


def print_summary(df):
    print(f"\nSo dong: {df.count():,}")
    print(f"So cot : {len(df.columns)}")
    print("\n--- Schema ---")
    df.printSchema()
    print("\n--- 5 dong dau ---")
    df.show(5, truncate=False)

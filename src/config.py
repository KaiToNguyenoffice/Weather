import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── WEATHER-5K dataset ───────────────────────────────────────
WEATHER5K_DIR = os.path.join(DATA_DIR, "WEATHER-5K")
STATIONS_DIR = os.path.join(WEATHER5K_DIR, "global_weather_stations")
META_INFO_FILE = os.path.join(WEATHER5K_DIR, "meta_info.json")
PERCENTILE_FILE = os.path.join(WEATHER5K_DIR, "percentile.json")

# So tram lay mau (giam de chay local, tang khi chay tren cluster)
NUM_STATIONS_SAMPLE = 200
MIN_VALID_PERCENT = 0.8  

# ── Checkpoints ──────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_weather5k")
MODELS_DIR = os.path.join(CHECKPOINT_DIR, "models")
DATA_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "processed_data.parquet")
TRAIN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "train.parquet")
TEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "test.parquet")
METRICS_FILE = os.path.join(CHECKPOINT_DIR, "metrics.json")

# ── Spark ────────────────────────────────────────────────────
SPARK_APP_NAME = "WeatherForecast_W5K"
# local[4] caps parallel tasks vs local[*] and lowers peak heap on big MLlib tree jobs (Windows).
SPARK_MASTER = "local[4]"
SPARK_DRIVER_MEMORY = "6g"
SPARK_SHUFFLE_PARTITIONS = "8"

# MLlib Random Forest is heap-heavy on large row counts; raise trees/depth only if RAM allows.
RF_NUM_TREES = 40
RF_MAX_DEPTH = 8
RF_MAX_BINS = 24
RF_SUBSAMPLING_RATE = 0.5

# ── Train / Eval ─────────────────────────────────────────────
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ── Columns ──────────────────────────────────────────────────
# Sau khi rename trong data_loader
NUMERIC_COLS = [
    "Temperature", "DewPoint", "Pressure", "WindSpeed", "WindDirection",
]

CATEGORICAL_COLS = ["StationID"]

# Nguong Dew Point Depression de phan loai mua
DEW_POINT_DEPRESSION_THRESHOLD = 2.5

TARGET_CLF = "Rain_next"
TARGET_REG = "Temp_next"

LABEL_CLF = "label_clf"
LABEL_REG = "label_reg"

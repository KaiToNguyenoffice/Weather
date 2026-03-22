import os
import shutil

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
)

from src.config import (
    NUMERIC_COLS, CATEGORICAL_COLS, LABEL_CLF, LABEL_REG,
    TARGET_REG, RANDOM_SEED, TEST_RATIO, MODELS_DIR,
    DATA_CHECKPOINT, TRAIN_CHECKPOINT, TEST_CHECKPOINT,
    RF_NUM_TREES, RF_MAX_DEPTH, RF_MAX_BINS, RF_SUBSAMPLING_RATE,
)
from src.preprocessing import get_string_indexers
from src.feature_engineering import get_ohe_encoders


# ── Checkpoint helpers ───────────────────────────────────────

def _model_path(task, model_name):
    return os.path.join(MODELS_DIR, f"{task}_{model_name}")


def save_model(model, task, model_name):
    path = _model_path(task, model_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    model.save(path)
    print(f"  [CHECKPOINT] Model saved -> {path}")


def load_model(task, model_name):
    path = _model_path(task, model_name)
    if os.path.exists(path):
        model = PipelineModel.load(path)
        print(f"  [CHECKPOINT] Model loaded <- {path}")
        return model
    return None


def save_dataframe(df, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    df.write.parquet(path)
    print(f"  [CHECKPOINT] Data saved -> {path}")


def load_dataframe(spark, path):
    if os.path.exists(path) and os.path.isdir(path):
        contents = os.listdir(path)
        if any(f.endswith('.parquet') or f == '_SUCCESS' for f in contents):
            df = spark.read.parquet(path)
            print(f"  [CHECKPOINT] Data loaded <- {path} ({df.count():,} dong)")
            return df
    return None


def save_train_test(train_df, test_df):
    save_dataframe(train_df, TRAIN_CHECKPOINT)
    save_dataframe(test_df, TEST_CHECKPOINT)


def load_train_test(spark):
    train_df = load_dataframe(spark, TRAIN_CHECKPOINT)
    test_df = load_dataframe(spark, TEST_CHECKPOINT)
    if train_df is not None and test_df is not None:
        return train_df, test_df
    return None, None


def save_processed_data(df):
    save_dataframe(df, DATA_CHECKPOINT)


def load_processed_data(spark):
    return load_dataframe(spark, DATA_CHECKPOINT)


# ── Feature / pipeline helpers ───────────────────────────────

def get_feature_cols():
    numeric = NUMERIC_COLS + [
        "DewPointDepression", "Rain",
        "Hour", "Month", "DayOfWeek", "DayOfYear",
        "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
        "Temp_lag1", "Temp_lag3", "Temp_lag6", "Temp_lag24",
        "DewPoint_lag1", "Pressure_lag1", "WindSpeed_lag1",
        "Rain_lag1", "Rain_lag3",
        "Temp_roll6h", "Temp_roll24h",
        "DewPoint_roll6h", "Pressure_roll6h",
        "Rain_sum6h", "Rain_sum24h",
        "TempChange_1h", "DewPointChange_1h", "PressureChange_1h",
        "TempDewPoint",
    ]
    categorical_vec = [f"{c}_vec" for c in CATEGORICAL_COLS]
    return numeric + categorical_vec


def build_assembler_scaler(feature_cols=None):
    if feature_cols is None:
        feature_cols = get_feature_cols()
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False,
    )
    return assembler, scaler


def split_data(df, test_ratio=None, seed=None):
    if test_ratio is None:
        test_ratio = TEST_RATIO
    if seed is None:
        seed = RANDOM_SEED
    train_df, test_df = df.randomSplit(
        [1 - test_ratio, test_ratio], seed=seed
    )
    print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")
    return train_df, test_df


def _base_stages():
    indexers = get_string_indexers()
    encoders = get_ohe_encoders()
    assembler, scaler = build_assembler_scaler()
    return indexers + encoders + [assembler, scaler]


# ── Classification ───────────────────────────────────────────

def build_clf_pipeline(model_name="lr"):
    stages = _base_stages()

    if model_name == "lr":
        clf = LogisticRegression(
            featuresCol="features", labelCol=LABEL_CLF,
            maxIter=100, regParam=0.01,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            featuresCol="features", labelCol=LABEL_CLF,
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            maxBins=RF_MAX_BINS,
            subsamplingRate=RF_SUBSAMPLING_RATE,
            seed=RANDOM_SEED,
        )
    elif model_name == "gbt":
        clf = GBTClassifier(
            featuresCol="features", labelCol=LABEL_CLF,
            maxIter=50, maxDepth=5, seed=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    stages.append(clf)
    return Pipeline(stages=stages)


def train_classification(train_df, model_name="lr"):
    cached = load_model("clf", model_name)
    if cached is not None:
        return cached

    print(f"\nTraining classification: {model_name.upper()}")
    pipeline = build_clf_pipeline(model_name)
    model = pipeline.fit(train_df)
    print(f"  Training complete.")
    save_model(model, "clf", model_name)
    return model


# ── Regression ───────────────────────────────────────────────

def build_reg_pipeline(model_name="linear"):
    stages = _base_stages()

    if model_name == "linear":
        reg = LinearRegression(
            featuresCol="features", labelCol=LABEL_REG,
            maxIter=100, regParam=0.01,
        )
    elif model_name == "rf":
        reg = RandomForestRegressor(
            featuresCol="features", labelCol=LABEL_REG,
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            maxBins=RF_MAX_BINS,
            subsamplingRate=RF_SUBSAMPLING_RATE,
            seed=RANDOM_SEED,
        )
    elif model_name == "gbt":
        reg = GBTRegressor(
            featuresCol="features", labelCol=LABEL_REG,
            maxIter=50, maxDepth=5, seed=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    stages.append(reg)
    return Pipeline(stages=stages)


def train_regression(train_df, model_name="linear"):
    cached = load_model("reg", model_name)
    if cached is not None:
        return cached

    print(f"\nTraining regression: {model_name.upper()}")
    pipeline = build_reg_pipeline(model_name)
    model = pipeline.fit(train_df)
    print(f"  Training complete.")
    save_model(model, "reg", model_name)
    return model

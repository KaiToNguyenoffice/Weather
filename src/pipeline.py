"""
Pipeline: WEATHER-5K CSV files -> Load -> Preprocessing -> Feature Engineering -> Model -> Evaluation

Checkpoint:
  Lan 1: train tu dau, luu data + models vao checkpoints_weather5k/
  Lan 2+: load tu checkpoint, skip processing + training
  Train lai: python src/pipeline.py --force

Chay: python src/pipeline.py
"""
import sys
import os
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LABEL_CLF, LABEL_REG, TARGET_REG, CHECKPOINT_DIR, METRICS_FILE,
    NUM_STATIONS_SAMPLE,
)
from src.data_loader import create_spark_session, load_weather5k, print_summary
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.model_training import (
    split_data, train_classification, train_regression,
    save_processed_data, load_processed_data,
    save_train_test, load_train_test,
)
from src.evaluation import (
    evaluate_classification, evaluate_regression,
    print_metrics, confusion_matrix_summary, compare_models,
)
from pyspark.sql import functions as F


def run_classification_pipeline(train_df, test_df):
    print("\n" + "=" * 60)
    print("CLASSIFICATION: Du doan Rain_next (co mua gio tiep theo?)")
    print("=" * 60)

    clf_results = {}

    for model_name in ["lr", "rf", "gbt"]:
        model = train_classification(train_df, model_name)
        preds = model.transform(test_df)
        metrics = evaluate_classification(preds, LABEL_CLF)
        print_metrics(metrics, model_name.upper())
        clf_results[model_name.upper()] = metrics

        if model_name == "rf":
            confusion_matrix_summary(preds, LABEL_CLF)

    compare_models(clf_results)
    return clf_results


def run_regression_pipeline(train_df, test_df):
    print("\n" + "=" * 60)
    print("REGRESSION: Du doan Temp_next (nhiet do gio tiep theo)")
    print("=" * 60)

    train_reg = train_df.withColumn(LABEL_REG, F.col(TARGET_REG).cast("double"))
    test_reg = test_df.withColumn(LABEL_REG, F.col(TARGET_REG).cast("double"))

    reg_results = {}

    for model_name in ["linear", "rf", "gbt"]:
        model = train_regression(train_reg, model_name)
        preds = model.transform(test_reg)
        metrics = evaluate_regression(preds, LABEL_REG)
        print_metrics(metrics, model_name.upper())
        reg_results[model_name.upper()] = metrics

    compare_models(reg_results)
    return reg_results


def save_metrics(clf_results, reg_results, train_count, test_count):
    """Luu metrics ra JSON cho dashboard."""
    data = {
        "classification": clf_results,
        "regression": reg_results,
        "train_count": train_count,
        "test_count": test_count,
    }
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [CHECKPOINT] Metrics saved -> {METRICS_FILE}")


def main():
    force = "--force" in sys.argv

    if force and os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print("[FORCE] Da xoa checkpoints. Train lai tu dau.\n")

    print("=" * 60)
    print("  HE THONG DU BAO THOI TIET - APACHE SPARK")
    print(f"  Dataset: WEATHER-5K ({NUM_STATIONS_SAMPLE} stations sampled)")
    print("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    # Load from checkpoint or raw
    train_df, test_df = load_train_test(spark)

    if train_df is not None and test_df is not None:
        print("\n>>> Load train/test tu checkpoint. Skip preprocessing.")
    else:
        df = load_processed_data(spark)

        if df is not None:
            print("\n>>> Load processed data tu checkpoint.")
        else:
            print("\n--- LOAD WEATHER-5K ---")
            df = load_weather5k(spark)
            print_summary(df)

            df = preprocess(df)
            df = engineer_features(df)
            save_processed_data(df)

        print("\n--- SPLIT DATA ---")
        train_df, test_df = split_data(df)
        save_train_test(train_df, test_df)

    train_df.cache()
    test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"\nTrain: {train_count:,} | Test: {test_count:,}")

    clf_results = run_classification_pipeline(train_df, test_df)
    reg_results = run_regression_pipeline(train_df, test_df)

    # Save metrics cho dashboard
    save_metrics(clf_results, reg_results, train_count, test_count)

    print("\n" + "=" * 60)
    print("PIPELINE HOAN TAT")
    print("=" * 60)
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print("Train lai: python src/pipeline.py --force")

    spark.stop()
    return clf_results, reg_results


if __name__ == "__main__":
    main()

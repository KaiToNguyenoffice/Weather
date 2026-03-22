from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import functions as F

from src.config import LABEL_CLF, LABEL_REG


def evaluate_classification(predictions, label_col=None):
    if label_col is None:
        label_col = LABEL_CLF

    binary_eval = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction"
    )
    multi_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction"
    )

    auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})

    multi_eval.setMetricName("accuracy")
    accuracy = multi_eval.evaluate(predictions)

    multi_eval.setMetricName("f1")
    f1 = multi_eval.evaluate(predictions)

    multi_eval.setMetricName("weightedPrecision")
    precision = multi_eval.evaluate(predictions)

    multi_eval.setMetricName("weightedRecall")
    recall = multi_eval.evaluate(predictions)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc,
    }
    return metrics


def evaluate_regression(predictions, label_col=None):
    if label_col is None:
        label_col = LABEL_REG

    reg_eval = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction"
    )

    reg_eval.setMetricName("rmse")
    rmse = reg_eval.evaluate(predictions)

    reg_eval.setMetricName("mae")
    mae = reg_eval.evaluate(predictions)

    reg_eval.setMetricName("r2")
    r2 = reg_eval.evaluate(predictions)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }
    return metrics


def print_metrics(metrics, model_name=""):
    title = f"  {model_name}" if model_name else ""
    print(f"\n--- Evaluation{title} ---")
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v:.4f}")


def confusion_matrix_summary(predictions, label_col=None):
    if label_col is None:
        label_col = LABEL_CLF

    cm = (
        predictions
        .groupBy(label_col, "prediction")
        .count()
        .orderBy(label_col, "prediction")
    )
    print("\n--- Confusion Matrix ---")
    cm.show()

    tp = predictions.filter(
        (F.col(label_col) == 1) & (F.col("prediction") == 1)
    ).count()
    tn = predictions.filter(
        (F.col(label_col) == 0) & (F.col("prediction") == 0)
    ).count()
    fp = predictions.filter(
        (F.col(label_col) == 0) & (F.col("prediction") == 1)
    ).count()
    fn = predictions.filter(
        (F.col(label_col) == 1) & (F.col("prediction") == 0)
    ).count()

    print(f"  TP={tp:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TN={tn:,}")
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compare_models(results):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    if not results:
        return

    first_metrics = list(results.values())[0]
    header = f"{'Model':<25s}"
    for metric_name in first_metrics:
        header += f"{metric_name:>12s}"
    print(header)
    print("-" * len(header))

    for model_name, metrics in results.items():
        row = f"{model_name:<25s}"
        for v in metrics.values():
            row += f"{v:>12.4f}"
        print(row)

    best_metric_name = list(first_metrics.keys())[0]
    best_model = max(results, key=lambda m: results[m][best_metric_name])
    print(f"\nBest model ({best_metric_name}): {best_model}")

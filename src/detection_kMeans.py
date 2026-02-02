import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, when, count, sum as spark_sum, variance
import pandas as pd

os.environ["HADOOP_HOME"] = "C:\\hadoop"

spark = SparkSession.builder.appName("FraudDetectionClustering").getOrCreate()

df = spark.read.csv("data/creditcard.csv", header=True, inferSchema=True)

total_count = df.count()
print(f"Total records: {total_count}")

print("\nClass distribution:")
class_dist = df.groupBy("Class").count().orderBy("Class")
class_dist.show()

# Prepare features
feature_cols = [c for c in df.columns if c != 'Class']
print(f"\nFeature count: {len(feature_cols)}")

# Vectorize features
vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = vec_assembler.transform(df)

# STD-Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scalerModel = scaler.fit(data)
data = scalerModel.transform(data)

# choose k (cluster amount) by testing 2 - 15
# does not really choose "optimal" k but good enough

fraud_variance_by_k = {}
for k in range(2, 16):
    kmeans_tmp = KMeans(k=k,seed=1,featuresCol="scaledFeatures",maxIter=20,initSteps=100)
    model_tmp = kmeans_tmp.fit(data)
    preds_tmp = model_tmp.transform(data)

    cluster_stats_tmp = (
        preds_tmp.groupBy("prediction")
        .agg(
            spark_sum("Class").alias("fraud"),
            count("*").alias("total")
        )
        .withColumn("fraud_rate", col("fraud") / col("total"))
    )

    fraud_variance = (
        cluster_stats_tmp
        .select(variance("fraud_rate"))
        .first()[0]
    )

    fraud_variance_by_k[k] = fraud_variance
    print(f"k={k}, fraud_rate_variance={fraud_variance:.6f}")

# pick smallest k that achieves max fraud separation
optimal_k = max(
    fraud_variance_by_k,
    key=lambda k: (fraud_variance_by_k[k], -k)
)

print(f"Using k={optimal_k}")

kmeans = KMeans(featuresCol="scaledFeatures", k=optimal_k, seed=1, maxIter=200,initSteps=100)
model = kmeans.fit(data)
predictions = model.transform(data)

# fraud distribution across clusters
cluster_analysis = predictions.groupBy("prediction", "Class").count()
cluster_pivot = cluster_analysis.groupBy("prediction").pivot("Class").sum("count").fillna(0)
cluster_pivot = cluster_pivot.withColumnRenamed("0", "normal").withColumnRenamed("1", "fraud")
cluster_pivot = cluster_pivot.withColumn("total", col("normal") + col("fraud"))
cluster_pivot = cluster_pivot.withColumn("fraud_rate_pct", (col("fraud") / col("total") * 100))
cluster_pivot = cluster_pivot.orderBy("fraud_rate_pct", ascending=False)

print("\nCluster Statistics:")
cluster_pivot.select("prediction", "normal", "fraud", "total", "fraud_rate_pct").show()

# Identify fraud clusters
fraud_threshold = 40.0
high_risk = cluster_pivot.filter(col("fraud_rate_pct") > fraud_threshold)
high_risk_clusters = [row['prediction'] for row in high_risk.collect()]

print(f"\nHigh-risk clusters (fraud rate > {fraud_threshold}%): {high_risk_clusters}")

predictions_flagged = predictions.withColumn(
    "predicted_fraud",
    when(col("prediction").isin(high_risk_clusters), 1).otherwise(0)
)

# Calculate confusion matrix
print("\nConfusion Matrix:")
confusion = predictions_flagged.groupBy("Class", "predicted_fraud").count()
confusion.orderBy("Class", "predicted_fraud").show()
confusion_pd = confusion.toPandas()
confusion_pd.to_csv("results/kmeans_confusion_matrix.csv", index=False)
print("Confusion matrix saved")

# Calculate metrics
TP = predictions_flagged.filter((col("Class") == 1) & (col("predicted_fraud") == 1)).count()
FP = predictions_flagged.filter((col("Class") == 0) & (col("predicted_fraud") == 1)).count()
TN = predictions_flagged.filter((col("Class") == 0) & (col("predicted_fraud") == 0)).count()
FN = predictions_flagged.filter((col("Class") == 1) & (col("predicted_fraud") == 0)).count()

print("=========METRICS=========")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

print(f"\nTrue Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Negatives (FN): {FN}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

cluster_stats_pd = cluster_pivot.toPandas()
cluster_stats_pd.to_csv("results/fraud_cluster_analysis.csv", index=False)
print("\nResults saved")

# Export metrics for R
metrics_df = pd.DataFrame({
    'metric': ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1_Score'],
    'value': [TP, FP, TN, FN, accuracy, precision, recall, f1]
})
metrics_df.to_csv("results/kmeans_metrics.csv", index=False)
print("Metrics saved")

spark.stop()
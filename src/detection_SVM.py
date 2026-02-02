import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import time

os.environ["HADOOP_HOME"] = "C:\\hadoop"

spark = SparkSession.builder.appName("FraudDetection_SVM").getOrCreate()
df = spark.read.csv("data/creditcard.csv", header=True, inferSchema=True)

total_records = df.count()
print(f"Total records: {total_records:,}")

print("\nClass distribution:")
class_dist = df.groupBy("Class").count().orderBy("Class")
class_dist.show()

fraud_count = df.filter(col("Class") == 1).count()
normal_count = df.filter(col("Class") == 0).count()
fraud_ratio = fraud_count / total_records * 100

print(f"Fraud: {fraud_count:,} ({fraud_ratio:.2f}%)")
print(f"Normal: {normal_count:,} ({100-fraud_ratio:.2f}%)")

# Prepare features
feature_cols = [c for c in df.columns if c != 'Class']
print(f"\nUsing {len(feature_cols)} features")

vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = vec_assembler.transform(df)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", 
                        withStd=True, withMean=False)
scalerModel = scaler.fit(data)
data = scalerModel.transform(data)

# Split data 80 20
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1)

train_count = train_data.count()
test_count = test_data.count()
print(f"Training set: {train_count:,}")
print(f"Test set: {test_count:,}")

print("\nTraining set distribution:")
train_data.groupBy("Class").count().show()

print("Test set distribution:")
test_data.groupBy("Class").count().show()


print("\nTraining SVM...")
start_time = time.time()

svm = LinearSVC(
    featuresCol="scaledFeatures",
    labelCol="Class",
    maxIter=100,
    regParam=0.01,
    standardization=False,
    threshold=0.5
)

model = svm.fit(train_data)
train_time = time.time() - start_time

print(f"train time: {train_time:.2f} seconds")

# Prediction
predictions = model.transform(test_data)

# Confusion Matrix
print("\nConfusion Matrix:")
confusion = predictions.groupBy("Class", "prediction").count()
confusion.orderBy("Class", "prediction").show()

# Export confusion matrix for R
confusion_pd = confusion.toPandas()
confusion_pd.to_csv("results/svm_confusion_matrix.csv", index=False)

# Calculate metrics
TP = predictions.filter((col("Class") == 1) & (col("prediction") == 1.0)).count()
FP = predictions.filter((col("Class") == 0) & (col("prediction") == 1.0)).count()
TN = predictions.filter((col("Class") == 0) & (col("prediction") == 0.0)).count()
FN = predictions.filter((col("Class") == 1) & (col("prediction") == 0.0)).count()

accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nTrue Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Negatives (FN): {FN}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

binary_evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
auc_roc = binary_evaluator.evaluate(predictions)
print(f"AUC-ROC:   {auc_roc:.4f}")

binary_evaluator_pr = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderPR")
auc_pr = binary_evaluator_pr.evaluate(predictions)
print(f"AUC-PR:    {auc_pr:.4f}")

print(f"\nFrauds detected: {TP}/{TP+FN}")

# Export performance metrics for R
metrics_df = pd.DataFrame({
    'metric': ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC', 'AUC_PR'],
    'value': [TP, FP, TN, FN, accuracy, precision, recall, f1, auc_roc, auc_pr]
})

metrics_df.to_csv("results/svm_metrics.csv", index=False)
spark.stop()
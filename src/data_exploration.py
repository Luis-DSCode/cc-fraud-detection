from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, mean, stddev
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("data/creditcard.csv", header=True, inferSchema=True)

print("Shape:")
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

df.groupBy("Class").count().show()

fraud_count = df.filter(col("Class") == 1).count()
total_count = df.count()
print(f"\nFraud percentage: {(fraud_count/total_count)*100:.2f}%")

df.describe().show()

print("\nMissing values:")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

spark.stop()
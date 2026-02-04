library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(scales)

theme_set(theme_minimal(base_size = 12))

kmeans_metrics <- read.csv("results/kmeans_metrics.csv")
kmeans_confusion <- read.csv("results/kmeans_confusion_matrix.csv")
kmeans_clusters <- read.csv("results/fraud_cluster_analysis.csv")
svm_metrics <- read.csv("results/svm_metrics.csv")
svm_confusion <- read.csv("results/svm_confusion_matrix.csv")

# Kmeans Fraud by Cluster
p1_kmeans <- ggplot(kmeans_clusters,aes(x = reorder(factor(prediction),-fraud_rate_pct),y = fraud_rate_pct)) +
  geom_col(aes(fill = fraud_rate_pct), width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", fraud_rate_pct)),vjust = -0.5, size = 3.5) +
  scale_fill_gradient(low = "#4CAF50", high = "#F44336",name = "Fraud Rate (%)") +
  labs(title = "K-Means: Fraud Rate by Cluster",
       subtitle = "Identifying high-risk transaction clusters",
       x = "Cluster ID",
       y = "Fraud Rate (%)") +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"))

ggsave("visualizations/kmeans_fraud_by_cluster.png", p1_kmeans, width = 10, height = 6, dpi = 300)

# Fraud in Cluster
p1_kmeans <- ggplot(kmeans_clusters, aes(x = reorder(factor(prediction), -fraud), y = fraud)) +
  geom_col(aes(fill = fraud), width = 0.7) +
  geom_text(aes(label = fraud), vjust = -0.5, size = 3.5) +
  scale_fill_gradient(low = "#4CAF50", high = "#F44336", name = "Fraud Cases") +
  labs(title = "K-Means: Fraud Cases by Cluster",
       subtitle = "Identifying high-risk transaction clusters",
       x = "Cluster ID",
       y = "Number of Fraud Cases") +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"))
ggsave("visualizations/kmeans_fraud_in_cluster.png", p1_kmeans, width = 10, height = 6, dpi = 300)

# Kmeans CMatrix
kmeans_conf_wide <- kmeans_confusion %>%
  mutate(Class = factor(Class, labels = c("Normal", "Fraud")),
         predicted_fraud = factor(predicted_fraud, labels = c("Normal", "Fraud"))) %>%
  rename(Actual = Class, Predicted = predicted_fraud)

p3_kmeans <- ggplot(kmeans_conf_wide, aes(x = Predicted, y = Actual, fill = count)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = comma(count)), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#E8F5E9", high = "#1B5E20", 
                      labels = comma, name = "Count") +
  labs(title = "K-Means: Confusion Matrix",
       subtitle = "Actual vs Predicted Classifications") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold"),
        legend.position = "right")

ggsave("visualizations/kmeans_confusion_matrix.png", p3_kmeans, width = 8, height = 6, dpi = 300)


# SVM CMatrix
svm_conf_wide <- svm_confusion %>%
  mutate(Class = factor(Class, labels = c("Normal", "Fraud")),
         prediction = factor(prediction, labels = c("Normal", "Fraud"))) %>%
  rename(Actual = Class, Predicted = prediction)

p1_svm <- ggplot(svm_conf_wide, aes(x = Predicted, y = Actual, fill = count)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = comma(count)), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#E3F2FD", high = "#0D47A1", 
                      labels = comma, name = "Count") +
  labs(title = "SVM: Confusion Matrix",
       subtitle = "Actual vs Predicted Classifications") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold"),
        legend.position = "right")

ggsave("visualizations/svm_confusion_matrix.png", p1_svm, width = 8, height = 6, dpi = 300)


#Comparison
comparison_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1_Score"),
  KMeans = kmeans_metrics$value[kmeans_metrics$metric %in% 
                                   c("Accuracy", "Precision", "Recall", "F1_Score")],
  SVM = svm_metrics$value[svm_metrics$metric %in% 
                            c("Accuracy", "Precision", "Recall", "F1_Score")]
) %>%
  pivot_longer(cols = c(KMeans, SVM), names_to = "Method", values_to = "Value")

p1_compare <- ggplot(comparison_df, aes(x = Metric, y = Value, fill = Method)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", Value)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("KMeans" = "#FF9800", "SVM" = "#2196F3")) +
  scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(title = "Performance Metrics Comparison",
       subtitle = "K-Means vs SVM Fraud Detection",
       x = "Metric",
       y = "Score",
       fill = "Method") +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "top",
        axis.text.x = element_text(angle = 0))

ggsave("visualizations/comparison_metrics.png", p1_compare, width = 10, height = 6, dpi = 300)
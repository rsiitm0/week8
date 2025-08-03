import mlflow
from mlflow import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--percentage_file',required=True)
args = parser.parse_args()
percentage_file_path = args.percentage_file

df = pd.read_csv(percentage_file_path)
fh_percentage_file_path = open(percentage_file_path)
all_percentages = [i.strip() for i in fh_percentage_file_path.readlines()]

mlflow_uri = "http://34.123.98.187:8000"
mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient(mlflow_uri)

mlflow.set_experiment("Week-8-data-poisoning") # experiment name
  
with mlflow.start_run(run_name="Accuracy metrics"):
    metrics_file = "iris_report.csv"
    df_metrics = pd.read_csv(metrics_file)
    # Log metrics to MLflow - without poison
    for idx, row in df_metrics.iterrows():
        mlflow.log_metric("Original - No poison", row["accuracy"], step=int(row["epoch"]))
        
    # Log metrics to MLflow - with data poison
    for i in all_percentages:
        metrics_file = "iris_report_"+str(i)+".csv"
        df_metrics = pd.read_csv(metrics_file)
        # Log metrics to MLflow
        for idx, row in df_metrics.iterrows():
            mlflow.log_metric(str(i)+" Percentage Data Poisoning", row["accuracy"], step=int(row["epoch"]))

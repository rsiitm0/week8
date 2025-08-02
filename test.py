import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data',required=True)
parser.add_argument('--model',required=True)
parser.add_argument('--report',required=True)
parser.add_argument('--epoch',required=True)
parser.add_argument('--ratio',required=True)
args = parser.parse_args()

data_path = args.data
model_path = args.model
report_path = args.report
epoch_count = int(args.epoch)
train_ratio = 1 - float(args.ratio)

data = pd.read_csv(data_path)
model_dt = joblib.load(model_path)

fh_report = open(report_path,'w')
fh_report.write(f"epoch,accuracy\n")

for i in (range(epoch_count)):
  train,test = train_test_split(data,test_size=train_ratio,stratify=data['species'],random_state=i)
  X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
  y_test = test.species
  y_pred = model_dt.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  fh_report.write(f"{i},{accuracy:.4f}\n")


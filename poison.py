import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--percent',required=True)
args = parser.parse_args()

poison_percentage = int(args.percent)

df_original = pd.read_csv("iris.csv")
number_of_rows = len(df_original)

def poison(percent: int, poisoned_csv: str):
    number_to_be_poisoned = int(number_of_rows * percent /100 )
    poisoned_rows = random.sample(range(number_of_rows),number_to_be_poisoned)
    df_poisoned = df_original.copy()
    all_species = set(df_original['species'].tolist())
    for i in poisoned_rows:
        current_species = df_original.loc[i,'species']
        new_species = random.sample(sorted(all_species-set([current_species])),1)[0]
        df_poisoned.loc[i,'species'] = new_species
    df_poisoned.to_csv(poisoned_csv,index=False)


out_poisoned_file_name = "iris_"+str(poison_percentage)+".csv"
poison(poison_percentage,out_poisoned_file_name)


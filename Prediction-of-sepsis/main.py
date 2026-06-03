import pandas as pd

df = pd.read_csv(r'C:\Users\poelarer\Downloads\prediction-of-sepsis\Dataset.csv')

patient_units = df.groupby("Patient_ID")[['Unit1', 'Unit2']].max().reset_index()

patient_units_sample = patient_units.sample(n=4000, random_state=42)

df_selected = df[df["Patient_ID"].isin(patient_units_sample["Patient_ID"])]

df_train = df[~df["Patient_ID"].isin(patient_units_sample["Patient_ID"])]

df_selected.to_csv("testset (with label).csv", index=False)

df_selected.drop(columns=["SepsisLabel"]).to_csv("testset (without label).csv", index=False)

df_train.to_csv("trainset.csv", index=False)

print(len(patient_units))

print(len(patient_units_sample))
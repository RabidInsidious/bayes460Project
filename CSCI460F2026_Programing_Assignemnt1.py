import pandas as pd

df = pd.read_csv("bank-full.csv")

df = df.replace(";", ",", regex=True)

d = {"yes":1,"no":0} # mapping to 0-1

for col in df.columns:
    df[col] = df[col].map(d)

df = df.dropna()

X = df.drop(columns=['y'])  # All columns except y
y = df['y']                 # Target column

# print(df)
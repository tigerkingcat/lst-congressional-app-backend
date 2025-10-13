import pandas as pd
import numpy as np

df = pd.read_csv(r"./model/final_hvi_data.csv")

first_col = df.columns[0]
df = df.drop(columns=[first_col])

df.to_csv("final_hvi_data.csv", index=False)

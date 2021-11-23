import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("covid.csv")
df = df.drop(columns=['Sorszám'])
df.sort_values(by=['Kor'], inplace=True)
#df.loc[df["Nem"] == "Nõ"] = "Women"
#df.to_csv("df.csv")
print(df.head(10))

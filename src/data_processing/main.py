import numpy as np
import requests
import pandas as pd
import os
import seaborn as sns


def main():
    url = 'https://koronavirus.gov.hu/elhunytak'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[-1]
    # Just printing the very first page
    for i in range(647):
        url = 'https://koronavirus.gov.hu/elhunytak?page=' + str(i)
        # print(url)
        html = requests.get(url).content
        df_list1 = pd.read_html(html)
        df1 = df_list1[-1]
        df = df.append(df1)
    df.to_csv("raw_data.csv", index=False)



if __name__ == "__main__":
    main()

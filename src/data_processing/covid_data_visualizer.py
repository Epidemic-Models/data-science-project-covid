from datetime import datetime
from IPython.display import display, HTML


import json as json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as gp
import plotly.io as pio
import requests
import seaborn as sns


class DataProcessor:
    def __init__(self, raw_data_path="data_processing/raw_data.csv",
                 date_data_path="data_processing/dates.csv"):
        self.raw_data_path = raw_data_path
        self.date_data_path = date_data_path

        # Declare member variables
        self.data = None
        self.dates = None
        self.data_with_dates = None
        self.aggregated_data = None
        self.heatmap = None
        self.aggregated_data_ratio = None
        self.normalized_data_heatmap = None
        self.population_pyramid = None
        self.figg = None


    @staticmethod
    def download_data() -> None:
        print("The download_data() method has started to run. It can take some time, please be patient")
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
        print("The download_data() method is done. Thank you for your time.")

    def read_raw_data(self):
        #path = self.raw_data_path
        df = pd.read_csv("raw_data.csv")
        df = df.drop(columns=["Sorszám", "Alapbetegségek"])
        df.drop(df.index[0:215], 0, inplace=True)
        df = df.reindex(index=df.index[::-1])
        df.reset_index(drop=True, inplace=True)
        self.data = df
        print("read_raw_data method is done")

    def get_date_data(self):
        read_file = pd.DataFrame(pd.read_excel("dates.xlsx"))
        read_file.to_csv("dates.csv", index=None, header=True)
        df1 = pd.read_csv("dates.csv", usecols=['Dátum', 'Elhunytak', 'New Deaths'])
        df1.drop(df1.index[0:12], 0, inplace=True)
        df1['Elhunytak'] = df1['Elhunytak'].fillna(0)
        df1['New Deaths'] = df1['New Deaths'].fillna(0)
        df1.reset_index(drop=True, inplace=True)
        self.dates = df1
        print("get_date_data method is done!")

    def get_data_with_dates(self):
        repeat = self.dates["New Deaths"]
        data = pd.DataFrame(np.repeat(self.dates.values, repeat, axis=0), columns=self.dates.columns)
        new_data = pd.merge(data, self.data, left_index=True, right_index=True)
        new_data = new_data.drop(columns=['Elhunytak', 'New Deaths'])
        new_data["Dátum"] = pd.to_datetime(new_data["Dátum"])
        new_data.set_index("Dátum", inplace=True)
        self.dates["Dátum"] = pd.to_datetime(self.dates["Dátum"])
        self.dates.set_index("Dátum", inplace=True)
        data_with_dates = pd.merge(self.dates, new_data, how='inner', left_index=True, right_index=True)
        data_with_dates.to_csv("data_with_dates.csv")
        data_with_dates.reset_index(level=0, inplace=True)
        bins = pd.DataFrame(pd.cut(x=data_with_dates['Kor'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109]))
        bins.rename(columns={'Kor': 'Age'}, inplace=True)
        data_with_dates = pd.concat((data_with_dates, bins), axis=1)
        data_with_dates.drop(columns=["Elhunytak", "New Deaths"], inplace=True)
        self.data_with_dates = data_with_dates
        print("get_data_with_dates method is done!")

    def get_aggregated_data(self):
        rep = pd.DataFrame(self.data_with_dates.groupby(pd.Grouper(freq='W', key='Dátum'))['Nem'].count())
        rep.reset_index(inplace=True)
        rep1 = rep["Nem"]
        mydata = pd.DataFrame(np.repeat(rep.values, rep1, axis=0), columns=rep.columns)
        mydata["Date"] = mydata["Dátum"]
        mydata["Count"] = mydata["Nem"]
        mydata.drop(columns=["Dátum", "Nem"], inplace=True)
        date_age = pd.merge(mydata, self.data_with_dates, how='inner', left_index=True, right_index=True)
        date_age.head()
        date_age.drop(columns=["Dátum"], inplace=True)
        date_age["Date"] = date_age["Date"].dt.date
        date_age.drop(columns=["Count", "Nem", "Kor"], inplace=True)
        aggregated_data = date_age.groupby(list(date_age))[['Date']].count()
        aggregated_data = aggregated_data.unstack(level=0)
        aggregated_data.columns = aggregated_data.columns.droplevel()
        aggregated_data = aggregated_data.reindex(index=aggregated_data.index[::-1])
        aggregated_data.to_csv("aggregated_data.csv")
        self.aggregated_data = aggregated_data
        print("get_aggregated_data method is done!")

    def plot_heatmap(self):
        self.heatmap = sns.heatmap(self.aggregated_data)
        #plt.show()
        #return self.heatmap

    def get_aggregated_data_ratio(self):
        weekly = self.aggregated_data.columns
        rows = self.aggregated_data.index
        rows = rows.categories[:10]
        rows = rows[::-1]
        table = self.aggregated_data.to_numpy()
        new_table = table[1:, :]
        file = open("populacio.json")
        population = json.load(file)
        dict_to_list = list(population.items())
        new_list = pd.DataFrame(dict_to_list, columns=['Age', 'popln'])
        new_list["Age"] = new_list["Age"].astype(int)
        new_list['Age'] = pd.DataFrame(
        pd.cut(x=new_list['Age'], bins=[-0.1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109]))
        aggregated_list = new_list.groupby(['Age'])['popln'].agg('sum')
        table1 = aggregated_list.to_numpy()
        table1 = table1[::-1]
        table1 = table1.reshape(len(table1), 1)
        table2 = new_table / table1
        aggregated_data_ratio = pd.DataFrame(table2, columns=weekly, index=rows)
        aggregated_data_ratio.to_csv("aggregated_data_ratio.csv")
        self.aggregated_data_ratio = aggregated_data_ratio
        print("get_aggregated_data_ratio method is done!")


    def plot_normalized_data_heatmap(self):
        self.normalized_data_heatmap = sns.heatmap(self.aggregated_data_ratio)

    def get_population_pyramid_data(self):
        self.data["Nem"].replace({"férfi": "Férfi", "nõ": "Nõ", "Nő": "Nõ"}, inplace=True)
        self.data = self.data.groupby(list(self.data))[['Nem']].count()
        self.data = self.data.unstack(level=0)
        self.data.reset_index(level=0, inplace=True)
        self.data.columns = self.data.columns.droplevel()
        self.data.rename(columns={"": "Kor"}, inplace=True)
        self.data['Férfi'] = self.data['Férfi'].fillna(0)
        self.data['Nõ'] = self.data['Nõ'].fillna(0)
        self.data["Férfi"] = - self.data["Férfi"]

    def plot_population_pyramid_1(self):
        plt.barh(self.data["Kor"], self.data["Férfi"], color='r', lw=0, )
        plt.barh(self.data["Kor"], self.data["Nõ"], color='b', lw=0, )
        plt.title("Pyramid Age Distribution of Genders", fontsize=12)
        plt.xlabel("Male/Female")
        plt.show()

    def plot_population_pyramid(self):
        #pio.renderers.default = 'png'
        fig = gp.Figure()

        # Adding Male data to the figure
        fig.add_trace(gp.Bar(y=self.data["Kor"], x=self.data["Férfi"],
                             name='Male',
                             orientation='h'))

        # Adding Female data to the figure
        fig.add_trace(gp.Bar(y=self.data["Kor"], x=self.data["Nõ"],
                             name='Female', orientation='h'))

        # Updating the layout for the graph
        fig.update_layout(title='Population Pyramid for COVID-19 deceased, Hungary',
                          title_font_size=22, barmode='relative',
                          bargap=0.0, bargroupgap=0,
                          xaxis=dict(tickvals=[-600, -500, -400, -300, -200, -100,
                                               0, 100, 200, 300, 400, 500, 600],

                                     ticktext=[600, 500, 400, 300, 200, 100,
                                               0, 100, 200, 300, 400, 500, 600],

                                     title='Population',
                                     title_font_size=14)
                          )
        self.figg = fig
        #fig.show()


def main():
    p = DataProcessor()
    p.download_data()
    p.read_raw_data()
    p.get_date_data()
    p.get_data_with_dates()
    p.get_aggregated_data()
    p.plot_heatmap()
    p.heatmap
    plt.show()
    p.get_aggregated_data_ratio()
    p.plot_normalized_data_heatmap()
    p.normalized_data_heatmap
    plt.show()
    p.get_population_pyramid_data()
    p.plot_population_pyramid_1()
    p.plot_population_pyramid()
    p.figg.show()


if __name__ == "__main__":
    main()

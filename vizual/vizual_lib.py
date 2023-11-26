import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np


class VizualLib:

    def __init__(self, X: pd.DataFrame, target_column: str, transform: bool = True):
        self.__data: pd.DataFrame = X.copy()
        self.__transform = transform

        self.__transform_data()
        self.__target = self.__data[target_column]

    def __transform_data(self):
        if not self.__transform:
            return

        object_columns = [col for col in self.__data if self.__data[col].dtype == 'object']

        ordinal_encoder = OrdinalEncoder()

        transformed_X = self.__data.copy()
        transformed_X[object_columns] = ordinal_encoder.fit_transform(transformed_X[object_columns])

        self.__data = transformed_X

    def show_data(self):
        print(self.__data)
        return self.__data

    def heat_map(self, figsize: tuple = (16, 12), fmt: str = '.2f', annot_kws: dict = {"size": 8},
                 linewidths: float = 1.9, cmap: str = 'BuGn',
                 linecolor: str = 'w', square=True, **kwargs):
        mask = np.zeros_like(self.__data.corr(), dtype=np.float32)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=figsize)
        seaborn.heatmap(self.__data.corr(), linewidths=linewidths, square=square, cmap="BuGn",
                        linecolor=linecolor, annot=True, annot_kws=annot_kws, mask=mask, cbar_kws={"shrink": .5}, ax=ax,
                        fmt=fmt, **kwargs)

        plt.show()

    def barplot(self, col: str | list[str], figsize: tuple = (10, 6)):
        plt.figure(figsize=figsize)
        seaborn.barplot(x=self.__data[col], y=self.__target)

    def lineplot(self, col: str | list[str], figsize: tuple = (20, 6)):
        plt.figure(figsize=figsize)
        seaborn.lineplot(x=self.__data[col], y=self.__target)

    def hist(self, bins=40, figsize=(18, 14)):
        self.__data.hist(bins=bins, figsize=figsize)

    def plt_hist(self, col: str, bins=50, figsize: tuple = (10, 6)):
        plt.figure(figsize=figsize)
        plt.hist(self.__data[col], bins=bins, ec='black', color='#2196f3')
        plt.xlabel(col)
        plt.ylabel(self.__target.name)
        plt.show()

    @staticmethod
    def scatter(X, y, result, line_color='pink', dots_color='gray'):
        plt.scatter(X, y, color=line_color)
        plt.plot(X.to_numpy(), result, color=dots_color)
        plt.show()

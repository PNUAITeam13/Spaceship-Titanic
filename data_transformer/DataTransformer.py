import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class DataWithBackUp:
    def __init__(self, columns, data, backups):
        self.__columns = columns
        self.__data = data
        self.__backups = backups

    def get_data(self):
        return self.__data

    def get_backups(self):
        return self.__backups[self.__columns]


class DataTransformer:

    def __init__(self, replace_data_after_transformation: bool = False, use_backups: bool = True):
        self.__replace_data = replace_data_after_transformation
        self.__use_backups = use_backups

    def __use(self, function, data, columns):
        need_to_transform = columns if columns is not None else data.columns.tolist()

        data_copy = data.copy()

        (result, result_cols) = function(data_copy=data_copy, need_to_transform=need_to_transform)

        if result_cols is None:
            result_cols = need_to_transform

        backups = result.copy()[result_cols] if self.__use_backups else None
        return DataWithBackUp(columns, result.copy(), backups)

    def use_ordinal_encoder(self, data: DataFrame, columns=None) -> DataWithBackUp:
        def use_function(data_copy, need_to_transform):
            encoder = OrdinalEncoder()

            data_copy[need_to_transform] = encoder.fit_transform(data_copy[need_to_transform])
            return data_copy, need_to_transform

        return self.__use(use_function, data=data, columns=columns)

    def use_one_hot_encoder(self, data, columns, handle_unknown='ignore', sparse_output=False):
        def use_function(data_copy, need_to_transform):
            OH_encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)

            OH_X_cols = pd.DataFrame(OH_encoder.fit_transform(data_copy[need_to_transform]))
            OH_X_cols.index = data_copy.index
            new_cols = OH_encoder.get_feature_names_out(need_to_transform)
            OH_X_cols.columns = new_cols

            num_X = data_copy.drop(need_to_transform, axis=1)
            OH_X = pd.concat([num_X, OH_X_cols], axis=1)
            return OH_X, new_cols

        return self.__use(use_function, data=data, columns=columns)

    def fill_na_value(self, data, columns=None, new_value=10000) -> DataWithBackUp:
        return self.fill_na_function(data, columns, lambda c: new_value)

    def fill_na_supplier(self, data, columns=None, value_supplier=None) -> DataWithBackUp:
        return self.fill_na_function(data, columns, lambda c: value_supplier())

    def fill_na_function(self, data, columns=None, value_function=None) -> DataWithBackUp:
        def use_function(data_copy, need_to_transform):
            for column in need_to_transform:
                data_copy[column] = data_copy[column].fillna(value_function(data_copy[column]))
            return data_copy, columns

        return self.__use(use_function, data=data, columns=columns)

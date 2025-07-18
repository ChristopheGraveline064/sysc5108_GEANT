import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import asarray
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display

class GEANT:
    def __init__(self):

        self.dataset_summary = {
            'n_node_src': 1,
            'n_node_dst': 1,
            'n_link': 0,
            'n_path': 0,
            'n_flows': 96000,
            'n_tm': 96,
            'tm_t': 900,
            'max_bw': 102e5 #Kb or 10G
            #34000 Kb or Mb/1000
        }


    def load_data(self, data, header=True, drop_del=False):
        print("load the csv: " + data)
        try:
            if header:
                #self.data = pd.read_csv(data)
                self.data = pd.read_csv(data, nrows=10000000)
            else:
                self.data = pd.read_csv(data, header=None)

            if drop_del:
                self.data = self.drop_delimitor(self.data)

            return self.data
        except Exception as e:
            print(e)
            return None

    def drop_delimitor(self, data_drop_del):
        data_drop_del = data_drop_del[~data_drop_del['src'].str.contains(".*###.*")]
        data_drop_del = data_drop_del.reset_index(drop=True)
        return data_drop_del

    def split_data(self):
        X_train, X, y_train, y = train_test_split(self.features, self.target, test_size=0.5)
        X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def seperate_feature_targets(self, data):
        self.dataset_summary['n_link'] = sum(self.data.columns.str.contains('util_link_.*'))
        self.dataset_summary['n_path'] = sum(self.data.columns.str.contains('trgt_link_.*'))

        features = data.iloc[:, data.columns.str.contains('util_link_.*|src.*|dst.*|exp_bw_kbps')]
        target = data.iloc[:, data.columns.str.contains('trgt_link_.*')]

        print("Features:")
        display(features.head())

        print("Targets:")
        display(target.head())

        return features, target

    def normalize(self, data, column):
        data[column] = data[column]/self.dataset_summary['max_bw']
        return data


    def one_hot_encode_column(self, data, column):
        df_to_encode = data.iloc[:, data.columns == column]
        df_not_encode = data.iloc[:, data.columns != column]


        data_to_encode_arr = asarray(df_to_encode)
        data_to_encode = data_to_encode_arr.reshape(-1, 1)

        # define one hot encoding
        encoder = OneHotEncoder(sparse=False)
        # transform data
        onehot = encoder.fit_transform(data_to_encode_arr)
        num_rows, num_cols = onehot.shape

        self.dataset_summary['n_node_{}'.format(column)] = num_cols

        encoded_df = pd.DataFrame(onehot, columns = ["{}_{}".format(column, i) for i in range(num_cols)])
        data = encoded_df.join(df_not_encode)

        del [[encoded_df, df_not_encode,data_to_encode_arr]]

        return data


    def split_by_time_series(self, test_size=0.25, train_size=0.25):
        #-----------------------------------------------#
        #                      testing set              #
        #                                               #
        #                                               #
        #-----------------------------------------------#
        #                      training set             #
        #                                               #
        #                                               #
        #-----------------------------------------------#
        #                      validation set           #
        #                                               #
        #                                               #
        #-----------------------------------------------#


        contains_delimiter = self.data['src'].str.contains(".*###.*")
        len_dataset = len(contains_delimiter)
        print("length of the dataset: {}".format(len_dataset))
        n_time_series = sum(contains_delimiter)
        print("number of time series: {}".format(n_time_series))

        testing_index = round(n_time_series * test_size)
        training_index = testing_index + round(n_time_series * train_size)
        #index of all the row containing ######
        self.delimiter_index = contains_delimiter[contains_delimiter].index.values

        self.testing_end_index = (self.delimiter_index[testing_index] + 1)
        print("testing end index: {}".format(self.testing_end_index))
        self.training_end_index = (self.delimiter_index[training_index] + 1)
        print("training end index: {}".format(self.training_end_index))

        contains_delimiter[0:self.testing_end_index] = 'testing'
        contains_delimiter[self.testing_end_index + 1:self.training_end_index] = 'training'
        contains_delimiter[self.training_end_index + 1:] = 'validation'

        contains_delimiter = pd.DataFrame(data=contains_delimiter)
        contains_delimiter.columns = ['set']

        self.data = self.data.join(contains_delimiter)

        self.testing_set = self.data.loc[self.data['set'] == 'testing']
        self.training_set = self.data.loc[self.data['set'] == 'training']
        self.validation_set = self.data.loc[self.data['set'] == 'validation']
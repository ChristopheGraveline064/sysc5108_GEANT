import os

import numpy as np

from ANN import ANN
import pandas as pd
from Dataset import GEANT
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':

    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")

    #####################################################DATA HANDLING##################################################
    #input: source_1 ... source_n, destination_1 ... destination_n, bw_j, utilization_1 ... utilization_l
    #len(input): 2 X |V| + 1 + |E|
    #output: path_1 ... path_l
    #len(output): |E|

    #load data
    data_dir = './'
    data_file = 'dump_23_03_2022_19-08-30_withheader.csv'
    path = data_dir + data_file

    dataset = GEANT()
    dataset.load_data(path)

    #process data
    dataset.split_by_time_series()

    dataset.training_set = dataset.drop_delimitor(dataset.training_set)
    testing_set = dataset.drop_delimitor(dataset.testing_set)
    dataset.validation_set = dataset.drop_delimitor(dataset.validation_set)


    dataset.training_set = dataset.normalize(dataset.training_set, 'exp_bw_kbps')
    dataset.training_set = dataset.one_hot_encode_column(dataset.training_set, 'dst')
    dataset.training_set = dataset.one_hot_encode_column(dataset.training_set, 'src')

    testing_set = dataset.normalize(testing_set, 'exp_bw_kbps')
    testing_set = dataset.one_hot_encode_column(testing_set, 'dst')
    testing_set = dataset.one_hot_encode_column(testing_set, 'src')

    dataset.validation_set = dataset.normalize(dataset.validation_set, 'exp_bw_kbps')
    dataset.validation_set = dataset.one_hot_encode_column(dataset.validation_set, 'dst')
    dataset.validation_set = dataset.one_hot_encode_column(dataset.validation_set, 'src')

    data_train, target_train = dataset.seperate_feature_targets(dataset.training_set)
    data_test, target_test = dataset.seperate_feature_targets(testing_set)
    data_val, target_val = dataset.seperate_feature_targets(dataset.validation_set)


    #####################################################LOAD MODEL#####################################################
    model = ANN(dataset.dataset_summary)

    #####################################################PREPROCESSING##################################################

    #####################################################TRAIN##########################################################
    #model.load('parameter30_03_2022_17_56_54.h5')
    history = model.train(data_train, target_train, data_val, target_val, save=True, file_name="parameter{}.h5".format(timestamp))

    #####################################################VALIDATION#####################################################

    print("Prediction")
    prediction = (model.predict(data_test))

    #prediction = pd.DataFrame(prediction, columns=["pred_link_{}".format(i.replace("trgt_link_", "")) for i in target_test.columns])

    '''print("Shift rows proba")
    for row in dataset.delimiter_index:
        if row <= (dataset.testing_end_index - 1):
            last_row = prediction.iloc[-1, :].copy()
            prediction.iloc[row:, :] = prediction.iloc[row:, :].shift(1, axis="index")
            if (row != (dataset.testing_end_index - 1)):
                prediction = prediction.append(last_row, ignore_index=True)'''

    thres = 1E-10
    #dataset.testing_set = dataset.testing_set.drop(['set'], axis=1)
    #testing_set_w_pred = dataset.testing_set.join(prediction)

    #print('export prediction_{}.csv'.format(timestamp))
    #testing_set_w_pred.to_csv('prediction_{}.csv'.format(timestamp),index=False)

    #select_column = testing_set_w_pred.columns.str.contains('pred_link_.*')

    #testing_set_w_pred.iloc[:, select_column] = np.where(testing_set_w_pred.iloc[:, select_column] >= thres, 1, testing_set_w_pred.iloc[:, select_column])
    #testing_set_w_pred.iloc[:, select_column] = np.where(testing_set_w_pred.iloc[:, select_column] < thres, 0, testing_set_w_pred.iloc[:, select_column])

    #prediction = np.where(prediction >= thres, 1, prediction)
    #prediction = np.where(prediction < thres, 0, prediction)

    target_pred = prediction.flatten()
    target = target_test
    target = target.to_numpy()
    target = target.flatten()
    model.precision_recall_curve_plot(target, target_pred)
    #model.get_confusion_matrix_result(target, target_pred, "th = {}".format(thres))
    plt.show()

    #print('export binary_prediction_{}.csv'.format(timestamp))
    #testing_set_w_pred.to_csv('binary_prediction_{}.csv'.format(timestamp), index=False)


    ###################################################POSTPROCESSING###################################################
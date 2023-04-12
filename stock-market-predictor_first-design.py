"""
Required actions:
 - upgrade analysis of up down betting with for loop
 - change all mentions of close to "output"
- rearrange method declaration as needed
- actions about the drop NA needs to be done, this will unalign my data
- I'm unsure if the finder function is wiping all information between iterations
- the disabling of the warnings needs to be better controlled

Dev notes:
 - Cound potentially add the function to view the model parameters scores

"""
#%% Imports Modules and Define Basic Parameters

import numpy as np
import pandas as pd

import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns 
import jupyter
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import copy
import datetime
import os
import sys

#DoE params
NN_hidden_layer_sizes_strat = (range(10, 31, 10), range(100, 301, 10))
NN_activation_strat         = ['relu', 'logistic']
model_types_and_params_dict = {
    "overall strat" : {
        "param search full" : True,
        "predict on only best params post CV analysis" : True
    },
    "MLPRegressor" : { #Multi-layer Perceptron regressor
        "hidden_layer_sizes"    : NN_hidden_layer_sizes_strat, 
        "activation"            : NN_activation_strat},
    "ElasticNet" : { #Linear regression with combined L1 and L2 priors as regularizer.
        'estimator__alpha':[0.1, 0.5, 0.9], 
        'estimator__l1_ratio':[0.1, 0.5, 0.9]
    }
}

#stratergy params
Features = 6
pred_output_and_tickers_combos_list = [("aapl", "<CLOSE>"), ("carm", "<HIGH>")]
pred_steps_list                     = [1,2,5,10]
time_series_split_qty               = 5
training_error_measure_main         = 'neg_mean_squared_error'
train_test_split                    = 0.7 #the ratio of the time series used for training
CV_Reps                             = 2
time_step_notation_sting            = "d" #day here is for a day, update as needed
fin_indi                            = [] #financial indicators

#file params
input_cols_to_include_list = ["<CLOSE>", "<HIGH>"]
index_cols_list = ["<DATE>","<TIME>"]
index_col_str = "datetime"
target_folder_path_list=["C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Financial Data\\h_us_txt\\data\\hourly\\us\\nasdaq stocks\\1 - Copy\\"]
target_file_folder_path = ""
feature_qty = 6
outputs_folder_path = ".\\outputs\\"


#Blank Variables (to remove problem messages)

#generic strings

#placeholder variables
df_financial_data = np.nan


#%% Misc Methods

def return_ticker_code_1(filename):
    return filename[:filename.index(".")]
    
def current_infer_values_method(df):
    
    nan_values_removed = 0
    for col in df.columns:
        good_indexes = df[col][df[col] > 0].index
        faulty_indexes = df[col].drop(good_indexes).index
        for faulty_index in faulty_indexes:
            nan_values_removed += 1
            #previous_row          = df.index.match(faulty_index) - 1
            previous_row          = list(df.index).index(faulty_index) - 1
            previous_index        = df.index[previous_row]
            df[col][faulty_index] = df[col][previous_index]
    
    return df, nan_values_removed



#%% Experiment Handlers Methods

def define_DoE(DoE_name, model_types_and_params_dict, fin_data_input, fin_indi, pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, train_test_split=train_test_split):
    df_financial_data = import_financial_data(
        target_folder_path_list=["C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Financial Data\\h_us_txt\\data\\hourly\\us\\nasdaq stocks\\1\\"], 
        index_cols_list = index_cols_list, 
        input_cols_to_include_list=input_cols_to_include_list)
    
    df_financial_data = populate_technical_indicators_2(df_financial_data, fin_indi)
    
    X_train, y_train, X_test, y_test, nan_values_replaced = create_step_responces_and_split_training_test_set(
        df_financial_data=df_financial_data, 
        pred_output_and_tickers_combos_list = pred_output_and_tickers_combos_list,
        pred_steps_list=pred_steps_list,
        train_test_split=train_test_split)
    
    prepped_fin_input = X_train, y_train, X_test, y_test
    
    return model_types_and_params_dict, prepped_fin_input

def run_DoE_return_models_and_result(DoE_orders_dict, prepped_fin_input):
    print("Hello")


    return results_dict, models_dict


#%% Prep Stock Market Data

"""import_data methods"""
def import_financial_data(
        target_folder_path_list=["C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Financial Data\\h_us_txt\\data\\hourly\\us\\nasdaq stocks\\1\\"], 
        index_cols_list = index_cols_list, 
        input_cols_to_include_list=input_cols_to_include_list):
    
    df_financial_data = pd.DataFrame()
    for folder in target_folder_path_list:
        if os.path.isdir(folder) == True:
            initial_list = os.listdir(folder)
            for file in os.listdir(folder):
                #extract values from file
                df_temp = pd.read_csv(folder + file, parse_dates=True)
                if len(input_cols_to_include_list)==2:
                    df_temp[index_col_str] = df_temp[index_cols_list[0]].astype(str) + "_" + df_temp[index_cols_list[1]].astype(str)
                    df_temp = df_temp.set_index(index_col_str)
                elif len(input_cols_to_include_list)==1:
                    df_temp = df_temp.set_index(index_col_str)
                    
                                        
                df_temp = df_temp[input_cols_to_include_list]
                
                if initial_list[0] == file:
                    df_financial_data   = copy.deepcopy(df_temp)
                else:
                    df_financial_data   = pd.concat([df_financial_data, df_temp], axis=1, ignore_index=False)
                col_rename_dict = dict()
                for col in input_cols_to_include_list:
                    col_rename_dict[col] = return_ticker_code_1(file) + "_" + col
                df_financial_data = df_financial_data.rename(columns=col_rename_dict)
                
                del df_temp
                
    return df_financial_data

def populate_technical_indicators_2(df_financial_data, technicial_indicators_to_add_list):
    #FG_Actions: to populate method
    return df_financial_data

#this method populates each row with the next X output results, this is done so that, each time step can be trained
#to predict the value of the next X steps
def create_step_responces_and_split_training_test_set(
        df_financial_data=df_financial_data, 
        pred_output_and_tickers_combos_list = pred_output_and_tickers_combos_list,
        pred_steps_list=pred_steps_list,
        train_test_split=train_test_split):
    
    new_col_str = "{}_{}_{}"
    old_col_str = "{}_{}"
    list_of_new_columns = []
    nan_values_replaced = 0
    
    df_financial_data, nan_values_replaced = current_infer_values_method(df_financial_data)
    
    #create regressors
    for combo in pred_output_and_tickers_combos_list:
        for step in pred_steps_list:
            list_of_new_columns = list_of_new_columns + [new_col_str.format(combo[0], combo[1], step)]
            df_financial_data[new_col_str.format(combo[0], combo[1], step)] = df_financial_data[old_col_str.format(combo[0], combo[1])].shift(-step)

    #split regressors and responses
    #Features = 6

    df_financial_data = df_financial_data[:-max(pred_steps_list)]

    X = copy.deepcopy(df_financial_data)
    y = copy.deepcopy(df_financial_data[list_of_new_columns])
    
    for col in list_of_new_columns:
        X = X.drop(col, axis=1)
    
    split = int(len(df_financial_data) * train_test_split)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test, nan_values_replaced


#%% Main Line Script


DoE_orders_dict, prepped_fin_input = define_DoE("initial test DoE", model_types_and_params_dict, target_folder_path_list, fin_indi, pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, train_test_split=train_test_split)
results_dict, models_dict = run_DoE_return_models_and_result(DoE_orders_dict, prepped_fin_input)


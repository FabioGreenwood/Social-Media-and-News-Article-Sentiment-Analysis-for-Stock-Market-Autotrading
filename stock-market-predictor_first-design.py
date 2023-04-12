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
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import copy
import datetime
import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import itertools as it


def return_conbinations_or_lists(list_a, list_b):
    unique_combinations = []
    permut = it.permutations(list_a, len(list_b))
    for comb in permut:
        zipped = zip(comb, list_b)
        unique_combinations.append(list(zipped))
    return unique_combinations


#DoE params


NN_hidden_layer_sizes_strat = [(100,), (200,), (100,10), (200,20)]
NN_activation_strat         = ['relu', 'logistic']
model_types_and_params_dict = {
    "overall strat" : {
        "param search full" : True,
        "predict on only best params post CV analysis" : True
    },
    "ElasticNet" : { #Linear regression with combined L1 and L2 priors as regularizer.
        'estimator__alpha':[0.1, 0.5, 0.9], 
        'estimator__l1_ratio':[0.1, 0.5, 0.9]
    },
    "MLPRegressor" : { #Multi-layer Perceptron regressor
        "estimator__hidden_layer_sizes"    : NN_hidden_layer_sizes_strat, 
        "estimator__activation"            : NN_activation_strat}
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
df_financial_data           = np.nan
bscv                        = np.nan
EXAMPLE_model_params_sweep  = np.nan
params_grid                 = np.nan
input_grid                  = np.nan

#%% Misc Methods and Classes

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


    



class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


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
"""XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"""
def run_DoE_return_models_and_result(DoE_orders_dict, prepped_fin_input,
                                     pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list):
    print("Hello")
    X_train, y_train, X_test, y_test = prepped_fin_input
    results_dict = dict()
    models_dict = dict()
    models_list = list(DoE_orders_dict)
    models_list.remove("overall strat")
    
    
    
    for key in models_list:
        params_sweep=DoE_orders_dict[key]
        #define time blocking
        btscv                               = BlockingTimeSeriesSplit(n_splits=time_series_split_qty)
        complete_record                     = return_CV_analysis_scores(X_train, y_train, CV_Reps=CV_Reps, model_str=key, cv=btscv, cores_used=4, params_grid=params_sweep, pred_steps_list=pred_steps_list)
        best_params_fg                      = return_best_model_paramemeters(complete_record=complete_record, params_sweep=params_sweep)
        print("Hello")
        #"""Train Model"""
        #models_dict, preds                  = return_models_and_preds(X_train = X_train, y_train = y_train, X_test = X_test, best_params=best_params_fg, pred_output_and_tickers_combos_list=pred_output_and_tickers_combos_list)
        #"""Report Results"""
        #df_realigned_dict                   = return_realign_plus_minus_table(preds, X_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True)
        #fig                                 = visualiser_plus_minus_results(df_realigned_dict ,preds, pred_steps_list, range_to_show=range(0,20,2), output_name="Test", outputs_folder_path = ".//outputs//", figure_name = "test_output")
        #results_X_day_plus_minus_accuracy   = return_df_X_day_plus_minus_accuracy(df_realigned_dict, X_test, pred_steps_list, pred_output_and_tickers_
    


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

#%% Model Training - Programmatically Define Model
def check_dict_keys_for_build_model(keys, dict, type_str):
    for key in keys:
        if not key in list(dict.keys()):
            raise ValueError("Key: " + key + " missing from model_types_and_params_dict[" + type_str  + "], dict must have the folliwing keys: " + str(keys))
    
def build_model(type_str, input_dict=None):
    
    match type_str:
        case "ElasticNet":
            
            keys = ["estimator__alpha", "estimator__l1_ratio"]
            check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = ElasticNet(
                alpha   =input_dict["estimator__alpha"],
                l1_ratio=input_dict["estimator__l1_ratio"],
                fit_intercept=True,
                #normalize=False,
                precompute=False,
                max_iter=16,
                copy_X=True,
                tol=0.1,
                warm_start=False,
                positive=False,
                random_state=None,
                selection='random'
            )
        case "MLPRegressor":
            
            keys = ["estimator__hidden_layer_sizes", "estimator__activation"]
            check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = MLPRegressor(
                activation=input_dict["estimator__activation"],
                hidden_layer_sizes=input_dict["estimator__hidden_layer_sizes"],
                alpha=0.001,
                random_state=20,
                early_stopping=False
            )
        case _:
            raise ValueError("the model type: " + type_str + " was not found in the method")
    
    return MultiOutputRegressor(estimator, n_jobs=4)
 


#%% Model Training - CV Analysis
#GridSearchCV works by exhaustively searching all the possible combinations of the model’s parameters

def return_CV_analysis_scores(X_train, y_train, CV_Reps=CV_Reps, cv=bscv, cores_used=4, model_str="ElasticNet",
                              params_grid=params_grid, pred_steps_list=pred_steps_list, 
                              pred_output_and_tickers_combos_list=pred_output_and_tickers_combos_list, 
                              input_grid=input_grid
                              ):
    global model_types_and_params_dict
    list_models = list(model_types_and_params_dict)
    #initialise 
    scores = []
    params_ = []
    complete_record = dict()
    for pred_step in pred_steps_list:
        complete_record[pred_step] = dict()
        
    #main method
    #prep only outputs for a single number of time steps
    for pred_step in pred_steps_list:
        
        
        y_temp = return_df_filtered_for_timestep(y_train, pred_step)
        
        #run study for that number of time steps
        for i in range(CV_Reps):
            input = return_default_values_of_param_dict(params_grid)
            with warnings.catch_warnings():
                
                
                model = build_model(model_str, input)
        
                finder = GridSearchCV(
                    estimator=model,
                    param_grid=params_grid,
                    scoring='r2',
                    n_jobs=4,
                    #iid=False,
                    refit=True,
                    cv=cv,  # change this to the splitter subject to test
                    verbose=0,
                    pre_dispatch=8,
                    error_score=-999,
                    return_train_score=True
                    )
                
                finder.fit(X_train, y_temp)
                
            #warnings.filterwarnings("default", category=ConvergenceWarning)
            
            #manually store stats
            for para, score in zip(finder.cv_results_["params"],finder.cv_results_["mean_test_score"]):
                a, b = para.values()
                if not (a, b) in complete_record[pred_step].keys():
                    complete_record[pred_step][(a, b)] = []
                complete_record[pred_step][(a, b)] = complete_record[pred_step][(a, b)] + [score]
            
            #complete_record[pred_step]["scores"] =  complete_record[pred_step]["scores"] + list(finder.cv_results_["mean_score_time"])
            #for para_name in  params_sweep.keys():
            #    complete_record[pred_step][para_name] = complete_record[pred_step][para_name] + list(finder.cv_results_["param_" + para_name])
    
            #preint progress
            
            
            best_params = finder.best_params_
            params_ = params_ + [best_params]
            best_score = round(finder.best_score_,4)
            last_score = round(finder.best_score_,4)
            scores.append(best_score)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        print("CV Analysis completed for pred step " + str(pred_step))
        print(str(pred_steps_list))
        print(model_str)
        print(list_models.index(model_str))
        print(str(pred_steps_list.index(pred_step)+1) + "/" + str(len(pred_steps_list)))
        
    return complete_record

def return_df_filtered_for_timestep(df, step):
    cols_to_keep_temp, cols_to_del_temp = [], []
    output_col_str = "{}_{}_{}"

    for ticker, value in pred_output_and_tickers_combos_list:
        cols_to_keep_temp = cols_to_keep_temp + [output_col_str.format(ticker, value, step)]
    
    for temp_col in df.columns:
        if not temp_col in cols_to_keep_temp:
            cols_to_del_temp = cols_to_del_temp + [temp_col]
    
    df = copy.deepcopy(df)
    df = df.drop(cols_to_del_temp, axis=1)
    return df
    
def return_best_model_paramemeters(complete_record, params_sweep):
        output = dict()
        average_scores_record   = copy.deepcopy(complete_record)
        steps_list              = list(complete_record.keys())
        variables_list          = list(params_sweep.keys())
        output["params order"]  = list(params_sweep.keys())
        
        for steps in steps_list:
            for index in average_scores_record[steps]:
                average_scores_record[steps][index] = sum(average_scores_record[steps][index]) / len(average_scores_record[steps][index])
            output[steps] = max(average_scores_record[steps], key=average_scores_record[steps].get)
        
        return output
    
def return_default_values_of_param_dict(page):
    output = dict()
    for key in page:
        output[key] = page[key][0]
    return output





#%% Main Line Script


DoE_orders_dict, prepped_fin_input  = define_DoE("initial test DoE", model_types_and_params_dict, target_folder_path_list, fin_indi, pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, train_test_split=train_test_split)
results_dict, models_dict           = run_DoE_return_models_and_result(DoE_orders_dict, 
                                                                       prepped_fin_input, 
                                                                       pred_output_and_tickers_combos_list, 
                                                                       pred_steps_list=pred_steps_list)


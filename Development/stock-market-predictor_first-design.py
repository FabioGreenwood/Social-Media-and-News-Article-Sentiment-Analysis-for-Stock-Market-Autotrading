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
import fnmatch
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
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import copy
from datetime import datetime
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import itertools as it

#misc methods
def return_conbinations_or_lists(list_a, list_b):
    unique_combinations = []
    permut = it.permutations(list_a, len(list_b))
    for comb in permut:
        zipped = zip(comb, list_b)
        unique_combinations.append(list(zipped))
    return unique_combinations

def return_conbinations_or_lists_fg(list_a,list_b):
    combined_lists = []
    for a in list_a:
        for b in list_b:
            if isinstance(a, list):
                combined_lists = combined_lists + [a + [b]]
            else:
                combined_lists = combined_lists + [[a, b]]
                
    return combined_lists

#DoE params
NN_hidden_layer_sizes_strat = [(100,), (200,), (100,10), (200,20)]
NN_activation_strat         = ['relu', 'logistic']
standard_intput_cohort_retention_rate_dict = {
            "Output_indicator" : 1,
            "Sentiment": 0.5,
            "Technical" : 0.5,
            "*" : 0.1}
model_types_and_params_dict = {
    "name" : "Test_DoE",
    "overall strat" : {
        "param search full" : True,
        "predict on only best params post CV analysis" : True
    },
    "RandomSubspace" : {
        "training_time_splits" : [10],
        "n_estimators"         : [3],
        "max_depth"            : [2],
        "max_features"         : [1.0],
        "random_state"         : [42],
        "cohort_retention_rate_dict" : standard_intput_cohort_retention_rate_dict
    },
    "ElasticNet" : { #Linear regression with combined L1 and L2 priors as regularizer.
        'estimator__alpha':[0.1, 0.5, 0.9], 
        'estimator__l1_ratio':[0.1, 0.5, 0.9], 
        'n_estimators' : [10, 20],
        'max_samples' : [1.0],
        'max_features' : [0.5],
        "cohort_retention_rate_dict" : standard_intput_cohort_retention_rate_dict,
    },
        "MLPRegressor" : { #Multi-layer Perceptron regressor
        "estimator__hidden_layer_sizes"    : NN_hidden_layer_sizes_strat, 
        "estimator__activation"            : NN_activation_strat,
        "cohort_retention_rate_dict" : standard_intput_cohort_retention_rate_dict}
}

#stratergy params
Features = 6
pred_output_and_tickers_combos_list = [("aapl", "<CLOSE>"), ("carm", "<HIGH>")]
pred_steps_list                     = [1,2,5,10]
time_series_split_qty               = 5
training_error_measure_main         = 'neg_mean_squared_error'
train_test_split                    = 0.7 #the ratio of the time series used for training
CV_Reps                             = 2
time_step_notation_sting            = "d" #day here is for a day, update as needed??????
fin_indi                            = [] #financial indicators
tweet_ratio_removed                 = int(5e2)
topic_qty                           = 7


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
PLACEHOLDER_best_params_fg  = np.nan
pred_col_name_str           = np.nan

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

def define_DoE(model_types_and_params_dict, fin_data_input, fin_indi, pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, train_test_split=train_test_split):
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
                                     pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, parent_output_dir=".\\outputs\\"):
    print("Hello")
    X_train, y_train, X_test, y_test = prepped_fin_input
    results_dict = dict()
    models_dict = dict()
    models_list = list(DoE_orders_dict)
    models_list.remove("overall strat")
    models_list.remove("name")
    
    
    #create folders for outputs
    now = datetime.now()
    #model_start_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    model_start_time = now.strftime("%Y%m%d_%H%M")
    output_parent_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\outputs"
    outputs_path = os.path.join(output_parent_path, model_start_time + "_" + DoE_orders_dict["name"])# + "\\")
    if not os.path.isdir(outputs_path):
        os.mkdir(outputs_path) 
    #create subfolders
        for model_type in models_list:
            subfolder_path = os.path.join(outputs_path, model_type)
            os.mkdir(subfolder_path)
        
    for key in models_list:
        params_sweep=DoE_orders_dict[key]
        
        #define time blocking
        btscv                               = BlockingTimeSeriesSplit(n_splits=time_series_split_qty)
        complete_record                     = return_CV_analysis_scores(X_train, y_train, X_test, y_test, CV_Reps=CV_Reps, model_str=key, cv=btscv, cores_used=4, params_grid=params_sweep, pred_steps_list=pred_steps_list)
        best_params_fg                      = return_best_model_paramemeters(complete_record=complete_record, params_sweep=params_sweep)
        
        #"""Train Model"""
        models_dict, preds                  = return_models_and_preds(X_train = X_train, y_train = y_train, X_test = X_test, model_str=key, best_params=best_params_fg, pred_output_and_tickers_combos_list=pred_output_and_tickers_combos_list)        
        
        #"""Calc Results"""
        df_realigned_dict                   = return_realign_plus_minus_table(preds, X_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True)
        results_tables_dict                 = return_results_X_day_plus_minus_accuracy( df_realigned_dict, X_test, pred_steps_list, pred_output_and_tickers_combos_list, confidences_before_betting_PC=[0, 0.01], model_type=key, model_start_time = model_start_time, output_name="Test2", outputs_folder_path = outputs_path, figure_name = "test_output2", pred_col_name_str=pred_col_name_str)
        #results_x_day_plus_minus_PC, results_x_day_plus_minus_PC_confindence, results_x_day_plus_minus_score_confidence, results_x_day_plus_minus_score_confidence_weighted,         
        
        plt, df_realigned_dict              = return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = DoE_orders_dict["name"], model_type=key, model_start_time = model_start_time, outputs_folder_path = outputs_path, timestamp = False)
        
        
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

#%% Prep Sentimental Data
"""import_data methods"""
def import_sentimental_data(

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




#%% Model Training - Programmatically Define Model
def check_dict_keys_for_build_model(keys, dict, type_str):
    for key in keys:
        if not key in list(dict) and not key.replace("estimator__","") in list(dict):
            raise ValueError("Key: " + key + " missing from model_types_and_params_dict[" + type_str  + "], dict must have the folliwing keys: " + str(keys))
    
def return_columns_to_remove(columns_list, self):
    
    columns_to_remove = list(copy.deepcopy(columns_list))
    retain_cols = []
    retain_dict = self.input_dict["cohort_retention_rate_dict"]
    max_features = self.max_features
    stock_strings_list = []
    columns_list = list(columns_list)
    for a in self.ticker_name:
        stock_strings_list = stock_strings_list + [a[0]]
    for key in retain_dict:
        cohort = []
        target_string = key
        if len(fnmatch.filter([key], "STOCK_NAME*")) > 0:
            for stock in stock_strings_list:
                target_string = key
                target_string = target_string.replace("STOCK_NAME", stock)
                cohort = cohort + fnmatch.filter(columns_list, target_string)
        else:
            cohort = cohort + fnmatch.filter(columns_list, target_string)
        if len(cohort) > 0:
            for col in cohort:
                columns_list.remove(col)
            retain_cols = retain_cols + list(np.random.choice(cohort, int(len(cohort) * retain_dict[key]), replace=False))

    for value in retain_cols:
        columns_to_remove.remove(value)
    
    return columns_to_remove

class DRSLinReg():
    def __init__(self, base_estimator=LinearRegression(),
                 input_dict=dict(),
                 ticker_name=[]):
        #expected keys: training_time_splits, max_depth, max_features, random_state,        
        for key in input_dict:
           setattr(self, key, input_dict[key])
        self.input_dict = input_dict
        self.ticker_name = ticker_name
        self.base_estimator = base_estimator
        self.estimators_ = []
        
    def fit(self, X, y):
        tscv = BlockingTimeSeriesSplit(n_splits=self.training_time_splits)
        count = 0
        for train_index, _ in tscv.split(X):
            #these are the base values that will be updated if there isn't a passed value in the input dict
            estimator = BaggingRegressor(base_estimator=self.base_estimator,
                                          #the assignment of "one" estimator is overwritten by the rest of the method
                                          n_estimators=1,
                                          max_samples=1.0,
                                          max_features=1.0,
                                          bootstrap=True,
                                          bootstrap_features=False,
                                          random_state=self.random_state)
            for key in self.input_dict:
                setattr(estimator, key, self.input_dict[key])
            
            estimator.base_estimator = self.base_estimator
            
            count += 1
            for i_random in range(self.n_estimators):
                # randomly select features to drop out
                n_features = X.shape[1]
                dropout_cols = return_columns_to_remove(columns_list=X.columns, self=self)
                #dropout_cols = np.random.choice(n_features, size=int(n_features * (1 - self.max_features)), replace=False)
                #dropout_cols = X.columns[dropout_cols]
                X_sel = X.loc[X.index[train_index].values].copy()
                X_sel.loc[:, dropout_cols] = 0
                y_sel= y.loc[y.index[train_index].values].copy()
                #print(str(i_random) + "/" + str(self.n_estimators) + "--" + str(count) + "/" + str(self.training_time_splits))
                
            # add layers to the estimator
                for j in range(self.max_depth):
                    
                    estimator.fit(X_sel, y_sel)
                    self.estimators_ = self.estimators_ + [estimator]
            
            #self.estimators_.append(estimator)
        return self

    def predict_ensemble(self, X, name_of_outputs=None):
        
        if name_of_outputs==None:
            outputs_range = range(self.estimators_[0].predict(X).shape[1])
        else:
            outputs_range = name_of_outputs
        output = pd.DataFrame(columns=outputs_range)

        y_pred = []
        for j in range(len(outputs_range)):
            y_pred = y_pred + [np.zeros((X.shape[0], len(self.estimators_)))]
            
        for i, estimator in enumerate(self.estimators_):
            # randomly select features to drop out
            y_temp = estimator.predict(X)
            if not len(outputs_range) == y_temp.shape[1]:
                raise ValueError("the number of values in name_of_outputs must equals the number of outputs generated, please check this entry matches the outputs for the trained model")
            
            for j in range(len(outputs_range)):
                y_pred[j][:,i] = y_temp[:,j]
            
        
        for j in range(len(outputs_range)):
            output[outputs_range[j]] = np.mean(y_pred[j], axis=1)
        
        return output
    
    def evaluate(self, X_test=None, y_test=None, y_pred=None, method="r2", return_high_good=False):
        if y_pred==None:
            y_pred = self.predict_ensemble(X_test).values
        
        match method:
            case "r2":
                output = r2_score(y_test, y_pred)
                high_good = True
            case "mse":
                output = mean_squared_error(y_test, y_pred)
                high_good = False
            case "mae":
                output = mean_absolute_error(y_test, y_pred)
                high_good = False
            case _:
                raise ValueError("passed method string not found")
        
        if return_high_good==False:
            return output
        else:
            return output, high_good
        

def build_model2(type_str, input_dict=None, ticker_name=None):
    
    match type_str:
        case "RandomSubspace":
            keys = ["estimator__alpha", "estimator__l1_ratio", "enforced_features", "proportion_of_random_features"]
            estimator = DRSLinReg(base_estimator=LinearRegression(),
                      input_dict=input_dict, ticker_name=ticker_name)
            
        
        case "ElasticNet":
            keys = ["estimator__alpha", "estimator__l1_ratio"]
            #check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = BaggingRegressor(
                MultiOutputRegressor(
                ElasticNet(
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
                )), n_estimators=10, random_state=0, max_features=0.5
            )
        case "MLPRegressor":
            keys = ["estimator__hidden_layer_sizes", "estimator__activation"]
            #check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = MLPRegressor(
                MultiOutputRegressor(
                activation=input_dict["estimator__activation"],
                hidden_layer_sizes=input_dict["estimator__hidden_layer_sizes"],
                alpha=0.001,
                random_state=20,
                early_stopping=False)
            )
        case _:
            raise ValueError("the model type: " + type_str + " was not found in the method")
    
    return estimator



#%% Model Training - CV Analysis
#GridSearchCV works by exhaustively searching all the possible combinations of the model’s parameters

def return_CV_analysis_scores(X_train, y_train, X_test, y_test, CV_Reps=CV_Reps, cv=bscv, cores_used=4, model_str="ElasticNet",
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
    complete_record["params order"] = []
    cohort_retention_rate_dict = params_grid.pop("cohort_retention_rate_dict")
    for pred_step in pred_steps_list:
        complete_record[pred_step] = dict()
    params_keys = list(params_grid.keys())
    complete_record["params order"] = params_keys
    params_sweep = params_grid[params_keys[0]]
    for i in range(1, len(params_keys)):
        params_sweep = return_conbinations_or_lists_fg(params_sweep, params_grid[params_keys[i]])
    #main method
    #prep only outputs for a single number of time steps
    
    for pred_step in pred_steps_list:
        y_train_temp = return_df_filtered_for_timestep(y_train, pred_step)
        y_test_temp  = return_df_filtered_for_timestep(y_test, pred_step)
        #run study for that number of time steps
        for params in params_sweep:
            #input = return_default_values_of_param_dict(params_grid)
            
            with warnings.catch_warnings():    
                input_dict = dict()
                for key, i in zip(params_keys, range(len(params_keys))):
                    input_dict[key] = params[i]
                input_dict["cohort_retention_rate_dict"] = cohort_retention_rate_dict
                model = build_model2(model_str, input_dict, ticker_name=pred_output_and_tickers_combos_list)
                #print(input_dict)
                model.fit(X_train, y_train_temp)
                score = model.evaluate(X_test=X_test , y_test=y_test_temp)
            
            
            
            
            #manually store stats
            if not tuple(params) in complete_record[pred_step].keys():
                complete_record[pred_step][tuple(params)] = []
            complete_record[pred_step][tuple(params)] = complete_record[pred_step][tuple(params)] + [score]
            
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(datetime.now().strftime("%H:%M:%S"))
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
        steps_list.remove("params order")
        variables_list          = list(params_sweep.keys())
        output["params order"]  = complete_record["params order"]
        
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


#%% Model Training - Model Training and tickers

def return_models_and_preds(X_train, y_train, X_test, model_str="ElasticNet", best_params=PLACEHOLDER_best_params_fg, pred_output_and_tickers_combos_list=pred_output_and_tickers_combos_list):
    preds = pd.DataFrame()
    output_col_str = "{}_{}_{}"
    preds_dict = dict()
    models_dict= dict()
    preds_steps_dict = dict()
    steps_list = list(best_params.keys())
    steps_list.remove("params order")
    remove_from_param_str  = "estimator__"
    params_long_names_list = list(best_params["params order"])
    
    
        #models_dict[(ticker, output)]
        #preds_steps_dict[(ticker, output)]
    for step in steps_list:
        input_dict = dict()
        preds_steps_dict[step] = dict()
        #create best model param values
        for param_name, i in zip(params_long_names_list, range(len(params_long_names_list))):
            param_short_name_str = param_name.replace(remove_from_param_str, "")
            input_dict[param_short_name_str] = best_params[step][i]
        #prep columns
        column_str_list = []
        for ticker, value in pred_output_and_tickers_combos_list:
            column_str_list = column_str_list + [output_col_str.format(ticker, value ,step)]
            y_temp = y_train[column_str_list]
        input_params = dict()
        for key, i2 in zip(best_params["params order"], range(len(best_params["params order"]))):
            input_params[key] = best_params[step][i2]
        
        model_temp        = build_model2(model_str, input_params)
        print(model_str)
        model_temp        = model_temp.fit(X_train, y_temp)
        models_dict[step] = model_temp
        preds_temp = model_temp.predict_ensemble(X_test)
        preds = pd.concat([preds, pd.DataFrame(preds_temp, index=X_test.index, columns=y_temp.columns)], axis=1)
            
        
    
    return models_dict, preds

#%% Model Testing and Reporting

def return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = "", model_type="", model_start_time = "", outputs_folder_path = ".//outputs//tables//", timestamp = False):
    
    outputs_folder_path = outputs_folder_path + "\\" + model_type + "\\"
    single_levelled_tables = ["results_x_day_plus_minus_PC"]
    double_levelled_tables = ["results_x_day_plus_minus_PC_confindence", "results_x_day_plus_minus_score_confidence", "results_x_day_plus_minus_score_confidence_weighted"]
    fig_i = 0
    
    for table_name in single_levelled_tables:
        fig = plt.figure(fig_i)
        figure = plt.figure(fig_i)
        target_dict = results_tables_dict[table_name]
        table = pd.DataFrame.from_dict(target_dict, orient="columns")
        fig = sns.heatmap(table, cmap="vlag_r", annot=True, vmin=0, vmax=1)
        fig.set_title(table_name)
        fig.get_figure()
        figure.tight_layout()
        figure.savefig(outputs_folder_path+model_type+"_"+table_name+".png")
        table.to_csv(outputs_folder_path+model_type+"_"+table_name+".csv")
        fig_i += 1
           
    #prep info for double layered lists
    tickers_list    = list(results_tables_dict  [double_levelled_tables[0]].keys())
    steps_list      = list(results_tables_dict  [double_levelled_tables[0]][tickers_list[0]].keys())
    confidences_list = list(results_tables_dict [double_levelled_tables[0]][tickers_list[0]][steps_list[0]].keys())
    tickers_confidences_combos = []
    for con in confidences_list:
        for tic in tickers_list:
            tickers_confidences_combos = tickers_confidences_combos + [str(tic) + "_" + str(con)]
    
    #print double layered list
    for table_name in double_levelled_tables:
        #del fig, figure, new_table, target_dict, new_dict
        new_dict = dict()
        target_dict = results_tables_dict[table_name]
        new_table = pd.DataFrame(columns=tickers_confidences_combos)
        for step in steps_list:
            new_dict[step] = dict()
            for con in confidences_list:
                for tic in tickers_list:
                    new_dict[step][(tic, con)] = target_dict[tic][step][con]
                    #new_table.iloc[str(tic) + "_" + str(con), step] = target_dict[tic][step][con]
                    #new_table.loc[step, str(tic) + "_" + str(con)] = target_dict[tic][step][con]
                    tt = pd.DataFrame.from_dict(new_dict, orient="columns")
                    
        fig = plt.figure(fig_i)
        figure = plt.figure(fig_i)
        new_table = pd.DataFrame.from_dict(new_dict, orient="columns")
        if "PC" in table_name:
            fig = sns.heatmap(tt, cmap="vlag_r", annot=True, vmin=0, vmax=1)
        else:
            fig = sns.heatmap(tt, cmap="vlag_r", annot=True, vmin=-4, vmax=4)
        fig.set_title(table_name)
        figure = fig.get_figure()
        figure.tight_layout()
        figure.savefig(outputs_folder_path+model_type+"_"+table_name+".png")
        new_table.to_csv(outputs_folder_path+model_type+"_"+table_name+".csv")
        fig_i += 1
        del fig, figure, new_table, target_dict, new_dict
        
    results_x_day_plus_minus_PC                         = results_tables_dict["results_x_day_plus_minus_PC"]
    results_x_day_plus_minus_PC_confindence             = results_tables_dict["results_x_day_plus_minus_PC_confindence"]
    results_x_day_plus_minus_score_confidence           = results_tables_dict["results_x_day_plus_minus_score_confidence"]
    results_x_day_plus_minus_score_confidence_weighted  = results_tables_dict["results_x_day_plus_minus_score_confidence_weighted"]

    return plt, df_realigned_dict


def return_realign_plus_minus_table(preds, X_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True):
    input_col_str = "{}_{}"
    output_col_str = "{}_{}_{}"
    reaglined_plus_minus_dict = dict()
    for ticker, output in pred_output_and_tickers_combos_list:
        #initial output
        input_col_str_curr = input_col_str.format(ticker, output)
        df_temp = pd.DataFrame(X_test[input_col_str_curr])
        #pop the preditions
        output_col_str_2 = output_col_str.format(ticker, output, {})
    
        for step_backs in pred_steps_list:
            df_temp[output_col_str_2.format(step_backs)+"_before"] = preds[output_col_str_2.format(step_backs)].shift(step_backs)
            if make_relative == True:
                df_temp[output_col_str_2.format(step_backs)+"_before"] -= df_temp[input_col_str_curr]
        
        reaglined_plus_minus_dict[(ticker, output)] = df_temp
        df_temp.to_csv("C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\" +"temp" + ".csv")
        preds.to_csv("C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\" +"preds" + ".csv")
    
    return reaglined_plus_minus_dict


def return_results_X_day_plus_minus_accuracy(df_temp_dict, X_test, pred_steps_list, pred_output_and_tickers_combos_list, confidences_before_betting_PC=[0, 0.01], model_type="", save=True, model_start_time = "", output_name="Test2", outputs_folder_path = ".//outputs//", figure_name = "test_output2", pred_col_name_str=pred_col_name_str):
    
    
    #df_realigned_temp = copy.deepcopy(df_realigned)
    pred_str = "{}_{}_{}_before"
    record_str = "{}_{}"
    
    results_x_day_plus_minus_PC                         = dict() 
    results_x_day_plus_minus_PC_confindence             = dict()
    results_x_day_plus_minus_score_confidence           = dict() 
    results_x_day_plus_minus_score_confidence_weighted  = dict() 
    
    count_bets_with_confidence               = dict()
    count_correct_bets_with_confidence       = dict()
    count_correct_bets_with_confidence_score = dict()
    count_correct_bets_with_confidence_score_weight = dict()
    count_correct_bets_with_confidence_score_weight_total = dict()
    
    for ticker in list(df_temp_dict.keys()):
        
        df_temp = df_temp_dict[ticker]
        results_x_day_plus_minus_PC                       [ticker]  = dict()
        results_x_day_plus_minus_PC_confindence           [ticker]  = dict()
        results_x_day_plus_minus_score_confidence         [ticker]  = dict()
        results_x_day_plus_minus_score_confidence_weighted[ticker]  = dict()
        
        count_bets_with_confidence[ticker]               = dict()
        count_correct_bets_with_confidence[ticker]       = dict()
        count_correct_bets_with_confidence_score[ticker] = dict()
        count_correct_bets_with_confidence_score_weight[ticker] = dict()
        count_correct_bets_with_confidence_score_weight_total[ticker] = dict()
        
        time_series_index = X_test.index
        
        for steps_back in pred_steps_list:
            #initialise variables
            
            col_name = df_temp.columns[pred_steps_list.index(steps_back)]
            count           = 0
            count_correct   = 0
            count_bets_with_confidence              [ticker][steps_back] = dict()
            count_correct_bets_with_confidence      [ticker][steps_back] = dict()
            count_correct_bets_with_confidence_score[ticker][steps_back] = dict()
            count_correct_bets_with_confidence_score_weight[ticker][steps_back] = dict()
            count_correct_bets_with_confidence_score_weight_total[ticker][steps_back] = dict()
            
            results_x_day_plus_minus_PC                     [ticker][steps_back] = dict()
            results_x_day_plus_minus_PC_confindence         [ticker][steps_back] = dict()
            results_x_day_plus_minus_score_confidence       [ticker][steps_back] = dict()
            results_x_day_plus_minus_score_confidence_weighted[ticker][steps_back] = dict()
            
            
            for confidence_threshold in confidences_before_betting_PC:
                count_bets_with_confidence              [ticker][steps_back][confidence_threshold] = 0
                count_correct_bets_with_confidence      [ticker][steps_back][confidence_threshold] = 0
                count_correct_bets_with_confidence_score[ticker][steps_back][confidence_threshold] = 0
                count_correct_bets_with_confidence_score_weight[ticker][steps_back][confidence_threshold] = 0
                count_correct_bets_with_confidence_score_weight_total[ticker][steps_back][confidence_threshold] = 0

            actual_values = X_test[record_str.format(ticker[0], ticker[1])]
            
            max_pred_steps = max(pred_steps_list)
            for row_num in range(max(pred_steps_list), len(X_test.index)):
                
                count += 1

                
                #basic values
                prediction_vector   = df_temp[pred_str.format(ticker[0], ticker[1], steps_back)]
                original_vaule      = actual_values[row_num - steps_back]
                expected_difference = prediction_vector[row_num]
                actual_difference   = actual_values[row_num] - actual_values[row_num - steps_back]
                relative_confidence = expected_difference / original_vaule
                
                #basic count scoring 
                if actual_difference * expected_difference > 0:
                    count_correct += 1
                
                #bets with confidence scoring
                for confidence_threshold in confidences_before_betting_PC:
                    if   abs(relative_confidence) > confidence_threshold and actual_difference * expected_difference > 0:
                        count_bets_with_confidence              [ticker][steps_back][confidence_threshold] += 1
                        count_correct_bets_with_confidence      [ticker][steps_back][confidence_threshold] += 1
                        count_correct_bets_with_confidence_score[ticker][steps_back][confidence_threshold] += abs(actual_difference)
                        count_correct_bets_with_confidence_score_weight[ticker][steps_back][confidence_threshold] += abs(actual_difference) * (abs(relative_confidence) - confidence_threshold)
                        count_correct_bets_with_confidence_score_weight_total[ticker][steps_back][confidence_threshold] += (abs(relative_confidence) - confidence_threshold)
                        
                    elif abs(relative_confidence) > confidence_threshold and actual_difference * expected_difference < 0:
                        count_bets_with_confidence              [ticker][steps_back][confidence_threshold] += 1
                        count_correct_bets_with_confidence      [ticker][steps_back][confidence_threshold] += 0
                        count_correct_bets_with_confidence_score[ticker][steps_back][confidence_threshold] -= abs(actual_difference)
                        count_correct_bets_with_confidence_score_weight[ticker][steps_back][confidence_threshold] -= abs(actual_difference) * (abs(relative_confidence) - confidence_threshold)
                        count_correct_bets_with_confidence_score_weight_total[ticker][steps_back][confidence_threshold] += (abs(relative_confidence) - confidence_threshold)
                        

            results_x_day_plus_minus_PC
            results_x_day_plus_minus_PC_confindence
            results_x_day_plus_minus_score_confidence
            results_x_day_plus_minus_score_confidence_weighted

            #Total scores for ticker steps_back conbination
            results_x_day_plus_minus_PC[ticker][steps_back] = count_correct / count
            for confidence_threshold in confidences_before_betting_PC:
                results_x_day_plus_minus_PC_confindence           [ticker][steps_back][confidence_threshold] = count_correct_bets_with_confidence             [ticker][steps_back][confidence_threshold] / count_bets_with_confidence[ticker][steps_back][confidence_threshold]
                results_x_day_plus_minus_score_confidence         [ticker][steps_back][confidence_threshold] = count_correct_bets_with_confidence_score       [ticker][steps_back][confidence_threshold] / count_bets_with_confidence[ticker][steps_back][confidence_threshold]
                results_x_day_plus_minus_score_confidence_weighted[ticker][steps_back][confidence_threshold] = count_correct_bets_with_confidence_score_weight[ticker][steps_back][confidence_threshold] / count_correct_bets_with_confidence_score_weight_total[ticker][steps_back][confidence_threshold]
        
    results_dict = dict()
    results_dict["results_x_day_plus_minus_PC"]                         = results_x_day_plus_minus_PC
    results_dict["results_x_day_plus_minus_PC_confindence"]             = results_x_day_plus_minus_PC_confindence
    results_dict["results_x_day_plus_minus_score_confidence"]           = results_x_day_plus_minus_score_confidence 
    results_dict["results_x_day_plus_minus_score_confidence_weighted"]  = results_x_day_plus_minus_score_confidence_weighted
    
    print("Hello")
    
    return results_dict


#%% Main Line Script

DoE_orders_dict, prepped_fin_input  = define_DoE(model_types_and_params_dict, target_folder_path_list, fin_indi, pred_output_and_tickers_combos_list, pred_steps_list=pred_steps_list, train_test_split=train_test_split)
results_dict, models_dict           = run_DoE_return_models_and_result(DoE_orders_dict, 
                                                                       prepped_fin_input, 
                                                                       pred_output_and_tickers_combos_list, 
                                                                       pred_steps_list=pred_steps_list)

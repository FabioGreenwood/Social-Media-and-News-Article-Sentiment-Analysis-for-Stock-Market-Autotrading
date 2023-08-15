"""
File: experiment_handler.py
Author: Fabio Greenwood
Created: 1st August 2023
Description: 


Required actions:
 - 
 - 
Dev notes:
 - 
 - 

"""

#%% Import Methods
import data_prep_and_model_training as FG_model_training
import additional_reporting_and_model_trading_runs as FG_additional_reporting
import GPyOpt
import numpy as np
import pandas as pd
import fnmatch
import pickle
import seaborn as sns 
import seaborn as sns
import copy
from datetime import datetime
from datetime import datetime, timedelta
import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

#%% Standard Parameters


#GLOBAL PARAMETERS
global_input_cols_to_include_list = ["<CLOSE>", "<HIGH>"]
global_index_cols_list = ["<DATE>","<TIME>"]
global_index_col_str = "datetime"
global_target_file_folder_path = ""
global_feature_qty = 6
global_outputs_folder_path = ".\\outputs\\"
global_financial_history_folder_path = "FG action, do I need to update this?"
global_df_stocks_list_file           = pd.read_csv(r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\data\support data\stock_info.csv")
global_start_time = datetime.now()
global_error_str_1 = "the input {} is wrong for the input training_or_testing"
global_random_state = 1
global_scores_database = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\outputs\scores_database.csv"
global_strptime_str = '%d/%m/%y %H:%M:%S'
global_strptime_str_filename = '%d_%m_%y %H:%M:%S'
global_precalculated_assets_locations_dict = {
    "root" : "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\",
    "topic_models"              : "topic_models\\",
    "annotated_tweets"          : "annotated_tweets\\",
    "predictive_model"          : "predictive_model\\",
    "sentimental_data"          : "sentimental_data\\",
    "experiment_records"         : "experiment_records\\"
    }
global_outputs_folder = "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\outputs\\"
global_designs_record_final_columns_list = ["experiment_timestamp", "training_r2", "training_mse", "training_mae", "testing_r2", "testing_mse", "testing_mae", "profitability", "predictor_names"]
SECS_IN_A_DAY = 60*60*24
SECS_IN_AN_HOUR = 60*60

#%% Default Input Parameters

default_temporal_params_dict    = {
    "train_period_start"    : datetime.strptime('04/06/18 00:00:00', global_strptime_str),
    "train_period_end"      : datetime.strptime('01/09/20 00:00:00', global_strptime_str),
    "time_step_seconds"     : 5*60, #5 mins,
    "test_period_start"     : datetime.strptime('01/09/20 00:00:00', global_strptime_str),
    "test_period_end"       : datetime.strptime('01/01/21 00:00:00', global_strptime_str),
}
default_fin_inputs_params_dict      = {
    "index_cols"        : "date",    
    "cols_list"         : ["open", "high", "low", "close", "volume"],
    "fin_indi"          : {#additional financial indicators to generate
        "sma" : [5, 15, 20],
        "ema" : [5, 15, 20]}, 
    "fin_match"         :{
        "Doji" : True},
    "index_col_str"     : "datetime",
    "historical_file"   : "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\data\\financial data\\tiingo\\aapl.csv",
}
default_senti_inputs_params_dict    = {
    "topic_qty"             : 7,
    "topic_training_tweet_ratio_removed" : int(1e5),
    "relative_lifetime"     : 60*60*24*7, # units are seconds
    "relative_halflife"     : 60*60*0.5, # units are seconds
    "topic_model_alpha"     : 1,
    "weighted_topics"       : False,
    "apply_IDF"             : True,
    "enforced_topics_dict_name" : "v1",
    "enforced_topics_dict"  : [
    ['investment', 'financing', 'losses'],
    ['risk', 'exposure', 'liability'],
    ["financial forces" , "growth", "interest rates"]],
    "sentiment_method"      : SentimentIntensityAnalyzer(),
    "tweet_file_location"   : r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\data\twitter data\Tweets about the Top Companies from 2015 to 2020\Tweet.csv\Tweet.csv"
}
default_outputs_params_dict         = {
    "output_symbol_indicators_tuple"    : ("aapl", "close"),
    "pred_steps_ahead"                  : 1,
}
default_cohort_retention_rate_dict = {
            "£_close" : 1, #output value
            "£_*": 1, #other OHLCV values
            "$_*" : 0.5, # technical indicators
            "match!_*" : 0.8, #pattern matchs
            "~senti_*" : 0.5, #sentiment analysis
            "*": 0.5} # other missed values
default_model_hyper_params          = {
    "name" : "RandomSubspace_MLPRegressor", #Multi-layer Perceptron regressor
        #general hyperparameters
    "n_estimators_per_time_series_blocking" : 2,
    "training_error_measure_main"   : 'neg_mean_squared_error',
    "testing_scoring"               : ["r2", "mse", "mae"],
    "time_series_blocking"          : "btscv",
    "time_series_split_qty"         : 5,
        #model specific rows
    "estimator__alpha"                 : 0.05,
    "estimator__hidden_layer_sizes"    : (100,10), 
    "estimator__activation"            : 'relu',
    "cohort_retention_rate_dict"       : default_cohort_retention_rate_dict}
default_input_dict = {
    "temporal_params_dict"      : default_temporal_params_dict,
    "fin_inputs_params_dict"    : default_fin_inputs_params_dict,
    "senti_inputs_params_dict"  : default_senti_inputs_params_dict,
    "outputs_params_dict"       : default_outputs_params_dict,
    "model_hyper_params"        : default_model_hyper_params
    }



#%% misc methods

def find_largest_number(mixed_list):
    numbers = [x for x in mixed_list if isinstance(x, (int, float))]
    
    if not numbers:
        return -1
    
    largest_number = max(numbers)
    return largest_number

def return_keys_within_2_level_dict(input_dict):
    output_list = []
    for key in input_dict:
        if type(input_dict[key]) == dict:
            for subkey in input_dict[key]:
                output_list = output_list + [key + "_" + subkey]
        else:
            output_list = output_list + [key]
    return output_list
    
def save_designs_record_csv_and_dict(records_path_list, df_designs_record=None, design_history_dict=None, optim_run_name=None):
    if not type(records_path_list) == list:
        records_path_list = [records_path_list]
    for path in records_path_list:
        file_path = os.path.join(path, optim_run_name)
        
        if type(df_designs_record) == pd.core.frame.DataFrame:
            try:
                df_designs_record.to_csv(file_path + ".csv", index=False)
                df_designs_record.to_csv(file_path + ".csvBACKUP", index=False)
            except:
                df_designs_record.to_csv(file_path + ".csvBACKUP", index=False)
                print("please close the csv")
        if design_history_dict != None:
            with open(file_path + ".py_dict", "wb") as file:
                pickle.dump(design_history_dict, file)

def update_df_designs_record(df_designs_record, design_history_dict, design_space_dict):
    global global_designs_record_final_columns_list
    
    if not all(df_designs_record.columns == return_keys_within_2_level_dict(design_space_dict) + global_designs_record_final_columns_list):
        raise ValueError("the schema for the design records table doesn't match, please review")
    
    input_param_cols = return_keys_within_2_level_dict(design_space_dict)
    
    for ID in range(find_largest_number(design_history_dict.keys())+1):
        df_designs_record.loc[ID, input_param_cols] = design_history_dict[ID]["X"]
        for subkey in design_history_dict[ID]:
            if not subkey in ["X", "predictor", "Y"]:
                df_designs_record.loc[ID, subkey] = design_history_dict[ID][subkey]
    return df_designs_record
        
        
#%% Experiment Parameters

DoE_1 = dict()

def return_list_of_experimental_params(experiment_instructions):
    list_of_experimental_parameters_names = []
    for key in experiment_instructions:
        for subkey in experiment_instructions[key]:
            name = key + "~" + subkey
            list_of_experimental_parameters_names = list_of_experimental_parameters_names + [name]
    return list_of_experimental_parameters_names

def convert_design_space_dict_to_GPyOpt_bounds_list(design_space_dict):
    output = []
    blank_parameter = {'name': None, 'type': None, 'domain': None}
    
    for key in design_space_dict:
        for subkey in design_space_dict[key]:
            name = key + "~" + subkey
            py_type = type(design_space_dict[key][subkey])
            value = design_space_dict[key][subkey]
            if py_type == dict:
                #type_str = "categorical" fg_placeholder
                type_str = "discrete"
                value = tuple(value.keys())
            elif py_type == list or py_type == range:
                type_str = "discrete"
                value = tuple(value)
            elif py_type == tuple:
                type_str = "continuous"
            else:
                raise ValueError("unrecognised input")
            output = output + [{'name': name, 'type': type_str, 'domain': value}]
    
    return output

def return_edited_input_dict(GPyOpt_input, exp_instr, default_input_dict):
    output_input_dict = copy.deepcopy(default_input_dict)
    input_index = 0
    for key in exp_instr:
        for subkey in exp_instr[key]:
            options = exp_instr[key][subkey]
            selection = GPyOpt_input[input_index]
            if type(options) == dict:
                output_input_dict[key][subkey] = options[selection]
            else:
                output_input_dict[key][subkey] = selection
            input_index += 1
    
    return output_input_dict
            
def template_experiment_requester(GPyOpt_input, design_space_dict, default_input_dict=default_input_dict, experiment_method=FG_model_training.retrieve_or_generate_model_and_training_scores):
    
    prepped_input_dict = return_edited_input_dict(GPyOpt_input, design_space_dict, default_input_dict)
    
    prepped_temporal_params_dict      = prepped_input_dict["temporal_params_dict"]
    prepped_fin_inputs_params_dict    = prepped_input_dict["fin_inputs_params_dict"]
    prepped_senti_inputs_params_dict  = prepped_input_dict["senti_inputs_params_dict"]
    prepped_outputs_params_dict       = prepped_input_dict["outputs_params_dict"]
    prepped_model_hyper_params        = prepped_input_dict["model_hyper_params"]
    
    if type(prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]) == str or type(prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]) == np.str_:
        original_value = prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]
        updated_val = []
        for x in original_value.split("_"):
            updated_val = updated_val + [x]
        prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"] = tuple(updated_val)
    
    global global_run_count
    
    print("xxxxxxxxxxxxxx" + str(global_run_count))
    
    global_run_count += 1
    
    predictor, training_scores = experiment_method(prepped_temporal_params_dict, prepped_fin_inputs_params_dict, prepped_senti_inputs_params_dict, prepped_outputs_params_dict, prepped_model_hyper_params)
    return predictor, training_scores
    
def return_experiment_requester(design_space_dict, default_input_dict=default_input_dict, experiment_method=FG_model_training.retrieve_or_generate_model_and_training_scores):
    output_method = lambda GPyOpt_input: template_experiment_requester(GPyOpt_input, design_space_dict, default_input_dict=default_input_dict, experiment_method=experiment_method)
    return output_method

def define_DoE(bounds, DoE_size):
    DoE = []
    for params in bounds:
        if params["type"] == "categorical" or params["type"] == "discrete":
            values_for_params = np.random.choice(params["domain"], size=(DoE_size, 1))
        elif params["type"] == "continuous":
            values_for_params = np.random.rand(iter, 1)
            lwr_lim, upr_lim = params["domain"]
            values_for_params = values_for_params * (upr_lim - lwr_lim) + lwr_lim
        else:
            raise ValueError("unrecognised param['type'] input")
        if bounds[0] == params:
            DoE = values_for_params
        else:
            DoE = np.hstack((DoE, values_for_params))
    return DoE

def return_X_and_Y_for_GPyOpt_optimisation(design_history_dict, opt_obj, inverse_for_minimise, objective_function_name="testing_mae"):
    output_X, output_Y = [], []
    if inverse_for_minimise == True:
        coff = -1
    else:
        coff = 1
    for ID in range(1, find_largest_number(design_history_dict.keys())+1):
        if not design_history_dict[ID][objective_function_name] == None:
            output_X = output_X + [design_history_dict[ID]["X"]]
            output_Y = output_Y + [design_history_dict[ID][objective_function_name] * coff]
    return output_X, output_Y

def run_experiment_and_return_updated_design_history_dict(design_history_dict_single, experiment_requester, model_testing_method, testing_measure="mae"):
    
    global global_strptime_str
    col_training_str = "training_" + testing_measure
    col_testing_str = "testing_" + testing_measure
    
    for col_str in [col_training_str, col_testing_str]:
       if not col_str in design_history_dict_single.keys():
           design_history_dict_single[col_str] = None
    
    if design_history_dict_single[col_training_str] == None:
        predictor, training_scores = experiment_requester(design_history_dict_single["X"])
        design_history_dict_single["predictor"] = predictor
        design_history_dict_single["training_r2"], design_history_dict_single["training_mse"], design_history_dict_single["training_mae"] = training_scores["r2"], training_scores["mse"], training_scores["mae"]
    if design_history_dict_single[col_testing_str] == None:
        temp_input_dict = return_edited_input_dict(design_history_dict_single["X"], design_space_dict, default_input_dict)
        testing_scores, X_testing, y_testing, Y_preds = model_testing_method(predictor, temp_input_dict)
        del temp_input_dict
        design_history_dict_single["testing_r2"], design_history_dict_single["testing_mse"], design_history_dict_single["testing_mae"] = testing_scores["r2"], testing_scores["mse"], testing_scores["mae"]
        design_history_dict_single["Y"] = testing_scores[testing_measure]
        # custom testing
        results_tables_dict, plt, df_realigned_dict = FG_additional_reporting.run_additional_reporting(preds=Y_preds,                                                                   
                X_test = X_testing, 
                pred_steps_list = [1], 
                pred_output_and_tickers_combos_list = ("company", "close"), 
                DoE_orders_dict = None, 
                model_type_name = "xxxModel_namexxx", 
                outputs_path = "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\outputs",
                model_start_time = datetime.now())
        
        
    design_history_dict_single["experiment_timestamp"] = datetime.now().strftime(global_strptime_str)
    
    return design_history_dict_single



#%% Module - Experiment Handler

def PLACEHOLDER_objective_function(x):
    return (x[:, 0] - 2)**2 + (x[:, 1] - 3)**2

def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

def convert_floats_to_int_if_whole(input_list):
    output_list = []
    for i in range(len(input_list)):
        if is_integer_num(input_list[i]):
            output_list = output_list + [int(input_list[i])]
        else:
            output_list = output_list + [float(input_list[i])]
    return output_list

def f2f(x1):
    return np.random.rand()

def experiment_manager(
    optim_run_name,
    design_space_dict,
    model_start_time = datetime.now(),
    model_training_method=FG_model_training.retrieve_or_generate_model_and_training_scores,
    model_testing_method=FG_model_training.return_testing_scores_and_testing_time_series,
    initial_doe_size_or_DoE=5,
    max_iter=5,
    optimisation_method=None,
    default_input_dict = default_input_dict,
    minimise=True,
    force_restart_run = False,
    testing_measure = "mae"
    ):
    
    #parameters
    global global_precalculated_assets_locations_dict, global_designs_record_final_columns_list, global_outputs_folder
    potential_experiment_records_path = os.path.join(global_precalculated_assets_locations_dict["root"], global_precalculated_assets_locations_dict["experiment_records"])
    output_results_path = os.path.join(global_outputs_folder, datetime.now().strftime("%Y%m%d_%H%M") + "_" + optim_run_name)
    experi_params_list = return_list_of_experimental_params(design_space_dict)
    bounds = convert_design_space_dict_to_GPyOpt_bounds_list(design_space_dict)
    experiment_requester = return_experiment_requester(design_space_dict, default_input_dict=default_input_dict, experiment_method=model_training_method)
    list_of_save_locations = [potential_experiment_records_path, output_results_path]
    
    #bo = GPyOpt.methods.BayesianOptimization(f=experiment_requester, domain=bounds)
    
    # create a optimisation run folder in the outputs folder
    try:
        os.makedirs(output_results_path)
    except:
        print("output folder already existed")
    
    print("XXXXXXXXXXXXXX")
    if os.path.exists(potential_experiment_records_path + optim_run_name + ".py_dict") and force_restart_run == False:
        # load previous work
        df_designs_record = pd.read_csv(potential_experiment_records_path + optim_run_name + ".csvBACKUP")
        with open(potential_experiment_records_path + optim_run_name + ".py_dict", 'rb') as file:
            design_history_dict = pickle.load(file)
        
        #check that the previous designs table, matches the format for this experiment
        if not sum(df_designs_record.columns == return_keys_within_2_level_dict(design_space_dict) + global_designs_record_final_columns_list) == len(df_designs_record.columns):
            raise ValueError("previous designs table, doesn't match the format for this experiment")
        
        
        
    #insert the completion of the DoE if not completed
    else:
        # create and save the design records table
        designs_record_cols = ["ID"] + return_keys_within_2_level_dict(design_space_dict) + global_designs_record_final_columns_list
        df_designs_record = pd.DataFrame(columns=designs_record_cols)
        df_designs_record.set_index("ID", inplace=True)
        design_history_dict = {"design_space_dict" : design_space_dict, "default_input_dict" : default_input_dict}    
        save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        
        # populate initial DoE
        if type(initial_doe_size_or_DoE) == int:
            X_init = define_DoE(bounds, initial_doe_size_or_DoE)
            X_init[0] = [7.00e+00, 3.00e-01, 1.00e+00, 2.52e+04, 2.00e+00, 1.00e-01] #fg_placeholder
            X_init[1] = [9.0e+00, 3.0e-01, 1.0e+00, 7.2e+03, 3.0e+00, 1.0e-02]
            X_init[2] = [5.0e+00, 3.0e-01, 1.0e+00, 7.2e+03, 2.0e+00, 1.0e-01]
            X_init[3] = [7.00e+00, 3.00e-01, 0.00e+00, 2.52e+04, 0.00e+00, 1.00e-01]
            X_init[4] = [5.0e+00, 3.0e-01, 1.0e+00, 1.8e+03, 1.0e+00, 5.0e-02]  
        else:
            X_init = initial_doe_size_or_DoE
        
        for ID in range(len(X_init)):
            design_history_dict[ID] = dict()
            design_history_dict[ID]["X"] = X_init[ID]
            for k in global_designs_record_final_columns_list:
                design_history_dict[ID][k] = None
            
    
    # complete all incomplete experiment runs (DoE or otherwise)
    df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
    for ID in range(find_largest_number(design_history_dict.keys()) + 1):
        print(return_keys_within_2_level_dict(design_space_dict))
        print(design_history_dict[ID]["X"])
        design_history_dict[ID]["X"] = convert_floats_to_int_if_whole(design_history_dict[ID]["X"])#[:len(design_history_dict[ID-1]["X"])]
        # only run value if testing measure missing
        if design_history_dict[ID]["testing_" + testing_measure] == None:
            design_history_dict[ID] = run_experiment_and_return_updated_design_history_dict(design_history_dict[ID], experiment_requester, model_testing_method, testing_measure="mae")
            # save
            df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
            save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
            # fg_placeholder - new functionality
            
            
            
            
            
    
    # continue optimisation
    bo = GPyOpt.methods.BayesianOptimization(f=PLACEHOLDER_objective_function, domain=bounds, initial_design_numdata=0)
    continue_optimisation = True
    if type(initial_doe_size_or_DoE) in [list, dict]:
        overall_max_runs = len(initial_doe_size_or_DoE) + max_iter
    else:
        overall_max_runs = initial_doe_size_or_DoE + max_iter
        
    while continue_optimisation == True:
        X, Y = return_X_and_Y_for_GPyOpt_optimisation(design_history_dict, bo, True, objective_function_name="testing_" + testing_measure)
        bo.X = np.array(X)
        bo.Y = np.array(Y).reshape(-1,1)
        bo.run_optimization()
        # find next design
        x_next = bo.acquisition.optimize()
        # save and run design
        ID = find_largest_number(design_history_dict.keys()) + 1
        design_history_dict[ID] = dict()
        design_history_dict[ID]["X"] = convert_floats_to_int_if_whole(list(x_next[0][0]))#[:len(design_history_dict[ID-1]["X"])]
        # FG_placeholder
        design_history_dict[ID] = run_experiment_and_return_updated_design_history_dict(design_history_dict[ID], experiment_requester, model_testing_method, testing_measure="mae")
        # save
        df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
        save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        # check loop
        
        if len(df_designs_record.index) >= overall_max_runs:
            continue_optimisation = False
        
#%% save dict for export testing


#preds, X_test, pred_steps_list, pred_output_and_tickers_combos_list, DoE_orders_dict, model_type_name, outputs_path, model_start_time
    
    
    
    
    

#%% main line

now = datetime.now()
model_start_time = now.strftime(global_strptime_str_filename)
    
#initial_doe_size_or_DoE=[[5, 3600, 0], [7, 7200, 0], [5, 3600, 1]] fg_placeholder

design_space_dict_original = {
    "senti_inputs_params_dict" : {
        "topic_qty" : range(4,9,1),
        "relative_halflife" : [SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR]
    },
    "model_hyper_params" : {
        #"estimator__hidden_layer_sizes" : ["100_10", "50_20_10", "20_10"]
        #"estimator__hidden_layer_sizes" : ["100_10", "15_10", "20_10"]
        #"estimator__hidden_layer_sizes" : [10, 10, 10]
        #"estimator__hidden_layer_sizes" : [(100, 10), (50, 20, 10), (20, 10)]
        "estimator__hidden_layer_sizes" : {0 : (10, 10),
                                           1 : (20, 10),
                                           2 : (100, 10),
                                           3 : (50, 20, 10),
                                           4 : (20, 10)}
    },
    "string_key" : {}
}

design_space_dict = {
    "senti_inputs_params_dict" : {
        "topic_qty" : [5, 7, 9],
        "topic_model_alpha" : [0.3, 0.7, 1],
        "weighted_topics" : [True, False],
        "relative_halflife" : [0.5 * SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR]
    },
    "model_hyper_params" : {
        "estimator__hidden_layer_sizes" : {0 : (10, 10),
                                           1 : (20, 10),
                                           2 : (100, 10),
                                           3 : (50, 20, 10)},
        "estimator__alpha"                 : [0.01, 0.05, 0.1]
    },
    "string_key" : {}
}

global_run_count = 0

experiment_manager(
    "test",
    design_space_dict,
    initial_doe_size_or_DoE=5,
    model_start_time = model_start_time,
    force_restart_run = True
    )







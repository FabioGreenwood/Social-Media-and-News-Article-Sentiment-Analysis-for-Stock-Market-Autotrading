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
import random
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
from datetime import datetime, timedelta
import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import random

#%% Standard Parameters


#GLOBAL PARAMETERS
global_master_folder_path = r"C:\Users\Public\fabio_uni_work\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\"
global_input_cols_to_include_list = ["<CLOSE>", "<HIGH>"]
global_index_cols_list = ["<DATE>","<TIME>"]
global_index_col_str = "datetime"
global_target_file_folder_path = ""
global_feature_qty = 6
global_outputs_folder_path = ".\\outputs\\"
global_financial_history_folder_path = "FG action, do I need to update this?"
global_df_stocks_list_file           = pd.read_csv(os.path.join(global_master_folder_path,r"data\support_data\stock_info.csv"))
global_start_time                    = datetime.now()
global_error_str_1 = "the input {} is wrong for the input training_or_testing"
global_random_state = 1
#global_scores_database = pd.read_csv(os.path.join(global_master_folder_path,r"outputs\scores_database.csv"))
global_strptime_str = '%d/%m/%y %H:%M:%S'
global_strptime_str_filename = '%d_%m_%y %H:%M:%S'
global_strptime_str_2 = '%d/%m/%y %H:%M'
global_precalculated_assets_locations_dict = {
    "root" : os.path.join(global_master_folder_path,r"precalculated_assets\\"),
    "topic_models"              : "topic_models\\",
    "annotated_tweets"          : "annotated_tweets\\",
    "predictive_model"          : "predictive_model\\",
    "sentiment_data"          : "sentiment_data\\",
    "technical_indicators"      : "technical_indicators\\",
    "experiment_records"        : "experiment_records\\",
    "clean_tweets"              : "cleaned_tweets_ready_for_subject_discovery\\"
    }
global_outputs_folder = os.path.join(global_master_folder_path,r"outputs\\")
global_designs_record_final_columns_list = ["experiment_timestamp", "training_r2", "training_mse", "training_mae", "validation_r2", "validation_mse", "validation_mae", "testing_r2", "testing_mse", "testing_mae", "profitability", "predictor_names"]


SECS_IN_A_DAY = 60*60*24
SECS_IN_AN_HOUR = 60*60
FIVE_MIN_TIME_STEPS_IN_A_DAY = SECS_IN_A_DAY / (5*60)


#%% Default Input Parameters

default_temporal_params_dict    = {
    "train_period_start"    : datetime.strptime('01/01/15 00:00:00', global_strptime_str),
    #"train_period_end"      : datetime.strptime('01/06/19 00:00:00', global_strptime_str),
    "train_period_end"      : datetime.strptime('01/02/15 00:00:00', global_strptime_str), #FG_placeholder
    "time_step_seconds"     : 5*60, #5 mins,
    "test_period_start"     : datetime.strptime('01/11/19 00:00:00', global_strptime_str), #FG_placeholder
    #"test_period_start"     : datetime.strptime('01/06/19 00:00:00', global_strptime_str),
    "test_period_end"       : datetime.strptime('01/01/20 00:00:00', global_strptime_str),
}
default_fin_inputs_params_dict      = {
    "index_cols"        : "date",    
    "cols_list"         : ["open", "high", "low", "close", "volume"],
    "fin_indi"          : {#additional financial indicators to generate
        "sma" : [5, 20, 50, int(FIVE_MIN_TIME_STEPS_IN_A_DAY), int(5 * FIVE_MIN_TIME_STEPS_IN_A_DAY)],
        "ema" : [5, 20, 50, int(FIVE_MIN_TIME_STEPS_IN_A_DAY), int(5 * FIVE_MIN_TIME_STEPS_IN_A_DAY)],
        "macd" : [[12, 26, 9]],
        "BollingerBands" : [[20, 2]],
        "PivotPoints" : [0]}, 
    "fin_match"         :{
        "Doji" : True},
    "index_col_str"     : "datetime",
    #"historical_file"   : "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\data\\financial data\\tiingo\\aapl.csv",
    "historical_file"   : os.path.join(global_master_folder_path,r"data\financial_data\firstratedata\AAPL_full_5min_adjsplit.txt")
}
default_senti_inputs_params_dict    = {
    "topic_qty"             : 7,
    "topic_training_tweet_ratio_removed" : int(1e5),
    "relative_lifetime"     : 60*60*24*7, # units are seconds
    "relative_halflife"     : 60*60*0.5, # units are seconds
    "topic_model_alpha"     : 1,
    "weighted_topics"       : False,
    "apply_IDF"             : True,
    "enforced_topics_dict_name" : "None",
    "enforced_topics_dict"  : None,
    "sentiment_method"      : SentimentIntensityAnalyzer(),
    "tweet_file_location"   : os.path.join(global_master_folder_path,r"data\twitter_data\apple.csv"),
    "regenerate_cleaned_tweets_for_subject_discovery" : False,
    "inc_new_combined_stopwords_list" : True,
    "topic_weight_square_factor" : 1
}
default_outputs_params_dict         = {
    "output_symbol_indicators_tuple"    : ("aapl", "close"), 
    "pred_steps_ahead"                  : 1
}
default_cohort_retention_rate_dict = {
            "£_close" : 1, #output value
            "£_*": 1, #other OHLCV values
            "$_*" : 0.4, # technical indicators
            "match!_*" : 0.6, #pattern matchs
            "~senti_*" : 0.6, #sentiment analysis
            "*": 0.5} # other missed values
default_model_hyper_params          = {
    "name" : "RandomSubspace_RNN_Regressor", #Multi-layer Perceptron regressor
    #"name" : "RandomSubspace_RNN_Regressor", #Multi-layer Perceptron regressor
        #general hyperparameters
    "n_estimators_per_time_series_blocking" : 1,
    "training_error_measure_main"   : 'neg_mean_squared_error',
    "testing_scoring"               : ["r2", "mse", "mae"],
    "time_series_split_qty"         : 5,
        #model specific rows
    "estimator__alpha"                 : 0.05,
    "estimator__hidden_layer_sizes"    : (100,10), 
    "estimator__activation"            : 'relu',
    "cohort_retention_rate_dict"       : default_cohort_retention_rate_dict,
    "epochs" : 5,
    "lookbacks" : 10,
    "shuffle_fit" : False}
default_reporting_dict              = {
    "confidence_thresholds" : [0, 0.01, 0.02, 0.035, 0.05, 0.1],
    "confidence_thresholds_inserted_to_df" : {
        "PC_confindence" : [0.02],
        "score_confidence" : [0.02],
        "score_confidence_weighted" : [0.02]}}
default_input_dict = {
    "temporal_params_dict"      : default_temporal_params_dict,
    "fin_inputs_params_dict"    : default_fin_inputs_params_dict,
    "senti_inputs_params_dict"  : default_senti_inputs_params_dict,
    "outputs_params_dict"       : default_outputs_params_dict,
    "model_hyper_params"        : default_model_hyper_params,
    "reporting_dict"            : default_reporting_dict
    }



#%% misc methods

def filter_df_multi_col(df, list_of_cols, values_to_match):
    #match only
    mask = df[list_of_cols[0]] == values_to_match[0]
    for val, col in zip(values_to_match, list_of_cols):
        new_con = df[col] == val
        mask = (mask) & (new_con)
    return df[mask]

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

def update_df_designs_record(df_designs_record, design_history_dict, design_space_dict, skip_column_check=False):
    global global_designs_record_final_columns_list
    
    if skip_column_check == False and not all(df_designs_record.columns == return_keys_within_2_level_dict(design_space_dict) + return_cols_for_additional_reporting(default_input_dict) + global_designs_record_final_columns_list):
        raise ValueError("the schema for the design records table doesn't match, please review")
    
    input_param_cols = return_keys_within_2_level_dict(design_space_dict)
    
    for ID in range(find_largest_number(design_history_dict.keys())+1):
        df_designs_record.loc[ID, input_param_cols] = design_history_dict[ID]["X"]
        # standard results
        for subkey in design_history_dict[ID]:
            if not subkey in ["X", "predictor", "Y", "validation_results_dict", "testing_results_dict"]:
                df_designs_record.loc[ID, subkey] = design_history_dict[ID][subkey]
        # additional results dict
        for prefix, additional_results_dict in zip(["validation", "testing"], ["validation_results_dict", "testing_results_dict"]):
            if additional_results_dict in design_history_dict[ID].keys():
                for result_type in design_history_dict[ID][additional_results_dict].keys():
                    for steps_back in design_history_dict[ID][additional_results_dict][result_type].keys():
                        for confidence in design_history_dict[ID][additional_results_dict][result_type][steps_back]:
                            df_designs_record.loc[ID, return_name_of_additional_reporting_col(prefix, result_type, confidence)] = design_history_dict[ID][additional_results_dict][result_type][steps_back][confidence]
                
    return df_designs_record

def return_name_of_additional_reporting_col(validation_or_testing_prefix, first_str, third_str_confidence):
    return validation_or_testing_prefix + "_" + str(first_str[8:]) + "_s" + "X" + "_c" + str(third_str_confidence)
    #return str(first_str[8:]) + "_s" + str(second_str_mins) + "_c" + str(third_str_confidence)

def return_cols_for_additional_reporting(input_dict):

    output_cols = []
    confidences = input_dict["reporting_dict"]["confidence_thresholds"]
    pred_steps_ahead_list = input_dict["outputs_params_dict"]["pred_steps_ahead"]
    
    template = FG_additional_reporting.run_additional_reporting(y_testing=pd.DataFrame(), pred_steps_list=[])
    
    if type(pred_steps_ahead_list) == int:
        pred_steps_ahead_list = [pred_steps_ahead_list]
    
    for pre_fix_str in ["validation", "testing"]:
        for result_type in list(template.keys()):
            for pred_steps in pred_steps_ahead_list:
                for confidence in confidences:
                    output_cols = output_cols + [return_name_of_additional_reporting_col(pre_fix_str, result_type, confidence)]
    
    return output_cols

def add_missing_designs_to_design_history_dict(design_history_dict, initial_doe_size_or_DoE):
    #check if any designs
    global global_designs_record_final_columns_list
    for new_design, id in zip(initial_doe_size_or_DoE, range(len(initial_doe_size_or_DoE))):
        design_found = False
        largest_existing_ID = find_largest_number(design_history_dict)
        for existing_design_ID in range(largest_existing_ID + 1):
            check = np.all(design_history_dict[existing_design_ID]["X"] == new_design)
            if check == True:
                design_found = True
                
        if design_found == False:
            design_history_dict[largest_existing_ID + 1] = dict()
            design_history_dict[largest_existing_ID + 1]["X"] = new_design
            for k in global_designs_record_final_columns_list:
                design_history_dict[largest_existing_ID + 1][k] = None
    return design_history_dict

def return_scenario_name_str(topic_qty, pred_steps, ratio_removed):
    if topic_qty == None:
        output = "multi_topic_steps_"
    elif topic_qty == 1:
        output = "no_topics_steps_"
    elif topic_qty == 0:
        output = "no_sentiment_steps_"
    else:
        raise ValueError("topic_qty value of: " + str(topic_qty) + " not recognised")
    if not pred_steps in [1, 3, 5, 15]:
        raise ValueError("double check value " + str(pred_steps) + " desired for pred steps input")
    
    valstr = "{0:e}".format(ratio_removed)
    epos = valstr.rfind('e')
    exponent = valstr[epos+1:]
    removal_str = "removal_1e" + str(exponent)
    
    return output + str(pred_steps) + "_" + removal_str

def update_design_hist_dict_post_training(design_history_dict_single, predictor, training_scores_dict, validation_scores_dict, additional_validation_dict):
    design_history_dict_single["predictor"] = predictor
    design_history_dict_single["training_r2"], design_history_dict_single["training_mse"], design_history_dict_single["training_mae"] = training_scores_dict["r2"], training_scores_dict["mse"], training_scores_dict["mae"]
    design_history_dict_single["validation_r2"], design_history_dict_single["validation_mse"], design_history_dict_single["validation_mae"] = validation_scores_dict["r2"], validation_scores_dict["mse"], validation_scores_dict["mae"]
    design_history_dict_single["validation_results_dict"] = additional_validation_dict
    design_history_dict_single["experiment_timestamp"] = datetime.now().strftime(global_strptime_str)
    return design_history_dict_single

def update_design_hist_dict_post_testing(design_history_dict_single, testing_scores, testing_results_dict):
    design_history_dict_single["testing_r2"], design_history_dict_single["testing_mse"], design_history_dict_single["testing_mae"] = testing_scores["r2"], testing_scores["mse"], testing_scores["mae"]
    design_history_dict_single["Y"] = testing_scores[testing_measure]
    design_history_dict_single["testing_results_dict"] = testing_results_dict
    return design_history_dict_single

        
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
    reporting_dict                    = prepped_input_dict["reporting_dict"]
    
    if type(prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]) == str or type(prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]) == np.str_:
        original_value = prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"]
        updated_val = []
        for x in original_value.split("_"):
            updated_val = updated_val + [x]
        prepped_input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"] = tuple(updated_val)
    
    global global_run_count
    
    print("xxxxxxxxxxxxxx" + str(global_run_count))
    print(GPyOpt_input)
    global_run_count += 1
    
    predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = experiment_method(prepped_temporal_params_dict, prepped_fin_inputs_params_dict, prepped_senti_inputs_params_dict, prepped_outputs_params_dict, prepped_model_hyper_params, reporting_dict)
    return predictor, training_scores_dict, validation_scores_dict, additional_validation_dict
    
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
    # please note this has been converted to a multi-objective function, it will cycle through the objective_function_name variable if list
    # this functionality can be removed by entering a non-iterable as objective_function_name
    output_X, output_Y = [], []
    
    if type(objective_function_name) == list:
        if not type(inverse_for_minimise) == list or not len(inverse_for_minimise) == len(objective_function_name):
            raise ValueError("if objective_function_name is a list, then inverse_for_minimise must also be a list of the same length")
        temp_index = len(design_history_dict) % len(objective_function_name)
        objective_function_name = objective_function_name[temp_index]
        inverse_for_minimise    = inverse_for_minimise[temp_index]
    
    if inverse_for_minimise == True:
        coff = -1
    else:
        coff = 1
    
        
    for ID in range(0, find_largest_number(design_history_dict.keys())+1):
        output_X = output_X + [design_history_dict[ID]["X"]]
        if type(objective_function_name) == str:
            output_Y = output_Y + [design_history_dict[ID][objective_function_name] * coff]
        else:
            target = design_history_dict[ID]
            for val in objective_function_name:
                if val == "validation":
                    val = val + "_results_dict"
                target = target[val]
            output_Y = output_Y + [target * coff]
    return output_X, output_Y

def run_experiment_and_return_updated_design_history_dict(design_history_dict_single, experiment_requester, model_testing_method, testing_measure="mae", confidences_before_betting_PC=[0.00, 0.01, 0.02]):
    
    global global_strptime_str, global_outputs_folder
    col_training_str    = "training_" + testing_measure
    col_testing_str     = "testing_" + testing_measure
    
    for col_str in [col_training_str, col_testing_str]:
       if not col_str in design_history_dict_single.keys():
           design_history_dict_single[col_str] = None
    
    if design_history_dict_single[col_training_str] == None:
        predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = experiment_requester(design_history_dict_single["X"])
        design_history_dict_single = update_design_hist_dict_post_training(design_history_dict_single, predictor, training_scores_dict, validation_scores_dict, additional_validation_dict)
    
    if design_history_dict_single[col_testing_str] == None:
        temp_input_dict = return_edited_input_dict(design_history_dict_single["X"], design_space_dict, default_input_dict)
        testing_scores, X_testing, y_testing, Y_preds = model_testing_method(design_history_dict_single["predictor"], temp_input_dict)
        del temp_input_dict
        testing_results_dict = FG_additional_reporting.run_additional_reporting(preds=Y_preds,
                y_testing = y_testing, 
                pred_steps_list = default_input_dict["outputs_params_dict"]["pred_steps_ahead"],
                confidences_before_betting_PC = confidences_before_betting_PC
                )
        design_history_dict_single = update_design_hist_dict_post_testing(design_history_dict_single, testing_scores, testing_results_dict)
        
    return design_history_dict_single



#%% Module - Experiment Handlerk

def PLACEHOLDER_objective_function(x1, x2, x3, x4, x5):
    #return (x[:, 0] - 2)**2 + (x[:, 1] - 3)**2
    return 5

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

# new methods for filing

def trim_string_from_substring(string, substring="_params_"):
    x = string.find(substring) + len(substring)
    return string[x:]

def print_desired_scores(design_history, df_designs_record, design_space_dict, optim_scores_vec, inverse_for_minimise_vec, optim_run_name):
    # this method produces the desired output at the end of a run, informing the user of:
    # number of design runs, last score, best scores of each high-lighted score and best run with each score in question
    # last score info
    """ ID X
        inputs: 
        scores: """
    """ top scores:
        pareto design scores:
        ID: X scores:
    """
    last_ID = max(df_designs_record[df_designs_record["testing_mse"] > 0].index)
    print(optim_run_name)
    print("ID: ", str(last_ID))
    output_string_2 = "inputs - "
    for name_1, val_1 in zip(return_keys_within_2_level_dict(design_space_dict), design_history[last_ID]["X"]):
        output_string_2 = output_string_2 + trim_string_from_substring(name_1) + ": " + str(val_1) + ", "
    print(output_string_2[:-2])
    print("short form - " + str(design_history[last_ID]["X"]))
    output_string_3 = "outputs - "
    for score_name in optim_scores_vec:
        if type(score_name) == str:
            temp_str = score_name
        else:
            temp_str = return_name_of_additional_reporting_col(score_name[0], score_name[1], score_name[3])
        output_string_3 = output_string_3 + str(temp_str) + ": " + "{:.5f}".format(df_designs_record[temp_str][last_ID]) + ", "
            #output_string_3 = output_string_3 + str(temp_str) + ": " + str(df_designs_record[temp_str][last_ID]) + ", "
    print(output_string_3[:-2])
    # pareto designs
    print("pareto designs")
    for objective, polarity in zip(optim_scores_vec, inverse_for_minimise_vec):
        if type(objective) == str:
            temp_str = objective
        else:
            temp_str = return_name_of_additional_reporting_col(objective[0], objective[1], objective[3])
        if polarity == True:
            temp_col = df_designs_record[temp_str].fillna(np.inf)
            pareto_index = temp_col.idxmin()
        else:
            temp_col = df_designs_record[temp_str].fillna(-np.inf)
            pareto_index = temp_col.idxmax()
        
        output_string_4a = "pareto " + temp_str + ": " + "{:.5f}".format(df_designs_record[temp_str][pareto_index]) + ", ID: " + str(pareto_index)
        output_string_4b = "inputs - "
        for name_4, val_4 in zip(return_keys_within_2_level_dict(design_space_dict), design_history[pareto_index]["X"]):
            output_string_4b = output_string_4b + trim_string_from_substring(name_4) + ": " + str(val_4) + ", "
        print("----")
        print(output_string_4a)
        print(output_string_4b)
        
        
        
def return_if_design_unique(x_next, design_history_dict):
    try:
        x_next = x_next[0][0]
    except:
        x_next = x_next
    
    largest_ID = find_largest_number(design_history_dict.keys())
    unique = True
    for ID in range(0, largest_ID):
        if all(design_history_dict[ID]["X"] == x_next):
            unique = False
    return unique
    

def check_if_experiment_already_ran_if_so_return_random_unique_design(x_next, design_history_dict, bounds):
    unique = return_if_design_unique(x_next, design_history_dict)
    if unique == False:
        x_next = define_DoE(bounds, 1)
        x_next = x_next[0]
        
    return x_next
        

def update_global_record(pred_steps, df_designs_record, experi_params_list, run_name, global_record_path):
    if global_record_path=="":
        return
    
    global_ID_str = "global_ID"
    local_ID_str = "local_ID"
    run_name_col_str = "run_name"
    pred_steps_str = "pred_steps"
    
    for i in range(len(experi_params_list)):
        experi_params_list[i] = experi_params_list[i].replace("~", "_")
    
    col1 = [local_ID_str, run_name_col_str, pred_steps_str]
    col2 = list(df_designs_record.columns)
    run_specific_cols = [local_ID_str, run_name_col_str, pred_steps_str]
    input_para_cols = run_specific_cols + experi_params_list
    
    if os.path.exists(global_record_path + "DONTTOUCH"):
        df_global_record = pd.read_csv(global_record_path+ "DONTTOUCH")
        df_global_record.set_index(df_global_record.columns[0], inplace=True)
    else:
        df_global_record = pd.DataFrame(columns=col1 + col2)

    for local_ID in df_designs_record.index:
        input_para = [local_ID, run_name, pred_steps] + list(df_designs_record.loc[local_ID, experi_params_list].values)
        #matching_rows = df_global_record[df_global_record[input_para_cols] == input_para]
        
        matching_rows = filter_df_multi_col(df_global_record, input_para_cols, input_para)
        
        if len(matching_rows) == 0:
            global_ID = max(list(df_global_record.index) + [-1]) + 1
            #df_global_record.loc[global_ID, input_para_cols] = input_para
        elif len(matching_rows) == 1:
            global_ID = matching_rows.index[0]
        elif len(matching_rows) > 1:
            
            raise ValueError("mulitple value of input: " + df_designs_record.loc[local_ID, experi_params_list] + "found")
        df_global_record.loc[global_ID,run_specific_cols] = [local_ID, run_name, pred_steps]
        df_global_record.loc[global_ID, df_designs_record.columns] = df_designs_record.loc[local_ID, :]
    #save resutls
    #print("printing results to " + str(global_record_path))
    try:
        df_global_record.to_csv(global_record_path)
        df_global_record.to_csv(global_record_path + "DONTTOUCH")
    except:
        df_global_record.to_csv(global_record_path + "DONTTOUCH")    
    
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
    testing_measure = "mae",
    inverse_for_minimise_vec = None,
    optim_scores_vec = None,
    global_record_path=""
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
        try:
            with open(potential_experiment_records_path + optim_run_name + ".py_dict", 'rb') as file:
                design_history_dict = pickle.load(file)
        except:
            designs_record_cols = ["ID"] + return_keys_within_2_level_dict(design_space_dict) + return_cols_for_additional_reporting(default_input_dict) + global_designs_record_final_columns_list
            df_designs_record = pd.DataFrame(columns=designs_record_cols)
            df_designs_record.set_index("ID", inplace=True)
            design_history_dict = {"design_space_dict" : design_space_dict, "default_input_dict" : default_input_dict}    
            save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        
        #check that the previous designs table, matches the format for this experiment
        if not sum(df_designs_record.columns == return_keys_within_2_level_dict(design_space_dict) + return_cols_for_additional_reporting(default_input_dict) + global_designs_record_final_columns_list) == len(df_designs_record.columns):
            raise ValueError("previous designs table, doesn't match the format for this experiment")

        # check if there are additional designs added to the DoE, if there are, add them to the design doc
        if isinstance(initial_doe_size_or_DoE, list) == True:
            design_history_dict = add_missing_designs_to_design_history_dict(design_history_dict, initial_doe_size_or_DoE)
            df_designs_record   = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
        
    
    else:
        # create and save the design records table
        designs_record_cols = ["ID"] + return_keys_within_2_level_dict(design_space_dict) + return_cols_for_additional_reporting(default_input_dict) + global_designs_record_final_columns_list
        df_designs_record = pd.DataFrame(columns=designs_record_cols)
        df_designs_record.set_index("ID", inplace=True)
        design_history_dict = {"design_space_dict" : design_space_dict, "default_input_dict" : default_input_dict}    
        save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        
        # populate initial DoE
        if type(initial_doe_size_or_DoE) == int:
            X_init = define_DoE(bounds, initial_doe_size_or_DoE)
        else:
            X_init = initial_doe_size_or_DoE
        
        for ID in range(len(X_init)):
            design_history_dict[ID] = dict()
            design_history_dict[ID]["X"] = X_init[ID]
            for k in global_designs_record_final_columns_list:
                design_history_dict[ID][k] = None
        df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
        save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        
    # complete all incomplete experiment runs !!of the same pred step and appropriate topic quant!! (DoE or otherwise)
    df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
    for ID in range(find_largest_number(design_history_dict.keys()) + 1):
        design_history_dict[ID]["X"] = convert_floats_to_int_if_whole(design_history_dict[ID]["X"])#[:len(design_history_dict[ID-1]["X"])]
        # only run value if testing measure missing
        if design_history_dict[ID]["testing_" + testing_measure] == None:
            print(return_keys_within_2_level_dict(design_space_dict))
            print(str(design_history_dict[ID]["X"]) + " running ID:" + str(ID))
            design_history_dict[ID] = run_experiment_and_return_updated_design_history_dict(design_history_dict[ID], experiment_requester, model_testing_method, testing_measure="mae", confidences_before_betting_PC=default_input_dict["reporting_dict"]["confidence_thresholds"])
            # save
            df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
            save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
            print_desired_scores(design_history_dict, df_designs_record, design_space_dict, optim_scores_vec, inverse_for_minimise_vec, optim_run_name)
        update_global_record(pred_steps, df_designs_record, experi_params_list, optim_run_name, global_record_path)
            
            
    
    # continue optimisation
    bo = GPyOpt.methods.BayesianOptimization(f=PLACEHOLDER_objective_function, domain=bounds, initial_design_numdata=0)
    continue_optimisation = True
    if type(initial_doe_size_or_DoE) in [list, dict]:
        overall_max_runs = len(initial_doe_size_or_DoE) + max_iter
    else:
        overall_max_runs = initial_doe_size_or_DoE + max_iter
    
    if len(df_designs_record.index) >= overall_max_runs:
        continue_optimisation = False
    
    while continue_optimisation == True:
        X, Y = return_X_and_Y_for_GPyOpt_optimisation(design_history_dict, bo, inverse_for_minimise=inverse_for_minimise_vec, objective_function_name=optim_scores_vec)
        bo.X = np.array(X)
        bo.Y = np.array(Y).reshape(-1,1)
        bo.run_optimization()
        # find next design
        dice_roll = random.random()
        print("dice_roll: " + str(dice_roll))
        if  dice_roll < 0.0:#25:
            x_next = define_DoE(bounds, 1)
            x_next = x_next[0]
        else:
            x_next = bo.acquisition.optimize()
            x_next = x_next[0][0]
        x_next = check_if_experiment_already_ran_if_so_return_random_unique_design(x_next, design_history_dict, bounds)
        # save and run design
        ID = find_largest_number(design_history_dict.keys()) + 1
        design_history_dict[ID] = dict()
        design_history_dict[ID]["X"] = convert_floats_to_int_if_whole(list(x_next))#[:len(design_history_dict[ID-1]["X"])]
        print(str(design_history_dict[ID]["X"]) + " running ID:" + str(ID))
        design_history_dict[ID] = run_experiment_and_return_updated_design_history_dict(design_history_dict[ID], experiment_requester, model_testing_method, testing_measure="mae", confidences_before_betting_PC=default_input_dict["reporting_dict"]["confidence_thresholds"])
        # save
        df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, design_space_dict)
        save_designs_record_csv_and_dict(list_of_save_locations, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name=optim_run_name)
        print_desired_scores(design_history_dict, df_designs_record, design_space_dict, optim_scores_vec, inverse_for_minimise_vec, optim_run_name)
        # check loop
        
        if len(df_designs_record.index) >= overall_max_runs:
            continue_optimisation = False
    print(datetime.now().strftime("%H:%M:%S") + "   normal termination")
        


#%% main line

now = datetime.now()
model_start_time = now.strftime(global_strptime_str_filename)
    
design_space_dict = {
    "senti_inputs_params_dict" : {
        "topic_qty" : [5, 9, 13, 17],
        "topic_model_alpha" : [0.3, 0.7, 1, 2, 3, 5],
        "weighted_topics" : [False, True],
        "relative_halflife" : [0.25 * SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR], 
        "apply_IDF" : [False, True],
        "topic_weight_square_factor" : [1, 2, 4]
        #,
        #"enforced_topics_dict"  : {
        #    0: None,
        #    1 : [['investment', 'financing', 'losses'], ['risk', 'exposure', 'liability'], ["financial",  "forces" , "growth", "interest",  "rates"]]}
    },
    "model_hyper_params" : {
        "estimator__hidden_layer_sizes" : {
            0 : [("simple", 50)],
            1 : [("simple", 40), ("simple", 30)],
            2 : [("GRU", 50)],
            3 : [("GRU", 40), ("GRU", 30)],
            4 : [("LSTM", 50)],
            5 : [("LSTM", 50), ("LSTM", 30)],
            7 : [("LSTM", 50), ("GRU", 50), ("simple", 50)],
            8 : [("simple", 50), ("GRU", 50), ("LSTM", 50)]
            },
        "estimator__alpha"                 : [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    },
    "string_key" : {}
}

global_run_count = 0

init_doe = 40

""" experiment checklist:
1. ensure that the value for steps ahead is updated on the dictionary line below
2. check the right lines are activated for the number of topics you wish to run
3. ensure the same is for the name of the run

checklist for restarting the experiment
1. del all the tables in [\experimental_records\]
2. the excels in [\outputs\]
"""

## SETTING THE RUN VARIABLES, READ ABOVE CHECK LIST

# definition of different scenarios is set by this dict, to access a different scenario, please change the scenario variable

# scenario parameters: topic_qty, pred_steps

scenario_ID = 0
removal_ratio = int(2e2)
scenario_dict = {
        0 : {"topics" : None, "pred_steps" : 1},
        1 : {"topics" : None, "pred_steps" : 3},
        2 : {"topics" : None, "pred_steps" : 5},
        3 : {"topics" : None, "pred_steps" : 15},
        4 : {"topics" : 1, "pred_steps" : 1},
        5 : {"topics" : 1, "pred_steps" : 3},
        6 : {"topics" : 1, "pred_steps" : 5},
        7 : {"topics" : 1, "pred_steps" : 15},
        8 : {"topics" : 0, "pred_steps" : 1},
        9 : {"topics" : 0, "pred_steps" : 3},
        10: {"topics" : 0, "pred_steps" : 5},
        11: {"topics" : 0, "pred_steps" : 15}
    }

for scenario_ID in [1,9,2,10]:#scenario_dict.keys():
    
    #editing topic quantity values for scenario, 2 lines
    topic_qty = scenario_dict[scenario_ID]["topics"]
    if isinstance(init_doe, list) and topic_qty != None:
            for i in range(len(init_doe)): init_doe[i][0] = 1
            design_space_dict["senti_inputs_params_dict"]["topic_qty"] = [1]
            default_input_dict["senti_inputs_params_dict"]["topic_qty"] = 1

    # setting various optimisation variabls
    pred_steps = scenario_dict[scenario_ID]["pred_steps"]
    testing_measure = "mae"
    default_input_dict["outputs_params_dict"]["pred_steps_ahead"] = pred_steps
    default_input_dict["senti_inputs_params_dict"]["topic_training_tweet_ratio_removed"] = removal_ratio

    # setting the optimisation objective functions
    confidence_scoring_measure_tuple_1 = ("validation","results_x_mins_weighted",pred_steps,0.02)
    confidence_scoring_measure_tuple_2 = ("validation","results_x_mins_PC",pred_steps,0.02)
    confidence_scoring_measure_tuple_3 = ("validation","results_x_mins_score",pred_steps,0.02)
    
    
    optim_scores_vec = ["validation_" + testing_measure, confidence_scoring_measure_tuple_1, confidence_scoring_measure_tuple_2, confidence_scoring_measure_tuple_3]
    
    inverse_for_minimise_vec = [True, False, False, False]
    
    optim_scores_vec = ["validation_" + testing_measure]
    inverse_for_minimise_vec = [True]
    

    #what around to ensure that single topic sentiment data in more used in the model
    if default_senti_inputs_params_dict["topic_qty"] == 1:
            default_model_hyper_params["cohort_retention_rate_dict"]["~senti_*"] = 1
    elif default_senti_inputs_params_dict["topic_qty"] == 0:
            default_model_hyper_params["cohort_retention_rate_dict"]["~senti_*"] = 0

    scenario_name_str = return_scenario_name_str(topic_qty, pred_steps, removal_ratio)
    scenario_name_str = "intergration_of_new_data_and_RNN"


    if __name__ == '__main__':
        #scenario_name_str = "test 19"
        print("running scenario " + str(scenario_ID) + ": " + scenario_name_str + " - " + datetime.now().strftime("%H:%M:%S"))
        experiment_manager(
            scenario_name_str,
            design_space_dict,
            initial_doe_size_or_DoE=init_doe,
            max_iter=0,
            model_start_time = model_start_time,
            force_restart_run = False,
            inverse_for_minimise_vec = inverse_for_minimise_vec,
            optim_scores_vec = optim_scores_vec,
            testing_measure = testing_measure,
            global_record_path=os.path.join(global_master_folder_path,r"outputs\intergration_of_new_data_and_RNN.csv")
            )
        print(str(scenario_ID) + " - complete" + " - " + datetime.now().strftime("%H:%M:%S"))







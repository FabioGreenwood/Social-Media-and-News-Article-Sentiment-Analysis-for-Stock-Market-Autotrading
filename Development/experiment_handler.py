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
global_designs_record_final_columns_list = ["training_r2", "training_mse", "training_mae", "testing_r2", "testing_mse", "testing_mae", "profitability", "predictor_names"]
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
        "sma" : [5],
        "ema" : [5]}, 
    "fin_match"         :{
        "Doji" : True},
    "index_col_str"     : "datetime",
    "historical_file"   : "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\data\\financial data\\tiingo\\aapl.csv",
}
default_senti_inputs_params_dict    = {
    "topic_qty"             : 7,
    "topic_training_tweet_ratio_removed" : int(1e6),
    "relative_lifetime"     : 60*60*1, #  hours
    "relative_halflife"     : 60*60*0.5, #one hour
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
    
def save_designs_record_csv_and_dict(potential_experiment_records_path, df_designs_record=None, design_history_dict=None):
    if type(df_designs_record) == pd.core.frame.DataFrame:
        try:
            with open(potential_experiment_records_path + ".csv", "wb") as file:
                pickle.dump(df_designs_record, file)
            with open(potential_experiment_records_path + ".csvBACKUP", "wb") as file:
                pickle.dump(df_designs_record, file)
        except:
            with open(potential_experiment_records_path + ".csvBACKUP", "wb") as file:
                pickle.dump(df_designs_record, file)
            print("please close the csv")
    if design_history_dict != None:
        with open(potential_experiment_records_path + ".py_dict", "wb") as file:
            pickle.dump(design_history_dict, file)

def update_df_designs_record(df_designs_record, design_history_dict, DoE_dict):
    global global_designs_record_final_columns_list
    
    
    if not all(df_designs_record.columns == return_keys_within_2_level_dict(DoE_dict) + global_designs_record_final_columns_list):
        raise ValueError("the schema for the design records table doesn't match, please review")
    
    input_param_cols = return_keys_within_2_level_dict(DoE_dict)
    
    for ID in range(find_largest_number(design_history_dict.keys())+1):
        df_designs_record.loc[ID, input_param_cols] = design_history_dict[ID]["X"]
        for subkey in design_history_dict[ID]:
            if subkey == "training_scores":
                df_designs_record.loc[ID, "training_r2"] = design_history_dict[ID][subkey]["r2"]
                df_designs_record.loc[ID, "training_mae"] = design_history_dict[ID][subkey]["mae"]
                df_designs_record.loc[ID, "training_mse"] = design_history_dict[ID][subkey]["mse"]
            elif subkey == "testing_scores":
                df_designs_record.loc[ID, "training_r2"] = design_history_dict[ID][subkey]["r2"]
                df_designs_record.loc[ID, "training_mae"] = design_history_dict[ID][subkey]["mae"]
                df_designs_record.loc[ID, "training_mse"] = design_history_dict[ID][subkey]["mse"]
            elif subkey != "X" and subkey != "predictor":
                df_designs_record.loc[ID, subkey] = design_history_dict[ID][subkey]
    return df_designs_record

        
        
#%% Experiment Parameters

DoE_1 = dict()

DoE_dict = {
    "senti_inputs_params_dict" : {
        "topic_qty" : range(4,9,1),
        "relative_halflife" : [SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR]
    },
    "model_hyper_params" : {
        "estimator__hidden_layer_sizes" : ["100_10", "50_20_10", "20_10"]
        #"estimator__hidden_layer_sizes" : [10, 10, 10]
    }
}

def return_list_of_experimental_params(experiment_instructions):
    list_of_experimental_parameters_names = []
    for key in experiment_instructions:
        for subkey in experiment_instructions[key]:
            name = key + "~" + subkey
            list_of_experimental_parameters_names = list_of_experimental_parameters_names + [name]
    return list_of_experimental_parameters_names

def convert_DoE_dict_to_GPyOpt_bounds_list(DoE_dict):
    output = []
    blank_parameter = {'name': None, 'type': None, 'domain': None}
    
    for key in DoE_dict:
        for subkey in DoE_dict[key]:
            name = key + "~" + subkey
            py_type = type(DoE_dict[key][subkey])
            value = DoE_dict[key][subkey]
            if py_type == list and type(value[0]) == str:
                type_str = "categorical"
                value = tuple(value)
            elif py_type == list and type(value[0]) == tuple:
                raise ValueError("Warning you can't enter tuple for the number of hidden layers yuo need to do a string of format 'X_X_X'")
                type_str = "categorical"
                value = tuple(value)
            elif py_type == list:
                type_str = "discrete"
                value = tuple(value)
            elif py_type == range:
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
            output_input_dict[key][subkey] = GPyOpt_input[input_index]
            input_index += 1
    
    return output_input_dict
            
def template_experiment_requester(GPyOpt_input, DoE_dict, default_input_dict=default_input_dict, experiment_method=FG_model_training.retrieve_or_generate_model_and_training_scores):
    
    prepped_input_dict = return_edited_input_dict(GPyOpt_input, DoE_dict, default_input_dict)
    
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
    
    
    
    predictor, training_scores = experiment_method(prepped_temporal_params_dict, prepped_fin_inputs_params_dict, prepped_senti_inputs_params_dict, prepped_outputs_params_dict, prepped_model_hyper_params)
    return predictor, training_scores
    
def return_experiment_requester(DoE_dict, default_input_dict=default_input_dict, experiment_method=FG_model_training.retrieve_or_generate_model_and_training_scores):
    output_method = lambda GPyOpt_input: template_experiment_requester(GPyOpt_input, DoE_dict, default_input_dict=default_input_dict, experiment_method=experiment_method)
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

def transfer_X_and_Y_to_GPyOpt_optimisation(design_history_dict, opt_obj):
    for ID in range(find_largest_number(design_history_dict.keys())+1):
        if "Y" in design_history_dict[ID].keys():
            opt_obj.X = np.vstack((opt_obj.X, design_history_dict[ID]["X"]))
            opt_obj.Y = np.vstack((opt_obj.Y, design_history_dict[ID]["Y"]))
    return opt_obj


def run_experiment_and_update_record(design_history_dict_single, experiment_requester, model_testing_method, testing_measure="mae"):
    if not "training_scores" in design_history_dict_single.keys():
        predictor, training_scores = experiment_requester(design_history_dict_single["X"])
        design_history_dict_single["predictor"], design_history_dict_single["training_scores"] = predictor, training_scores
    if not "testing_scores" in design_history_dict_single.keys():
        temp_input_dict = return_edited_input_dict(design_history_dict_single["X"], DoE_dict, default_input_dict)
        testing_scores, Y_preds_testing = model_testing_method(predictor, temp_input_dict)
        del temp_input_dict
        design_history_dict_single["testing_scores"] = testing_scores
        design_history_dict_single["Y"] = design_history_dict_single["testing_scores"][testing_measure]
    
    return design_history_dict_single
    
    
    


#%% Module - Experiment Handler

def PLACEHOLDER_objective_function(x):
    return (x[:, 0] - 2)**2 + (x[:, 1] - 3)**2

def experiment_manager(
    optim_run_name,
    DoE_dict,
    model_training_method=FG_model_training.retrieve_or_generate_model_and_training_scores,
    model_testing_method=FG_model_training.generate_testing_scores,
    initial_doe_size=5,
    max_iter=20,
    optimisation_method=None,
    default_input_dict = default_input_dict,
    minimise=True,
    force_restart_run = False,
    testing_measure = "mae"
    ):
    
    #parameters
    global global_precalculated_assets_locations_dict, global_designs_record_final_columns_list
    potential_experiment_records_path = os.path.join(global_precalculated_assets_locations_dict["root"], global_precalculated_assets_locations_dict["experiment_records"], optim_run_name)
    experi_params_list = return_list_of_experimental_params(DoE_dict)
    bounds = convert_DoE_dict_to_GPyOpt_bounds_list(DoE_dict)
    experiment_requester = return_experiment_requester(DoE_dict, default_input_dict=default_input_dict, experiment_method=model_training_method)
    #bo = GPyOpt.methods.BayesianOptimization(f=experiment_requester, domain=bounds)
    
    print("XXXXXXXXXXXXXX")
    if os.path.exists(potential_experiment_records_path) and force_restart_run == False:
        # load previous work
        
        
        raise SyntaxError("this isn't designed yet")
    #insert the completion of the DoE if not completed
    else:
        # create and save the design records table
        designs_record_cols = ["ID"] + return_keys_within_2_level_dict(DoE_dict) + global_designs_record_final_columns_list
        df_designs_record = pd.DataFrame(columns=designs_record_cols)
        df_designs_record.set_index("ID", inplace=True)
        design_history_dict = {"DoE_dict" : DoE_dict, "default_input_dict" : default_input_dict}    
        save_designs_record_csv_and_dict(potential_experiment_records_path, df_designs_record=df_designs_record, design_history_dict=design_history_dict)
        
        # populate initial DoE
        X_init = define_DoE(bounds, initial_doe_size)
        X_init = np.array([[5, 3600, "20_10"], [7, 7200, "20_10"]]) #FG_Placeholder
        for ID in range(len(X_init)):
            design_history_dict[ID] = dict()
            design_history_dict[ID]["X"] = X_init[ID]
            for k in global_designs_record_final_columns_list:
                design_history_dict[ID][k] = None
            
    
    # complete all incomplete experiment runs (DoE or otherwise)
    df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, DoE_dict)
    for ID in range(find_largest_number(design_history_dict.keys()) + 1):
        
        design_history_dict[ID] = run_experiment_and_update_record(design_history_dict[ID], experiment_requester, model_testing_method, testing_measure="mae")
        
        #update records
        df_designs_record = update_df_designs_record(df_designs_record, design_history_dict, DoE_dict)
    
    # continue optimisation
    bo = GPyOpt.methods.BayesianOptimization(f=PLACEHOLDER_objective_function, domain=bounds)
    continue_optimisation = True
    iter_count = 0
    while continue_optimisation == True:
        bo = transfer_X_and_Y_to_GPyOpt_optimisation(design_history_dict, bo)
        bo.run_optimization()
        # find next design
        x_next = bo.acquisition.optimize()
        # save and run design
        predictor, training_scores = experiment_requester(design_history_dict[ID]["X"])
        design_history_dict[ID]["predictor"], design_history_dict[ID]["training_scores"] = predictor, training_scores
        
        y_next = experiment_requester(x_next[0].reshape(1, -1))
        
        
        iter_count += 1
        if iter_count >= max_iter:
            continue_optimisation = False
        
        
        
        


#%% main line

experiment_manager(
    "test",
    DoE_dict,
    force_restart_run = True
    )







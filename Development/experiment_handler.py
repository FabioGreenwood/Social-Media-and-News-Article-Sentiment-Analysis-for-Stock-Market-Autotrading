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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns 
import jupyter
import sklearn
import math
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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import copy
from datetime import datetime
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle 
from gensim.models.ldamulticore import LdaMulticore
import gensim.models.ldamodel


from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.corpora as corpora
from pprint import pprint
from wordcloud import WordCloud
import os
from time import strftime, localtime
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing
from multiprocessing import Process
import itertools as it
from stock_indicators import indicators, Quote
from stock_indicators.indicators.common.quote import Quote
from stock_indicators.indicators.common.enums import Match


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
    "topic_training_tweet_ratio_removed" : int(1e5),
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
    "name" : "RandomSubspace_MLPRegressor ", #Multi-layer Perceptron regressor
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

#%% define variables

experiment_record = dict() # contains the runs, their inputs and outputs (inculding scores and models)




#%% Experiment Parameters

DoE_1 = dict()

experiment_instructions = {
    "DoE" : {
        "default_senti_inputs_params_dict" : {
            "topic_qty" : range(4,9,1),
            "relative_halflife" : [SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR]
        }
    },
    "default_model_hyper_params" : {
        "estimator__hidden_layer_sizes" : [(100,10), (50,20,10), (20,10)]
    },
    "optimisation" : False
}

def return_list_of_experimental_params(experiment_instructions):
    list_of_experimental_parameters_names = []
    for key in experiment_instructions:
        for subkey in experiment_instructions[key]:
            name = key + "~" + subkey
            list_of_experimental_parameters_names = list_of_experimental_parameters_names + [name]
    return list_of_experimental_parameters_names



def convert_experiment_instructions_to_GPyOpt_bounds_list(exp_instr):
    output = []
    blank_parameter = {'name': None, 'type': None, 'domain': None}
    
    for key in exp_instr:
        for subkey in exp_instr[key]:
            name = key + "~" + subkey
            py_type = type(exp_instr[key][subkey])
            value = exp_instr[key][subkey]
            if isinstance(py_type, list) == True and isinstance(py_type,value[0], str):
                type = "categorical"
            elif isinstance(py_type, list) == True:
                type = "descrete"
            elif isinstance(py_type, range) == True:
                type = "descrete"
                value = list(value)
            elif isinstance(py_type, tuple) == True:
                type = "continuous"
            else:
                raise ValueError("unrecognised input")
            output = output + [{'name': name, 'type': type, 'domain': value}]
    
    return output

def return_edited_input_dict(GPyOpt_input, exp_instr, default_input_dict):
    output_input_dict = copy.deepcopy(output_input_dict)
    input_index = 0
    for key in exp_instr:
        for subkey in exp_instr[key]:
            default_input_dict[key][subkey] = GPyOpt_input[input_index]
            input_index += 1
    
    return GPyOpt_input

            
def experiment_handler(GPyOpt_input, experiment_instructions, default_input_dict=default_input_dict, experiment_method=FG_model_training.retrieve_or_generate_model_and_training_scores):
    
    prepped_input_dict = return_edited_input_dict(GPyOpt_input, experiment_instructions, default_input_dict)
    
    prepped_temporal_params_dict      = prepped_input_dict["temporal_params_dict"]
    prepped_fin_inputs_params_dict    = prepped_input_dict["fin_inputs_params_dict"]
    prepped_senti_inputs_params_dict  = prepped_input_dict["senti_inputs_params_dict"]
    prepped_outputs_params_dict       = prepped_input_dict["outputs_params_dict"]
    prepped_model_hyper_params        = prepped_input_dict["model_hyper_params"]
    
    predictor, training_scores = experiment_method(prepped_temporal_params_dict, prepped_fin_inputs_params_dict, prepped_senti_inputs_params_dict, prepped_outputs_params_dict, prepped_model_hyper_params)
    return predictor, training_scores
    




#%% Module - Experiment Handler



def experiment_hander(
    optim_run_name,
    experiment_instructions,
    model_training_method=FG_model_training.retrieve_or_generate_model_and_training_scores,
    initial_doe_size=10,
    max_iter=20,
    optimisation_method=None,
    default_input_dict = default_input_dict,
    minimise=True,
    force_restart_run = False
    ):
    
    #parameters
    global global_precalculated_assets_locations_dict
    potential_experiment_records_path = os.path.join(global_precalculated_assets_locations_dict["root"], global_precalculated_assets_locations_dict["experiment_records"], optim_run_name)
    experi_params_list = return_list_of_experimental_params(experiment_instructions)
    bounds = convert_experiment_instructions_to_GPyOpt_bounds_list
    bo = GPyOpt.methods.BayesianOptimization(f=model_training_method, domain=bounds)
    
    #variables
    X_init, Y_init = None
    
    # load previous work
    if os.path.exists(potential_experiment_records_path) and force_restart_run == False:
        raise SyntaxError("this isn't designed yet")
    else:
        print("hello")
        #X_init = 
            
        #X_init = np.random.rand(initial_points, 2) * 5
        #Y_init = objective_function(X_init)
        
        
    minimized_value, optimal_params, bo = "hello"
    
    return minimized_value, optimal_params, bo



#import data_prep_and_model_training as FG_model_training







#%% main line

experiment_hander(
    "test",
    experiment_instructions
    )







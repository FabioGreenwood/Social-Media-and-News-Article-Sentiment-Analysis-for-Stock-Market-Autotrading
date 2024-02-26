"""
Required actions:
1. check that topic weight functionality is working
2. 
3. 
4. 
5. 

"""


#%%

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
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, date, time
import os
import warnings
import sys
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import clone_model
import additional_reporting_and_model_trading_runs as FG_additional_reporting
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
from sklearn.preprocessing import MinMaxScaler
import random

import pyLDAvis

from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim as gensimvis

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
#from stock_indicators import indicators, Quote
#from stock_indicators.indicators.common.quote import Quote
#from stock_indicators.indicators.common.enums import Match
#from stock_indicators import PeriodSize, PivotPointType 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Bidirectional, LSTM, Dense, GRU
import keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import timeseries_dataset_from_array
from config import global_strptime_str, global_strptime_str_filename, global_exclusively_str, global_precalculated_assets_locations_dict, global_financial_history_folder_path, global_error_str_1, global_financial_history_folder_path, global_df_stocks_list_file, global_random_state, global_strptime_str_2, global_index_cols_list, global_input_cols_to_include_list, global_start_time, global_financial_history_folder_path
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import Dense
import hashlib
from tensorflow.keras.callbacks import EarlyStopping

#tf.config.set_visible_devices([], 'GPU')


#%% EXAMPLE INPUTS FOR MAIN METHOD

#Temporal
#    Period
#    Time step (seconds)
#    Splitting method & quantity (tup)
#Input Features
#    Technical Indicators
#    Sentiment Parameters
#Output Features:
#    Prediction Features
#    Prediction Timesteps
#Model Hyperparameters
#    Inc rep number

#GLOBAL PARAMETERS



#%% misc methods
#misc methods

def return_input_dict(**kwargs):
    input_dict = {}
    potential_subdicts_names = ["temporal_params_dict", "fin_inputs_params_dict", "senti_inputs_params_dict", "outputs_params_dict", "model_hyper_params", "reporting_dict"]
    for subdict in potential_subdicts_names:
        if subdict in kwargs:
            input_dict[subdict] = kwargs[subdict]
    return input_dict

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

def return_topic_mode_seed_hash(enforced_topic_model_nested_list):
    name = "_seed_"
    if enforced_topic_model_nested_list == None:
        name2 = "na"
    else:
        name2 = ""
        #add the first initial of the first word of the first three subjects
        for topic_id in range(min(3,len(enforced_topic_model_nested_list))):
            name2 += enforced_topic_model_nested_list[topic_id][0][0]
        # then a hash number
        name2 += str(hash(str(enforced_topic_model_nested_list)))[:4]
    return name + name2
    
def return_topic_model_name(topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc):
    if int(topic_model_qty) > 1:
        file_string = "tm_qty" + str(topic_model_qty) + "_tm_alpha" + str(topic_model_alpha) + "_IDF-" + str(apply_IDF) + "_t_ratio_r" + str(tweet_ratio_removed) + return_topic_mode_seed_hash(enforced_topic_model_nested_list) + "_newStops" + str({True: 1, False: 0}[new_combined_stopwords_inc])
    elif int(topic_model_qty) == 1:
        file_string = "single_topic"
    elif int(topic_model_qty) == 0:
        file_string = "no_topics"
    return file_string

def return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor):
    global global_strptime_str, global_strptime_str_filename
    if topic_model_qty == 1 or topic_model_qty == 0:
        topic_weight_square_factor, topic_model_alpha = "NA", "NA"
    name = company_symbol + "_ps" + train_period_start.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_pe" + train_period_end.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_twsf" + str(topic_weight_square_factor) + "_"
    name = name + return_topic_model_name(topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc)
    return name

def return_sentiment_data_name(company_symbol, train_period_start, train_period_end, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe, factor_tweet_attention, factor_topic_volume):
    global global_strptime_str, global_strptime_str_filename
    name = return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor)
    name = name + "_ts_sec" + str(time_step_seconds)
    if topic_model_qty > 0:
        name += "_r_lt" + str(rel_lifetime) + "_r_hl" + str(int(rel_hlflfe))
        if factor_tweet_attention == True:
            name = name + "_factor_tweet_attentionTRUE"
        if factor_topic_volume != False:
            name = name + "_topicVol" + str(factor_topic_volume)

    return name

def return_predictor_name(input_dict):
    global global_strptime_str, global_strptime_str_filename
    company_symbol = input_dict["outputs_params_dict"]["output_symbol_indicators_tuple"][0]
    train_period_start  = input_dict["temporal_params_dict"]["train_period_start"]
    train_period_end    = input_dict["temporal_params_dict"]["train_period_end"]
    topic_model_qty     = input_dict["senti_inputs_params_dict"]["topic_qty"]
    topic_model_alpha   = input_dict["senti_inputs_params_dict"]["topic_model_alpha"]
    apply_IDF           = input_dict["senti_inputs_params_dict"]["apply_IDF"]
    tweet_ratio_removed = input_dict["senti_inputs_params_dict"]["topic_training_tweet_ratio_removed"]
    enforced_topic_model_nested_list = input_dict["senti_inputs_params_dict"]["enforced_topics_dict"]
    new_combined_stopwords_inc  = input_dict["senti_inputs_params_dict"]["inc_new_combined_stopwords_list"]
    topic_weight_square_factor  = input_dict["senti_inputs_params_dict"]["topic_weight_square_factor"]
    time_step_seconds   = input_dict["temporal_params_dict"]["time_step_seconds"]
    rel_lifetime        = input_dict["senti_inputs_params_dict"]["relative_lifetime"]
    rel_hlflfe          = input_dict["senti_inputs_params_dict"]["relative_halflife"]
    pred_steps_ahead    = input_dict["outputs_params_dict"]["pred_steps_ahead"]
    estm_alpha          = input_dict["model_hyper_params"]["estimator__alpha"]
    general_adjusting_square_factor   = input_dict["model_hyper_params"]["general_adjusting_square_factor"]
    lookbacks                         = input_dict["model_hyper_params"]["lookbacks"]
    batch_ratio                       = input_dict["model_hyper_params"]["batch_ratio"]
    financial_value_scaling           = input_dict["fin_inputs_params_dict"]["financial_value_scaling"]
    factor_tweet_attention            = input_dict["senti_inputs_params_dict"]["factor_tweet_attention"]
    factor_topic_volume               = input_dict["senti_inputs_params_dict"]["factor_topic_volume"]

    name = return_sentiment_data_name(company_symbol, train_period_start, train_period_end, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe, factor_tweet_attention, factor_topic_volume)
    predictor_hash = ""
    if not financial_value_scaling == None:
        predictor_hash += "_" + str(financial_value_scaling)
    predictor_hash += "_" + str(input_dict["model_hyper_params"]["n_estimators_per_time_series_blocking"]) + "_" + str(input_dict["model_hyper_params"]["testing_scoring"]) + "_" + str(input_dict["model_hyper_params"]["estimator__alpha"]) + "_" + str(input_dict["model_hyper_params"]["estimator__activation"]) + "_" + str(input_dict["model_hyper_params"]["cohort_retention_rate_dict"]) + "_" + str(input_dict["model_hyper_params"]["general_adjusting_square_factor"]) + "_" + str(input_dict["model_hyper_params"]["epochs"]) + "_" + str(input_dict["model_hyper_params"]["lookbacks"]) + "_" + str(input_dict["model_hyper_params"]["shuffle_fit"]) + "_" + str(input_dict["model_hyper_params"]["K_fold_splits"]) + "_" + str(pred_steps_ahead) + "_" + str(input_dict["model_hyper_params"]["estimator__alpha"]) + "_" + str(input_dict["model_hyper_params"]["general_adjusting_square_factor"]) + "_" + str(input_dict["model_hyper_params"]["lookbacks"]) + "_" + str(input_dict["model_hyper_params"]["batch_ratio"]) + "_" + str(input_dict["model_hyper_params"]["scaler_cat"]) + "_" + str(input_dict["model_hyper_params"]["estimator__hidden_layer_sizes"])
    if input_dict["model_hyper_params"]["early_stopping"] != 3 or input_dict["model_hyper_params"]["learning_rate"] != 0.001 or input_dict["model_hyper_params"]["testing_scoring"] != "mae" or input_dict["model_hyper_params"]["epochs"] != 1:
        predictor_hash += "_" + str(input_dict["model_hyper_params"]["early_stopping"]) + "_" + str(input_dict["model_hyper_params"]["learning_rate"]) + "_" + str(input_dict["model_hyper_params"]["testing_scoring"])  + "_" + str(input_dict["model_hyper_params"]["epochs"])
    name = name + predictor_hash
    return str(name)
    

def return_ticker_code_1(filename):
    return filename[:filename.index(".")]

def fg_timer(curr_iter_num, total_iter_num, callback_counts, task_name="", start_time=None):
    if ((curr_iter_num) % max(1,int(total_iter_num / callback_counts)) == 0 and curr_iter_num > 0) or curr_iter_num == total_iter_num - 1:
        proportion_done = (curr_iter_num + 1) / total_iter_num
        output_string = ""
        if not task_name=="":
            output_string += task_name + " : "
        output_string += str(proportion_done) + " : "
        output_string += str(curr_iter_num) + "/" + str(total_iter_num) + " "
        output_string += datetime.now().strftime("%H:%M:%S")
        if not start_time==None:
            time_past = datetime.now() - start_time
            output_string += " : etc: " + (datetime.now() + (((1 - proportion_done) / proportion_done) * time_past)).strftime("%H:%M:%S")
        print(output_string)
        
    return curr_iter_num + 1

def average_list_of_identical_dicts(dict_list, prime_dict=None):
    #values prep
    output = dict()
    if prime_dict==None:
        prime_dict = dict_list[0]
    
    for key in prime_dict.keys():
        key_sum     = 0
        key_count   = 0
        key_subdict = dict()
        
        if isinstance(prime_dict[key], dict):
            new_dict_list = reshape_list_of_dicts(dict_list, key)
            output[key] = average_list_of_identical_dicts(new_dict_list, prime_dict[key])
        elif isinstance(prime_dict[key], float) or isinstance(prime_dict[key], int):
            for dict_list_exponent in dict_list:
                key_sum += dict_list_exponent[key]
            output[key] = key_sum / len(dict_list)
        else:
            raise ValueError("unexpected value type within dict, type:{}, print:{}".format(str(type(prime_dict[key])), str(prime_dict[key])))
            
    return output
            
def reshape_list_of_dicts(dict_list, key):
    #takes a list of identically nested dicts and returns a list of identical subdicts within the original dict
    output_list = []
    for x in dict_list:
        output_list += [x[key]]
    return output_list

def compare_dicts(dict1, dict2):
    # Find keys that are common to both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Find keys that are unique to each dictionary
    unique_keys_dict1 = set(dict1.keys()) - set(dict2.keys())
    unique_keys_dict2 = set(dict2.keys()) - set(dict1.keys())

    # Find key-value pairs with different values
    different_values = {key: (dict1[key], dict2[key]) for key in common_keys if dict1[key] != dict2[key]}

    # Combine all differences into a summary dictionary
    differences = {
        'common_keys': common_keys,
        'unique_keys_dict1': unique_keys_dict1,
        'unique_keys_dict2': unique_keys_dict2,
        'different_values': different_values
    }

    return differences



#%% SubModule – Stock Market Data Prep 

def import_financial_data(
        target_file_path=None, 
        input_cols_to_include_list=[],
        temporal_params_dict=None, training_or_testing="training"):
    #set the import period
    if training_or_testing=="training" or training_or_testing=="train":
        period_start = temporal_params_dict["train_period_start"]
        period_end   = temporal_params_dict["train_period_end"]
    elif training_or_testing=="testing" or training_or_testing=="test":
        period_start = temporal_params_dict["test_period_start"]
        period_end   = temporal_params_dict["test_period_end"]
    else:
        raise ValueError("value " + str(training_or_testing) + " for 'training_or_testing' input not recognised")
    
    #format the data
    
    if target_file_path[-3:] == "txt":
        columns = ["date", "open", "high", "low", "close", "volume"]
        df_financial_data = pd.read_csv(target_file_path, header=None, names=columns, parse_dates=[0])
    elif target_file_path[-3:] == "csv":
        df_financial_data = pd.read_csv(target_file_path)
        df_financial_data["date"] = df_financial_data["date"].str.replace("T", " ")
        df_financial_data["date"] = df_financial_data["date"].str.replace("Z", "")
        df_financial_data["date"] = df_financial_data["date"].str.replace(".000", "")
        df_financial_data["date"] = pd.to_datetime(df_financial_data["date"], format='%Y-%m-%d %H:%M:%S')
    df_financial_data.set_index("date", inplace=True)
    df_financial_data.index.names = ['datetime']
    index = list(df_financial_data.index)
    time_step = index[1] - index[0]
    
    
    #check for faulty time windows
    if temporal_params_dict["train_period_start"] < min(index) - time_step: #timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["test_period_start"] < min(index)  - time_step: #timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["train_period_end"] > max(index)   + time_step and (training_or_testing=="test" or training_or_testing=="testing"): #timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["test_period_end"] > max(index)    + time_step and (training_or_testing=="test" or training_or_testing=="testing"): #timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    
    ##check for the wrong timestep
    #if not index[1] - index[0] - timedelta(seconds=5*60) == timedelta(0):
    #    raise ValueError("the financial data provided doesn't cover the experiment time window")
    
    #trim for time window
    mask_a = pd.to_datetime(index) > period_start
    mask_b = pd.to_datetime(index) < period_end
    mask = mask_a & mask_b
    df_financial_data = df_financial_data[mask]
    index = list(df_financial_data.index)
    
    #remove or annotate columns
    temp_list = df_financial_data.columns
    for col in temp_list:
        if not col in input_cols_to_include_list:
            df_financial_data = df_financial_data.drop(col, axis=1)
        else:
            df_financial_data = df_financial_data.rename(columns={col: '£_'+col})
        
    return df_financial_data

def return_technical_indicators_name(df, tech_indi_dict, match_doji, target_file_path):
    file_name = os.path.split(target_file_path)[1].split(".")[0]
    file_string = "fn_" + file_name + "_" + min(df.index).strftime('%d%m%y') + "to" + max(df.index).strftime('%d%m%y')
    for key in tech_indi_dict:
        file_string = file_string + "_" + key
        for val in tech_indi_dict[key]:
            if type(val) == float:
                val = int(val)
            file_string = file_string + "_" + str(val)
    file_string = file_string + "_" + str(match_doji)
    return file_string

def rescale_financial_data_if_needed(df_fin_data, format):
    
    if format == "day_scaled" or format == "delta_scaled":
        #determine the distance between days
        df_fin_data['normalizing_column'] = df_fin_data.groupby(df_fin_data.index.day)['£_open'].transform(lambda x: x.iloc[0])
        df_fin_data['normalizing_column'] = df_fin_data['normalizing_column'].astype(float)
        for col in df_fin_data.columns:
            if not col == "normalizing_column" and not  col == "£_volume":
                df_fin_data[col] = df_fin_data[col].astype(float)
                df_fin_data[col] = df_fin_data[col] / df_fin_data["normalizing_column"]
        df_fin_data = df_fin_data.drop('normalizing_column', axis=1)
    if format == "delta_scaled":
        df_fin_data = df_fin_data.diff().fillna(0)

    return df_fin_data

import pandas_ta as ta
from datetime import datetime
import os
from dataclasses import dataclass

@dataclass
class Quote:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def retrieve_or_generate_then_populate_technical_indicators(df, tech_indi_dict, match_doji, target_file_path, fin_data_scale):
    global global_strptime_str_filename

    quotes_list = [
        Quote(d, o, h, l, c, v)
        for d, o, h, l, c, v
        in zip(df.index, df['£_open'], df['£_high'], df['£_low'], df['£_close'], df['£_volume'])
    ]

    technical_indicators_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["technical_indicators"]
    technical_indicators_name = return_technical_indicators_name(df, tech_indi_dict, match_doji, target_file_path)
    technical_indicators_location_file = technical_indicators_folder_location_string + technical_indicators_name + ".csv"

    if os.path.exists(technical_indicators_location_file):
        df = pd.read_csv(technical_indicators_location_file)
        df.set_index(df.columns[0], inplace=True)
        df.index.name = "datetime"
        df.index = pd.to_datetime(df.index)
    else:
        for key in tech_indi_dict:
            for value in tech_indi_dict[key]:
                # calculate indicators
                if key == "sma":
                    df[f'$_sma_{value}'] = ta.sma(df['£_close'], length=int(value))
                elif key == "ema":
                    df[f'$_ema_{value}'] = ta.ema(df['£_close'], length=int(value))
                elif key == "macd":
                    if not len(value) == 3:
                        raise ValueError("the entries for macd, must be lists of a 3 value length")
                    temp_val = ta.macd(df['£_close'], fast=int(value[0]), slow=int(value[1]), signal=int(value[2]))
                    df[temp_val.columns] = temp_val
                elif key == "BollingerBands":
                    if not len(value) == 2:
                        raise ValueError("the entries for BollingerBands, must be lists of a 2 value length")
                    temp_val = ta.bbands(df['£_close'], length=value[0], std=value[1])
                    df[temp_val.columns] = temp_val
                else:
                    raise ValueError("technical indicator " + key + " not programmed")
        df = df.dropna(axis=1, how='all')

        df.to_csv(technical_indicators_location_file)

    df = rescale_financial_data_if_needed(df, fin_data_scale)

    return df

#%% SubModule – Sentiment Data Prep

def retrieve_or_generate_sentiment_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training"):
    #desc
    
    #general parameters
    company_symbol                   = outputs_params_dict["output_symbol_indicators_tuple"][0]
    time_step_seconds                = temporal_params_dict["time_step_seconds"]
    topic_model_qty                  = senti_inputs_params_dict["topic_qty"]
    rel_lifetime                     = senti_inputs_params_dict["relative_lifetime"]
    rel_hlflfe                       = senti_inputs_params_dict["relative_halflife"]
    topic_model_alpha                = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed              = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    apply_IDF                        = senti_inputs_params_dict["apply_IDF"]
    enforced_topic_model_nested_list = senti_inputs_params_dict["enforced_topics_dict"]
    new_combined_stopwords_inc       = senti_inputs_params_dict["inc_new_combined_stopwords_list"]
    topic_weight_square_factor       = senti_inputs_params_dict["topic_weight_square_factor"]
    factor_tweet_attention           = senti_inputs_params_dict["factor_tweet_attention"]
    factor_topic_volume              = senti_inputs_params_dict["factor_topic_volume"]
    
    #set period start/ends
    if training_or_testing == "training" or training_or_testing == "train":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        train_period_start  = temporal_params_dict["test_period_start"]
        train_period_end    = temporal_params_dict["test_period_end"]
    else:
        raise ValueError("the input " + str(training_or_testing) + " is wrong for the input training_or_testing")
       
    #method
    #search for predictor
    sentiment_data_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["sentiment_data"]
    sentiment_data_name = return_sentiment_data_name(company_symbol, train_period_start, train_period_end, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe, factor_tweet_attention, factor_topic_volume)
    sentiment_data_location_file = sentiment_data_folder_location_string + sentiment_data_name + ".csv"
    if os.path.exists(sentiment_data_location_file):
        df_sentiment_data = pd.read_csv(sentiment_data_location_file)
        df_sentiment_data.set_index(df_sentiment_data.columns[0], inplace=True)
        df_sentiment_data.index.name = "datetime"
    elif not os.path.exists(sentiment_data_location_file) and topic_model_qty > 0:
        df_sentiment_data = generate_sentiment_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing=training_or_testing)
        df_sentiment_data.to_csv(sentiment_data_location_file)
    else:
        df_sentiment_data = None
    return df_sentiment_data

def retrieve_sentiment_data():
    raise ValueError('needs writing') 
    return None



def generate_sentiment_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training", hardcode_df_annotated_tweets=None):
    
    #general parameters
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    num_topics          = senti_inputs_params_dict["topic_qty"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    relavance_lifetime      = senti_inputs_params_dict["relative_lifetime"]
    relavance_halflife      = senti_inputs_params_dict["relative_halflife"]
    apply_IDF               = senti_inputs_params_dict["apply_IDF"]
    new_combined_stopwords_inc          = senti_inputs_params_dict["inc_new_combined_stopwords_list"]
    enforced_topic_model_nested_list    = senti_inputs_params_dict["enforced_topics_dict"]
    topic_weight_square_factor          = senti_inputs_params_dict["topic_weight_square_factor"]
    factor_tweet_attention              = senti_inputs_params_dict["factor_tweet_attention"]
    epoch_time                          = datetime(1970, 1, 1)
    factor_topic_volume                 = senti_inputs_params_dict["factor_topic_volume"]

    if training_or_testing == "training" or training_or_testing == "train":
        period_start  = temporal_params_dict["train_period_start"]
        period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        period_start  = temporal_params_dict["test_period_start"]
        period_end    = temporal_params_dict["test_period_end"]
    else:
        global global_error_str_1
        raise ValueError(global_error_str_1.format(str(training_or_testing)))
    
    #search for annotated tweets
    annotated_tweets_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["annotated_tweets"]
    annotated_tweets_name = return_annotated_tweets_name(company_symbol, period_start, period_end, num_topics, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor)
    annotated_tweets_location_file = annotated_tweets_folder_location_string + annotated_tweets_name + ".csv"
    print("approaching gate")
    if isinstance(hardcode_df_annotated_tweets, pd.DataFrame):
        print("gate works")
        df_annotated_tweets = hardcode_df_annotated_tweets
        print(datetime.now().strftime("%H:%M:%S") + " - generating sentiment data")
    elif os.path.exists(annotated_tweets_location_file):
        df_annotated_tweets = pd.read_csv(annotated_tweets_location_file)
        df_annotated_tweets.set_index(df_annotated_tweets.columns[0], inplace=True)
        df_annotated_tweets.index.name = "datetime"
    else:
        df_annotated_tweets = generate_annotated_tweets(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing=training_or_testing)
        df_annotated_tweets.to_csv(annotated_tweets_location_file)
        print(datetime.now().strftime("%H:%M:%S") + " - generating sentiment data")
    
    # generate output dataFrame
    columns = ["tweet_cohort_start_post", "tweet_cohort_end_post"]
    target_columns = []
    if senti_inputs_params_dict["factor_topic_volume"] != global_exclusively_str:
        #columns += ["~sent_topic_t{}".format(i) for i in range(num_topics)]
        target_columns += ["~sent_topic_t{}".format(i) for i in range(num_topics)]
    if senti_inputs_params_dict["factor_topic_volume"] != False:
        #columns += ["~sent_topic_W{}".format(i) for i in range(num_topics)]
        target_columns += ["~sent_topic_W{}".format(i) for i in range(num_topics)]
    df_sentiment_scores = pd.DataFrame(index=index, columns=columns)
    df_sentiment_scores["tweet_cohort_start_post"]    = df_sentiment_scores.index
    df_sentiment_scores["tweet_cohort_end_post"]    = df_sentiment_scores["tweet_cohort_start_post"].apply(lambda datetime: ((datetime - epoch_time)).total_seconds())
    df_sentiment_scores["tweet_cohort_start_post"]    = df_sentiment_scores["tweet_cohort_start_post"].apply(lambda datetime: ((datetime - epoch_time) - timedelta(seconds=relavance_lifetime)).total_seconds())
    
    # prep input data
    if factor_tweet_attention == False:
        df_annotated_tweets["tweet_attention_score"] = 1
    elif factor_tweet_attention == True:
        df_annotated_tweets["tweet_attention_score"] = (0.2 + (df_annotated_tweets["comment_num"] + 0.2) * (df_annotated_tweets["retweet_num"] + 0.2) * (df_annotated_tweets["like_num"] + 0.2))
    else:
        raise ValueError("this value must be included")
    df_annotated_tweets = df_annotated_tweets.drop(['Unnamed: 0', 'tweet_id', 'writer', 'body', 'comment_num', 'retweet_num', 'like_num'], axis=1)
    

    # define internal methods
    def return_tweet_cohort_from_scratch(tweet_cohort_start_post, tweet_cohort_end_post, df_annotated_tweets=df_annotated_tweets):
        mask = (df_annotated_tweets["post_date"] >= tweet_cohort_start_post) & (df_annotated_tweets["post_date"] <= tweet_cohort_end_post)
        tweet_cohort = df_annotated_tweets[mask]
        return tweet_cohort

    def return_topic_sentiment_and_or_volume(row, df_annotated_tweets, num_topics, relavance_halflife, factor_topic_volume):
        senti_col_name = "~sent_topic_W{}"
        tweet_cohort_start_post = row["tweet_cohort_start_post"]
        tweet_cohort_end_post = row["tweet_cohort_end_post"]
        tweet_cohort = return_tweet_cohort_from_scratch(tweet_cohort_start_post, tweet_cohort_end_post)

        topic_scores = []
        senti_scores = []
        time_weight = (0.5 ** ((tweet_cohort["post_date"] - tweet_cohort_start_post) / relavance_halflife)) * tweet_cohort["tweet_attention_score"]
        topic_weights = tweet_cohort.loc[:,senti_col_name.format(0):senti_col_name.format(num_topics-1)].mul(time_weight, axis=0)
        
        if factor_topic_volume != False:
            topic_scores = topic_weights.sum() / time_weight.sum()
            topic_scores = list(topic_scores.values)
    
        if factor_topic_volume != global_exclusively_str:
            score_numer = topic_weights.mul(tweet_cohort["~sent_overall"], axis=0)
            score_denom = topic_weights
            senti_scores = score_numer.sum() / score_denom.sum()
            senti_scores = list(senti_scores.values)
        
        if len(senti_scores + topic_scores) % num_topics != 0:
            raise ValueError("gggg")

        return pd.Series(senti_scores + topic_scores, index=target_columns) #list(senti_scores.values) + list(topic_weights.values)

    df_sentiment_scores[target_columns] = df_sentiment_scores.apply(lambda row: return_topic_sentiment_and_or_volume(row, df_annotated_tweets, num_topics, relavance_halflife, factor_topic_volume), axis=1)
    df_sentiment_scores = df_sentiment_scores[target_columns]
    return df_sentiment_scores

def generate_annotated_tweets(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, repeat_timer=10, training_or_testing="training", hardcode_location_for_topic_model="", k_fold_textual_sources=None):
    
    
    #general parameters
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    num_topics          = senti_inputs_params_dict["topic_qty"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    relavance_lifetime  = senti_inputs_params_dict["relative_lifetime"]
    apply_IDF           = senti_inputs_params_dict["apply_IDF"]
    sentiment_method    = senti_inputs_params_dict["sentiment_method"]
    enforced_topic_model_nested_list = senti_inputs_params_dict["enforced_topics_dict"]
    inc_new_combined_stopwords_list = senti_inputs_params_dict["inc_new_combined_stopwords_list"]
    topic_weight_square_factor = senti_inputs_params_dict["topic_weight_square_factor"]
        
    if training_or_testing == "training" or training_or_testing == "train":
        period_start  = temporal_params_dict["train_period_start"]
        period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        period_start  = temporal_params_dict["test_period_start"]
        period_end    = temporal_params_dict["test_period_end"]
    else:
        global global_error_str_1
        raise ValueError(global_error_str_1.format(str(training_or_testing)))
    
    
    
    #search for topic_model
    topic_model_folder_folder = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["topic_models"]
    
    topic_model_name = return_topic_model_name(num_topics, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, inc_new_combined_stopwords_list)
    if not hardcode_location_for_topic_model=="":
        with open(hardcode_location_for_topic_model, 'rb') as file:
            topic_model_dict = pickle.load(file)
    elif os.path.exists(topic_model_folder_folder + "topic_model_dict_" + topic_model_name + ".pkl"):
        with open(topic_model_folder_folder + "topic_model_dict_" + topic_model_name + ".pkl", "rb") as file:
            topic_model_dict = pickle.load(file)
    else:
        wordcloud, topic_model_dict, visualisation = generate_and_save_topic_model(topic_model_name, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, inc_new_combined_stopwords_list=inc_new_combined_stopwords_list)
  
    
    #generate annotated tweets
    print(datetime.now().strftime("%H:%M:%S") + " - generating annotated tweets")
    df_annotated_tweets   = import_twitter_data_period(senti_inputs_params_dict["tweet_file_location"], period_start, period_end, relavance_lifetime, tweet_ratio_removed, k_fold_textual_sources=k_fold_textual_sources)
    
    sentiment_col_strs = []
    for num in range(num_topics):
        sentiment_col_strs = sentiment_col_strs + ["~sent_topic_W" + str(num)]
    
    columns_to_add = ["~sent_overall"]
    for num in range(num_topics):
        columns_to_add = columns_to_add + ["~sent_topic_W" + str(num)]
    
    df_annotated_tweets[columns_to_add] = float("nan")
    count = 0; counter_len = len(df_annotated_tweets.index) # FG_Counter
    print("annotating {} tweets/news articles".format(len(df_annotated_tweets.index)))
    start_time          = datetime.now()
    
    sentiment_scores = df_annotated_tweets['body'].apply(lambda text: sentiment_method.polarity_scores(text)['compound'])
    # Calculate topic weights for all tweets
    topic_weights = df_annotated_tweets['body'].apply(lambda text: return_topic_weight(text, topic_model_dict, num_topics))
    topic_weights = topic_weights.apply(lambda topic_tuples: [t[1] for t in topic_tuples] if len(topic_tuples) == num_topics else list(np.zeros(num_topics)))
    if topic_weight_square_factor != 1:
        topic_weights = topic_weights.apply(lambda x: adjust_topic_weights(x, topic_weight_square_factor))

    # Combine sentiment and topic weights
    topic_weights_prepped = pd.DataFrame(topic_weights.tolist(), index=sentiment_scores.index, columns=sentiment_col_strs)
    sentiment_scores.name = "~sent_overall"
    sentiment_analysis = pd.concat([sentiment_scores, topic_weights_prepped], axis=1, ignore_index=False)

    # Add the sentiment and topic weight columns to the DataFrame
    df_annotated_tweets[sentiment_analysis.columns] = sentiment_analysis
    
    return df_annotated_tweets

def adjust_topic_weights(lst, factor):
    a = [x ** factor for x in lst]
    return [x / sum(a) for x in a]

def return_topic_weight(text_body, topic_model_dict, num_topics):
    if num_topics > 1:
        id2word = topic_model_dict["id2word"]
        lda_model = topic_model_dict["lda_model"]
        bow_doc = id2word.doc2bow(text_body.split(" "))
        doc_topics = lda_model.get_document_topics(bow_doc)
    else:
        doc_topics = [(0,1)]
    return doc_topics

def generate_and_save_topic_model(run_name, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, inc_new_combined_stopwords_list=False):
    folder_path = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["topic_models"]
    file_location_wordcloud         = folder_path + "wordcloud_" + run_name + ".png"
    file_location_topic_model_dict  = folder_path + "topic_model_dict_" + run_name + ".pkl"
    file_location_visualisation     = folder_path + "visualisation_" + run_name + '.html'
    tweet_file_location             = senti_inputs_params_dict["tweet_file_location"]
    enforced_topics_dict            = senti_inputs_params_dict["enforced_topics_dict"]
    enforced_topics_dict_name       = senti_inputs_params_dict["enforced_topics_dict_name"]
    num_topics                      = senti_inputs_params_dict["topic_qty"]
    topic_model_alpha               = senti_inputs_params_dict["topic_model_alpha"]
    train_period_start              = temporal_params_dict["train_period_start"]
    train_period_end                = temporal_params_dict["train_period_end"]
    relavance_lifetime              = senti_inputs_params_dict["relative_lifetime"]
    tweet_ratio_removed             = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    apply_IDF                       = senti_inputs_params_dict["apply_IDF"]
    reclean_tweets                  = senti_inputs_params_dict["regenerate_cleaned_tweets_for_subject_discovery"]
    
    print(datetime.now().strftime("%H:%M:%S") + " - generating topic model")
    print("-------------------------------- Importing Sentiment Data --------------------------------")
    df_prepped_tweets                           = import_twitter_data_period(tweet_file_location, train_period_start, train_period_end, relavance_lifetime, tweet_ratio_removed)
    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Prepping Sentiment Data --------------------------------")


    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Creating Subject Keys --------------------------------")
    wordcloud, topic_model_dict, visualisation  = return_subject_keys(df_prepped_tweets, topic_qty = num_topics, topic_model_alpha=topic_model_alpha, apply_IDF=apply_IDF,
                                                                      enforced_topics_dict=enforced_topics_dict, return_LDA_model=True, return_png_visualisation=True, return_html_visualisation=False)
    save_topic_clusters(wordcloud, topic_model_dict, None, file_location_wordcloud, file_location_topic_model_dict, file_location_visualisation)
    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Completed Subject Keys --------------------------------")
    print(datetime.now().strftime("%H:%M:%S"))
    return wordcloud, topic_model_dict, visualisation

def import_twitter_data_period(target_file, period_start, period_end, relavance_lifetime, tweet_ratio_removed, k_fold_textual_sources=None):
    #prep data
    
    if target_file[-4:] == ".pkl": #news articles detacted, change function
        input_df = import_and_prep_news_articles_as_tweets(target_file, tweet_ratio_removed, k_fold_textual_sources=k_fold_textual_sources)
    else:
        input_df = pd.read_csv(target_file)
    
    epoch_time  = datetime(1970, 1, 1)
    period_start -= timedelta(seconds=relavance_lifetime)
    epoch_start = (period_start - epoch_time).total_seconds()
    epoch_end   = (period_end - epoch_time).total_seconds()
    
    #trim according to time window    
    input_df = input_df[input_df["post_date"]>epoch_start]
    input_df = input_df[input_df["post_date"]<epoch_end]
    
    if tweet_ratio_removed > 1 and not target_file[-4:] == ".pkl":
        new_index = input_df.index
        if k_fold_textual_sources != None:
            new_index = new_index[k_fold_textual_sources:]
        new_index = new_index[::tweet_ratio_removed]
        input_df = input_df.loc[new_index]
    
    return input_df

def import_and_prep_news_articles_as_tweets(target_file, tweet_ratio_removed, k_fold_textual_sources=None):
    with open(target_file, 'rb') as file:
        news_rich_format = pickle.load(file)

    
    if tweet_ratio_removed > 1:
        if k_fold_textual_sources != None:
            news_rich_format = news_rich_format[k_fold_textual_sources:]
        news_rich_format = news_rich_format[::tweet_ratio_removed]

    print("dd-d")
    data = [[int((article["publish_datetime"].replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()), article["text"]] for article in news_rich_format]
    
    return pd.DataFrame(data, columns=["post_date", "body"])    

from datetime import datetime, timedelta
def convert_epoch_time_to_datetime_string(post_date):
    period = datetime(1970, 1, 1) + timedelta(seconds=post_date)
    return period.strftime('%d/%m/%y %H:%M:%S')

def convert_datetime_to_epoch_time(input_datetime):
    return int((input_datetime - datetime(1970, 1, 1)).total_seconds())

from datetime import datetime, timedelta
def return_time(post_date):
    period = datetime(1970, 1, 1) + timedelta(seconds=post_date)
    return period.strftime('%d/%m/%y %H:%M:%S')


def prep_twitter_text_for_subject_discovery(input_list, df_stocks_list_file=None,
                                                     make_company_agnostic=False , inc_new_combined_stopwords_list=True):
    global global_precalculated_assets_locations_dict, combined_stop_words_list_file_path, custom_stop_words_list_file_path
    global global_df_stocks_list_file
    
    from config import global_combined_stopwords_list_path
    #prep parameters
    death_characters    = ["$", "amazon", "apple", "goog", "tesla", "http", "@", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "compile", "www","(", ")"]

    stopwords_english   = stopwords.words('english')
    if inc_new_combined_stopwords_list == True:
        with open(global_combined_stopwords_list_path, 'r') as file:
            file_content = file.read()
            stopwords_english = stopwords_english + [s.strip(' "\'') for s in file_content.split(",")]
        #with open(custom_stop_words_list_file_path, 'r') as file2:
        #    file_content2 = file2.read()
        #    stopwords_english = stopwords_english + [s.strip(' "\'') for s in file_content2.split(",")]

        
        
        print("english stopwords are now long: " + str(len(stopwords_english)))
        
    #these are words are removed from company names to create additional shortened versions of those names. This is so these version can be eliminated from the tweets to make the subjects agnostic
    corp_stopwords      = [".com", "company", "corp", "froup", "fund", "gmbh", "global", "incorporated", "inc.", "inc", "tech", "technology", "technologies", "trust", "limited", "lmt", "ltd"]
    #these are words are directly removed from tweets
    misc_stopwords      = ["iphone", "airpods", "jeff", "bezos", "#microsoft", "#amzn", "volkswagen", "microsoft", "amazon's", "tsla", "androidwear", "ipad", "amzn", "iphone", "tesla", "TSLA", "elon", "musk", "baird", "robert", "pm", "androidwear", "android", "robert", "ab", "ae", "dlvrit", "https", "iphone", "inc", "new", "dlvrit", "py", "twitter", "cityfalconcom", "aapl", "ing", "ios", "samsung", "ipad", "phones", "cityfalconcom", "us", "bitly", "utmmpaign", "etexclusivecom", "cityfalcon", "owler", "com", "stock", "stocks", "buy", "bitly", "dlvrit", "alexa", "zprio", "billion", "seekalphacom", "et", "alphet", "seekalpha", "googl", "zprio", "trad", "jt", "windows", "adw", "ifttt", "ihadvfn", "nmona", "pphppid", "st", "bza", "twits", "biness", "tim", "ba", "se", "rat", "article"]

    #prep stocks_list_shortened

    
    if make_company_agnostic==False:
        #death_characters        = 
        stocks_list             = []
        tickers_list            = []
        #stopwords_english       = 
        #corp_stopwords          = 
        #misc_stopwords          = 
        stocks_list_shortened   = [] 
    else:
        stocks_list         = list(df_stocks_list_file["Name"].map(lambda x: x.lower()).values)
        tickers_list        = list(df_stocks_list_file["Ticker"].map(lambda x: x.lower()).values)
        stocks_list_shortened_dict  = update_shortened_company_file(global_df_stocks_list_file, corp_stopwords)
        stocks_list_shortened       = list(stocks_list_shortened_dict.values())

    
    #prep variables
    split_tweets = []
    output = []
    for tweet in input_list:
        split_tweet_pre = tweet.split(" ")
        split_tweet = []
        for word in split_tweet_pre:
            #split_tweet = split_tweet + [" " + word.lower() + " "]
            split_tweet = split_tweet + [word.lower()]
        split_tweets = split_tweets + [split_tweet]

    #removal of words
    pos = 0
    
    for tweet in split_tweets:
        #for word in reversed(tweet):
        copy_of_tweet = tweet.copy()
        for word in copy_of_tweet:
            Removed = False
            # remove words containing "x"
            for char in death_characters:
                if char in word:
                    tweet.remove(word)
                    Removed = True
                    break
            if Removed == False:
                for char in tickers_list + stopwords_english + misc_stopwords + list(reversed(stocks_list)) + list(reversed(stocks_list_shortened)): #corp_stopwords + 
                    if char == word:
                        tweet.remove(word)
                        S = False
                        break
            # remove words equalling stop words
        pos += 1
        
        
    #finalise and remove stock names
    output = []
    #iteration_list = list(reversed(stocks_list)) + list(reversed(stocks_list_shortened))
    for split_tweet in split_tweets:
        #recombined_tweet = list(map(lambda x: x.strip(), split_tweet))
        recombined_tweet = " ".join(split_tweet)#.replace("  "," ")
        #recombined_tweet = " ".join(recombined_tweet)#.replace("  "," ")
        #for stock_name in iteration_list:
        #    recombined_tweet = recombined_tweet.replace(stock_name, "")
        output = output + [recombined_tweet]
    
    df_output = pd.DataFrame(output, columns=["body"])
    #df_output.to_csv(save_location)
    #if not save_location == None:
    #    with open(save_location, "wb") as file:
    #        pickle.dump(output, file)
    
    return output



def update_shortened_company_file(df_stocks_list_file, corp_stopwords, file_location=None):
    stocks_list         = list(df_stocks_list_file["Name"].map(lambda x: x.lower()).values)
            
    stocks_list_shortened_dict = dict()
    for stock_name in stocks_list:
        shortened = False
        stock_name_split = stock_name.split(" ")
        for word in reversed(stock_name_split):
            for stop in corp_stopwords:
                if stop == word:
                    stock_name_split.remove(word)
                    shortened = True
        if shortened == True:
            stocks_list_shortened_dict[stock_name] = " ".join(stock_name_split)
    
    return stocks_list_shortened_dict

def return_initial_eta(num_topics, initial_topics_dict, id2word):
    eta = np.ones((num_topics, len(id2word.id2token))) * 0.5
    not_found_words_list = []
    for topic_num in range(min(num_topics,len(initial_topics_dict))):
        for enforced_word_str in initial_topics_dict[topic_num]:
            if enforced_word_str in id2word.token2id.keys():
                enforced_word_id = id2word.token2id[enforced_word_str]
                # enforce word for topic and weaken for others
                for topic_num_2 in range(num_topics):
                    if topic_num_2 == topic_num:
                        eta[topic_num_2][enforced_word_id] = 0.9
                    else: 
                        eta[topic_num_2][enforced_word_id] = 0.1
            else:
                not_found_words_list = not_found_words_list + [enforced_word_str]
    print("{} words not found: {}".format(str(len(not_found_words_list)), str(not_found_words_list)))
    return eta



def return_subject_keys(df_tweets, topic_qty=10, enforced_topics_dict=None, stock_names_list=None, words_to_remove=None, 
                        return_LDA_model=True, return_png_visualisation=False, return_html_visualisation=False, 
                        topic_model_alpha=0.1, apply_IDF=True, cores=2, passes=60, iterations=800, return_perplexity=False):
    output = []

    tweets_to_sample = 750
    import re
    def count_words(tweet):
        words = re.findall(r'\b\w+\b', tweet)
        return len(words)
    long_tweets = df_tweets[df_tweets['body'].apply(lambda tweet: count_words(tweet) > 15)]
    selected_tweets = long_tweets.sample(n=min(min(tweets_to_sample, len(long_tweets)-1), len(long_tweets)))
    
    selected_tweets = prep_twitter_text_for_subject_discovery(list(selected_tweets["body"]))


    data = selected_tweets
    
    print("topic model trained from {} tweets".format(len(data)))
    data_words = list(sent_to_words(data))
    
    if return_LDA_model < return_html_visualisation:
        raise ValueError("You must return the LDA model if you return the LDA visualisation")

    if return_png_visualisation:
        # Create a long string for word cloud visualization
        long_string = "start"
        for w in data_words:
            long_string = long_string + ',' + ','.join(w)
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
        #wordcloud.generate(long_string)
        #wordcloud.to_image()
        output = output + [None]
    else:
        output = output + [None]

    if return_LDA_model:
        if return_perplexity == True:
            # Split the data into training and test sets
            data_train, data_test = train_test_split(data_words, test_size=0.1, random_state=42)
            texts_test = data_test
        else:
            data_train = data_words

        # Create Dictionary
        id2word = corpora.Dictionary(data_train)

        # Create Corpus
        texts_train = data_train
        

        # Term Document Frequency
        corpus_train = [id2word.doc2bow(text) for text in texts_train]
        if return_perplexity == True:
            corpus_test = [id2word.doc2bow(text) for text in texts_test]

        # Translate the enforced_topics_dict input
        if enforced_topics_dict is not None:
            if len(id2word.id2token) == 0:
                # Manual transfer to id2token, required.
                for key in id2word.token2id:
                    id2word.id2token[id2word.token2id[key]] = key
            eta = return_initial_eta(topic_qty, enforced_topics_dict, id2word)
        else:
            eta = 'auto'  # Let Gensim estimate eta (topic-word distribution)

        # Apply IDF
        if apply_IDF:
            # Create tfidf model
            tfidf = TfidfModel(corpus_train)

            # Apply tfidf to both training and test corpora
            corpus_train = tfidf[corpus_train]
            if return_perplexity == True:
                corpus_test = tfidf[corpus_test]

        # Build LDA model on the training data
        if passes == None:
            print("LDA gate works")
            lda_model = gensim.models.LdaModel(corpus=corpus_train,
                                                id2word=id2word,
                                                num_topics=topic_qty,
                                                alpha=topic_model_alpha,
                                                eta=eta,
                                                iterations=iterations,
                                                random_state=42)
        else:
            lda_model = gensim.models.LdaModel(corpus=corpus_train,
                                                id2word=id2word,
                                                num_topics=topic_qty,
                                                alpha=topic_model_alpha,
                                                eta=eta,
                                                passes=passes,
                                                iterations=iterations,
                                                random_state=42)
        
        if return_perplexity == True:
            # Calculate perplexity on the test data
            perplexity = lda_model.log_perplexity(corpus_test)

        # Print the Keyword in the 10 topics
        # pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus_train]
        topic_model_dict = {"lda_model": lda_model, "doc_lda": doc_lda, "corpus": corpus_train, "id2word": id2word}
        output = output + [topic_model_dict]

        if return_perplexity:
            output = output + [perplexity]
        
    else:
        output = output + [None]

    if return_html_visualisation:
        pyLDAvis.enable_notebook
        if topic_qty > 1:
            
            LDAvis_prepared = gensimvis.prepare(lda_model, corpus_train, id2word)
        else:
            LDAvis_prepared = None
        output = output + [LDAvis_prepared]
    else:
        output = output + [None]
    print("topic model complete with {} passes".format(str(lda_model.passes)))
    return tuple(output)

def save_topic_clusters(wordcloud=None, topic_model_dict=None, visualisation=None, file_location_wordcloud=None, file_location_topic_model_dict=None, file_location_visualisation=None):
    if wordcloud != None:
        wordcloud.to_file(file_location_wordcloud)
    
    #"lda_model" : lda_model, "doc_lda" : doc_lda, "corpus" : corpus, "id2word" : id2word}
    if topic_model_dict != None:
        file_path = file_location_topic_model_dict
        #LDAvis_prepared = gensimvis.prepare(topic_model_dict["doc_lda"], topic_model_dict["corpus"], topic_model_dict["id2word"])
            
        with open(file_path, "wb") as file:
            pickle.dump(topic_model_dict, file)
    
    if visualisation != None:
        try:
            pyLDAvis.save_html(visualisation, file_location_visualisation)
        except:
            print("vis not made")
    
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



def update_tweet_cohort(df_annotated_tweets_temp, tweet_cohort_start, tweet_cohort_end):
    epoch_time          = datetime(1970, 1, 1)
    #delete old tweets
    tweet_cohort                = tweet_cohort[tweet_cohort["post_date"] >= tweet_cohort_start]
    
    #find new tweets
    new_tweets                  = df_annotated_tweets_temp[df_annotated_tweets_temp["post_date"] <= tweet_cohort_end]
    df_annotated_tweets_temp    = df_annotated_tweets_temp.drop(new_tweets.index)
    tweet_cohort                = pd.concat([tweet_cohort, new_tweets], axis=0)
    
    return tweet_cohort, df_annotated_tweets_temp

def generate_datetimes(train_period_start, train_period_end, seconds_per_time_steps):
    format_str = '%Y%m%d_%H%M%S'  # specify the desired format for the datetime strings
    current_time = train_period_start
    datetimes = []
    while current_time <= train_period_end:
        datetimes.append(current_time.strftime(format_str))
        current_time += timedelta(seconds=seconds_per_time_steps)
    return datetimes



#%% SubModule - Model Training

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
            #mid = int(0.5 * (stop - start)) + start
            mid = int(1.0 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def create_step_responces(df_financial_data, df_sentiment_data_input, pred_output_and_tickers_combos_list, pred_steps_ahead, financial_value_scaling):
    #this method populates each row with the next X output results, this is done so that, each time step can be trained
    #to predict the value of the next X steps
    
    new_col_str = "{}_{}"
    list_of_new_columns = []
    nan_values_replaced = 0
    train_test_split = 1
    if isinstance(df_sentiment_data_input, pd.DataFrame):
        df_sentiment_data = copy.copy(df_sentiment_data_input)
        df_sentiment_data.index = pd.to_datetime(df_sentiment_data.index)
        common_indices = df_financial_data.index.intersection(df_sentiment_data.index)
        df_sentiment_data = df_sentiment_data.loc[common_indices]
    else:
        df_sentiment_data = df_sentiment_data_input
    data = pd.concat([df_financial_data, df_sentiment_data], axis=1, ignore_index=False)

    
    #create regressors
    symbol, old_col = pred_output_and_tickers_combos_list
    old_col = "£_" + old_col
    
    if isinstance(pred_steps_ahead,list):
        raise ValueError("only runs with singluar pred_steps_list values are allow, no lists")
        
    new_col = new_col_str.format(old_col, pred_steps_ahead)
    list_of_new_columns = list_of_new_columns + [new_col]
    if financial_value_scaling == "delta_scaled":
        data[new_col] = data[old_col].rolling(window=pred_steps_ahead).sum()
        data[new_col] = data[new_col].shift(-pred_steps_ahead)
    elif financial_value_scaling == None or financial_value_scaling == "day_scaled":
        data[new_col] = data[old_col].shift(-pred_steps_ahead)
    else:
        raise ValueError("financial_value_scaling: {}, not recognised".format(financial_value_scaling))
    

    #split regressors and responses
    #Features = 6

    data = data.dropna(axis=1, how='all', inplace=False)
    data = data.dropna(inplace=False)

    X = copy.copy(data)
    y = copy.copy(data[list_of_new_columns])
    
    for col in list_of_new_columns:
        X = X.drop(col, axis=1)
    
    return X, y

#create model

def initiate_model(input_dict, hash_name=None):
    if input_dict["model_hyper_params"]["name"] == "RandomSubspace_RNN_Regressor":
        estimator = DRSLinRegRNN(base_estimator=Sequential(),
            input_dict = input_dict, hash_name=hash_name)
    else:
        raise ValueError("the model type: " + str(input_dict["model_hyper_params"]["name"]) + " was not found in the method")
    
    return estimator

def return_RNN_ensamble_estimator(model_hyper_params, global_random_state, n_features):
    
    ensemble_estimator = Sequential()
    
    for id, layer in enumerate(model_hyper_params["estimator__hidden_layer_sizes"]):
        # prep key word arguments
        kwargs = {
            "units" : layer[1], "activation" : model_hyper_params["estimator__activation"], "return_sequences" : True
        }
        if id == 0 or id == 1:
            kwargs["input_shape"] = (model_hyper_params["lookbacks"], n_features)
        if id == len(model_hyper_params["estimator__hidden_layer_sizes"]) - 1:
            kwargs["return_sequences"] = False
        kwargs["kernel_regularizer"]   = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])
        kwargs["bias_regularizer"]     = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])
        kwargs["activity_regularizer"] = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])    
        # add layer 
        if layer[0] != "simple":
            kwargs["recurrent_activation"] = "sigmoid"
            kwargs["recurrent_dropout"]= 0
            kwargs["unroll"] = False
            kwargs["use_bias"] = True

        if layer[0] == "simple":
            ensemble_estimator.add(SimpleRNN(**kwargs))            
        elif layer[0] == "GRU":
            ensemble_estimator.add(GRU(**kwargs))
        elif layer[0] == "LSTM":
            ensemble_estimator.add(LSTM(**kwargs))
        
    ensemble_estimator.add(Dense(units=1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=model_hyper_params["learning_rate"])
    ensemble_estimator.compile(optimizer=opt, loss=model_hyper_params["testing_scoring"])
    ensemble_estimator.random_state = global_random_state
    
    
    return ensemble_estimator


class DRSLinRegRNN():
    def __init__(self, base_estimator=None,#Sequential(),
                 input_dict=None, hash_name=None):
        #expected keys: training_time_splits, max_depth, max_features, random_state,        
        self.estimator_info_pack = {}
        for key in input_dict["model_hyper_params"]: #FG_action: ensure this is aligned
           setattr(self, key, input_dict["model_hyper_params"][key])
        self.model_hyper_params   = input_dict["model_hyper_params"]
        self.input_dict           = input_dict
        self.base_estimator       = base_estimator
        self.estimators_          = []
        if hash_name == None:
            raise ValueError("models must be named during creation now")
        else:
            self.name = hash_name


    def scale_output_according_to_input_scaler(self, X, Y, prediction_col_in_input="£_close"):
        # to ensure that the ouput uses the same scaling as the input
        original_col_string = Y.columns[0]
        Y[X.columns] = np.nan
        Y[prediction_col_in_input] = Y[original_col_string]
        Y = Y.drop(original_col_string, axis=1)
        Y = pd.DataFrame(self.scaler_X.transform(Y), index=Y.index, columns=Y.columns)
        Y[original_col_string] = Y[prediction_col_in_input]
        Y = pd.DataFrame(Y[original_col_string])
        return Y
    
    def inverse_scale_output_according_to_input_scaler(self, X, Y, prediction_col_in_input="£_close"):
        # to ensure that the ouput uses the same scaling as the input
        original_col_string = Y.columns[0]
        Y[X.columns] = np.nan
        Y[prediction_col_in_input] = Y[original_col_string]
        Y = Y.drop(original_col_string, axis=1)
        Y = pd.DataFrame(self.scaler_X.inverse_transform(Y), index=Y.index, columns=Y.columns)
        Y[original_col_string] = Y[prediction_col_in_input]
        Y = pd.DataFrame(Y[original_col_string])
        return Y

    def align_X_and_Y_for_fitting(self, X_indy, X, Y_indy, Y, pred_steps):
        # this method trims both X and Y so that they are aligned with a displacement equal to the prediction steps for the model.
        # in this process, X steps that dont have an appropriate following Y step (i.e in the next step) are deleted
        # it is assumed that the upstream processes are outputting duplicate X_indys and Y_indys. if this is not the case this 
        # will have to be adjusted within the model
        time_shift_secs = self.input_dict["temporal_params_dict"]["time_step_seconds"]

        if not (X_indy == Y_indy).all():
            raise validation_dict("expected these values to equal, this method needs to be adapted to accomodate this")
        if pred_steps != 0:
            raise validation_dict("pred steps expected to be zero")

        # y indy are adjusted to match thier corresponding x indexs (the time the prediction is made)
        Y_indy = Y_indy.copy()
        Y_indy -= timedelta(seconds=time_shift_secs)
        
        output_X = np.array([X[0]])
        output_Y = np.array([Y[0]])
        for i, X_ts in enumerate(X_indy):
            if X_ts in Y_indy:
                output_X = np.append(output_X, [X[i]], axis=0)
                output_Y = np.append(output_Y, [Y[i]], axis=0)
        output_X = output_X[1:]
        output_Y = output_Y[1:]
        return  output_X, output_Y       



    def return_single_component_model_fitted_with_early_stopping(self, model, X_input, Y_input, X_val_input, Y_val_input):
        verbose = 1
        X, Y, X_val, Y_val = copy.copy(X_input), copy.copy(Y_input), copy.copy(X_val_input), copy.copy(Y_val_input)
        

        X_indy, X = return_lookback_appropriate_index_andor_data(X, self.lookbacks, scaler=self.scaler_X, dropout_cols=model.dropout_cols)
        Y = self.scale_output_according_to_input_scaler(X_input, Y, prediction_col_in_input="£_close")
        Y_indy, Y = return_lookback_appropriate_index_andor_data(Y, self.lookbacks, scaler=None)
        X_indy_val, X_val = return_lookback_appropriate_index_andor_data(X_val, self.lookbacks, scaler=self.scaler_X, dropout_cols=model.dropout_cols)
        Y_val = self.scale_output_according_to_input_scaler(X_val_input, Y_val, prediction_col_in_input="£_close")
        Y_indy_val, Y_val = return_lookback_appropriate_index_andor_data(Y_val, self.lookbacks, scaler=None)
        #X_indy, X, Y_indy, Y                 = self.align_X_and_Y_for_fitting(X_indy, X, Y_indy, Y, 0)
        #X_indy_val, X_val, Y_indy_val, Y_val = self.align_X_and_Y_for_fitting(X_indy_val, X_val, Y_indy_val, Y_val, 0)

        if self.model_hyper_params["early_stopping"] != 0:
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_hyper_params["early_stopping"], restore_best_weights=True)
            history = model.fit(X, Y, epochs=self.model_hyper_params["epochs"], validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=verbose)
        else:
            history = model.fit(X, Y, epochs=self.model_hyper_params["epochs"], validation_data=(X_val, Y_val), verbose=verbose)
        #print(datetime.now().strftime("%H:%M:%S") + " - fit complete")
        # Train the model with early stopping
        
        
        #model.train_loss = history.history['loss']
        #model.val_loss   = history.history['val_loss']
        
        return model#, history.history['loss'][-1], history.history['val_loss'][-1]
             
    def evaluate_ensemble(self, df_X, df_y, pred_steps_value, confidences_before_betting_PC, financial_value_scaling):
        count = 0
        global global_random_state
        global_random_state = 42

        if True == False and hasattr(self,"training_scores_dict") and hasattr(self,"validation_scores_dict") and hasattr(self,"additional_validation_dict"):
            training_scores_dict        = self.training_scores_dict
            validation_scores_dict      = self.validation_scores_dict
            additional_validation_dict  = self.additional_validation_dict
        else:
            # variables
            training_scores_dict_list, validation_scores_dict_list, additional_validation_dict_list = [], [], []
            kf = KFold(n_splits=self.K_fold_splits, shuffle=False)

            print(datetime.now().strftime("%H:%M:%S") + " - evaluating model")
            
            for train_index, val_index in kf.split(df_X):
                print(datetime.now().strftime("%H:%M:%S") + "-" + str(count))
                
                for i_random in range(self.n_estimators_per_time_series_blocking):
                    # data prep
                    single_estimator = self.estimators_[count]

                    X_train = df_X.loc[df_X.index[train_index].values].copy()
                    y_train = df_y.loc[df_y.index[train_index].values].copy()

                    X_val = df_X.loc[df_X.index[val_index].values].copy()
                    y_val = df_y.loc[df_y.index[val_index].values].copy()
                    

                    # produce standard training scores
                    y_pred_train = self.custom_single_predict(X_train, single_estimator)
                    y_pred_val = self.custom_single_predict(X_val, single_estimator)


                    # collect training, validation, and validation additional analysis scores
                    training_scores_dict_list_new, additional_training_dict_list_new        = self.evaluate_results(y_train, y_pred_train, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"], self.input_dict["fin_inputs_params_dict"]["financial_value_scaling"])
                    validation_scores_dict_list_new, additional_validation_dict_list_new    = self.evaluate_results(y_val, y_pred_val, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"], self.input_dict["fin_inputs_params_dict"]["financial_value_scaling"])
                    training_scores_dict_list += [training_scores_dict_list_new]
                    validation_scores_dict_list += [validation_scores_dict_list_new]
                    additional_validation_dict_list += [additional_validation_dict_list_new]
                    count += 1

            training_scores_dict = average_list_of_identical_dicts(training_scores_dict_list)
            validation_scores_dict = average_list_of_identical_dicts(validation_scores_dict_list)
            additional_validation_dict = average_list_of_identical_dicts(additional_validation_dict_list)

            self.training_scores_dict = training_scores_dict
            self.validation_scores_dict = validation_scores_dict
            self.additional_validation_dict = additional_validation_dict

        return self, training_scores_dict, validation_scores_dict, additional_validation_dict

    def fit_ensemble(self, df_X, df_y):
        count = 0
        global global_random_state
        global_random_state = 42
        # variables
        training_scores_dict_list, validation_scores_dict_list, additional_validation_dict_list = [], [], []
        kf = KFold(n_splits=self.K_fold_splits, shuffle=False)

        print(datetime.now().strftime("%H:%M:%S") + " - training model")

        scaler_X = MinMaxScaler()
        self.scaler_X = scaler_X.fit(df_X)
        self.X_train_list = []
        self.y_train_list = []
        self.y_pred_list = []
        self.y_val_list = []
        del scaler_X
        for train_index, val_index in kf.split(df_X):
            count += 1
            print(datetime.now().strftime("%H:%M:%S") + "-" + str(count))
            for i_random in range(self.n_estimators_per_time_series_blocking):
                n_features = df_X.shape[1]
                dropout_cols = return_columns_to_remove(columns_list=df_X.columns, model=self)

                # data prep
                X_train = df_X.loc[df_X.index[train_index].values].copy()
                X_train.loc[:,dropout_cols] = 0
                y_train = df_y.loc[df_y.index[train_index].values].copy()

                X_val = df_X.loc[df_X.index[val_index].values].copy()
                X_val.loc[:,dropout_cols] = 0
                y_val = df_y.loc[df_y.index[val_index].values].copy()
                #print(datetime.now().strftime("%H:%M:%S") + "- return predictor")
                
                # initialising and prepping
                single_estimator = return_RNN_ensamble_estimator(self.model_hyper_params, global_random_state, n_features)
                single_estimator.dropout_cols = dropout_cols
                global_random_state += 1
                #print(datetime.now().strftime("%H:%M:%S") + "- fitting")
                single_estimator = self.return_single_component_model_fitted_with_early_stopping(single_estimator, X_train, y_train, X_val, y_val)
                                
                #record training data, without scaling
                self.X_train_list += [X_train]
                self.y_train_list += [y_train]
                
                # produce standard training scores
                y_pred_train = self.custom_single_predict(X_train, single_estimator) # pred here is the prediction of the price at [time + pred horizon] made at [time]
                y_pred_val = self.custom_single_predict(X_val, single_estimator)
                self.y_pred_list += [y_pred_val]
                self.y_val_list += [y_val]

                # collect training, validation, and validation additional analysis scores
                training_scores_dict_list_new, additional_training_dict_list_new        = self.evaluate_results(y_train, y_pred_train, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"],self.input_dict["fin_inputs_params_dict"]["financial_value_scaling"])
                validation_scores_dict_list_new, additional_validation_dict_list_new    = self.evaluate_results(y_val, y_pred_val, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"],self.input_dict["fin_inputs_params_dict"]["financial_value_scaling"])
                training_scores_dict_list += [training_scores_dict_list_new]
                validation_scores_dict_list += [validation_scores_dict_list_new]
                additional_validation_dict_list += [additional_validation_dict_list_new]
                self.estimators_ = self.estimators_ + [single_estimator]

        training_scores_dict = average_list_of_identical_dicts(training_scores_dict_list)
        validation_scores_dict = average_list_of_identical_dicts(validation_scores_dict_list)
        additional_validation_dict = average_list_of_identical_dicts(additional_validation_dict_list)
        self.training_scores_dict       = training_scores_dict
        self.validation_scores_dict     = validation_scores_dict
        self.additional_validation_dict = additional_validation_dict

        return self, training_scores_dict, validation_scores_dict, additional_validation_dict

    def custom_single_predict(self, df_X, single_estimator, output_col_name="prediction_X_ahead"):

        index, input_data   = return_lookback_appropriate_index_andor_data(df_X, self.lookbacks, scaler=self.scaler_X, dropout_cols=single_estimator.dropout_cols)
        y_pred_values       = single_estimator.predict(input_data, verbose=0)
        y_pred_values       = pd.DataFrame(y_pred_values, index=index, columns=[output_col_name])
        y_pred_values       = self.inverse_scale_output_according_to_input_scaler(df_X, y_pred_values)
        
        return y_pred_values
        
    def evaluate_results(self, y_test_input, y_pred_input, outputs_params_dict, reporting_dict, financial_value_scaling):
        #print(datetime.now().strftime("%H:%M:%S") + " - evaluate_results")
        pred_steps_value              = outputs_params_dict["pred_steps_ahead"]
        confidences_before_betting_PC = reporting_dict["confidence_thresholds"]
        seconds_per_time_steps        = self.input_dict["temporal_params_dict"]["time_step_seconds"]
        
        # they are then trimmed so that they align for the traditional measures
        y_test, y_pred = copy.copy(y_test_input), copy.copy(y_pred_input)
        if isinstance(y_pred, pd.Series):
            y_pred = pd.DataFrame(y_pred)
        merged_df = pd.merge(y_test, y_pred, left_index=True, right_index=True, how='inner')
        y_test = y_test.loc[merged_df.index]
        y_pred = y_pred.loc[merged_df.index]
        traditional_scores_dict_list = {"r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred)}
        additional_results_dict_list = FG_additional_reporting.return_results_X_min_plus_minus_accuracy(y_pred, y_test, pred_steps_value, confidences_before_betting_PC=confidences_before_betting_PC, financial_value_scaling=financial_value_scaling, seconds_per_time_steps=seconds_per_time_steps)
        return traditional_scores_dict_list, additional_results_dict_list
        
    def save(self, general_save_dir = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["predictive_model"], Y_preds_testing=None, y_testing=None):
        if hasattr(self, "name") and self.name != None:
            model_name = self.name
        else:
            model_name = custom_hash(return_predictor_name(self.input_dict))
        folder_path = os.path.join(general_save_dir, model_name + "\\")
        extension = "h5"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            # Delete existing model files with the same extension
            extensions_to_del = [extension, ".csv", ".pkl"]
            files_to_delete = []
            for ex in extensions_to_del:
                files_to_delete += [f for f in os.listdir(folder_path) if f.endswith(f'.{ex}')]
            for file in files_to_delete:
                os.remove(os.path.join(folder_path, file))
        # Save each model in the ensemble
        for i, model in enumerate(self.estimators_):
            model.save(os.path.join(folder_path, f'component_model_{i}.{extension}'))
        with open(os.path.join(folder_path,"input_dict.pkl"), "wb") as file:
                pickle.dump(self.input_dict, file)
        #save predictions if specified
        if isinstance(Y_preds_testing, pd.Series) or isinstance(Y_preds_testing, pd.DataFrame):
            Y_preds_testing.to_csv(os.path.join(folder_path, 'Y_preds_testing.csv'))
        if isinstance(y_testing, pd.Series) or isinstance(y_testing, pd.DataFrame):
            y_testing.to_csv(os.path.join(folder_path, 'y_testing.csv'))
        #save scaler and other assets
        dropout_cols_dict = {}
        for i, single_estimator in enumerate(self.estimators_):
            dropout_cols_dict[i] = single_estimator.dropout_cols
        additional_assets_dict = {
            "scaler_X" : self.scaler_X,
            "training_scores_dict"        : self.training_scores_dict,
            "validation_scores_dict"      : self.validation_scores_dict,
            "additional_validation_dict"  : self.additional_validation_dict,
            "dropout_cols"                : dropout_cols_dict
            }
        with open(os.path.join(folder_path,"additional_assets.pkl"), "wb") as file:
                pickle.dump(additional_assets_dict, file)
        self.save_training_data(folder_path)
        
    def save_training_data(self, folder_path):
        for target_str in ["X_train_list", "y_train_list", "y_pred_list", "y_val_list"]:
            if hasattr(self, target_str):
                for i, target_list_single in enumerate(getattr(self, target_str)):
                    target_list_single.to_csv(os.path.join(folder_path,"{}_{}.csv".format(target_str, i)))

    def predict_ensemble(self, X): #FG_action: This is where the new error is
        
        y_ensemble = pd.DataFrame()
        for i, single_estimator in enumerate(self.estimators_):
            # randomly select features to drop out
            y_ensemble[i] = self.custom_single_predict(X, single_estimator)
        
        output = y_ensemble.mean(axis=1)

        return output
    
    def load(self, predictor_location_folder_path, only_return_viability=False):
        folder_name = custom_hash(return_predictor_name(self.input_dict))
        folder_path = os.path.join(predictor_location_folder_path, folder_name+ "\\")
        
        with open(os.path.join(predictor_location_folder_path,"input_dict.pkl"), "rb") as file:
            save_input_dict = pickle.load(file)
        
        # check that the input parameters are the same
        if only_return_viability==True:
            copy_A, copy_B = copy.deepcopy(save_input_dict), copy.deepcopy(self.input_dict)
            # delete values that dont matter in comparition
            del copy_A["senti_inputs_params_dict"]["sentiment_method"], copy_B["senti_inputs_params_dict"]["sentiment_method"]
            del copy_A["temporal_params_dict"]['test_period_start'], copy_A["temporal_params_dict"]['test_period_end'], copy_B["temporal_params_dict"]['test_period_start'], copy_B["temporal_params_dict"]['test_period_end']
            del copy_A["fin_inputs_params_dict"]["historical_file"], copy_B["fin_inputs_params_dict"]["historical_file"]
            del copy_A["senti_inputs_params_dict"]["tweet_file_location"], copy_B["senti_inputs_params_dict"]["tweet_file_location"]
                        
            # new consideration for: consider tweet attention & include_subject_vol, if the order doesnt ask for the new system, and the loaded order doesn't state if it has the old or new system. Assume it has the old system and 
            # insert it ready for comparison
            for sentiment_variable_str in ["factor_tweet_attention", "factor_topic_volume"]:
                for item in [copy_A, copy_B]:
                    if not sentiment_variable_str in item["senti_inputs_params_dict"].keys():
                        item["senti_inputs_params_dict"][sentiment_variable_str] = False
                    elif item["senti_inputs_params_dict"][sentiment_variable_str] == 0:
                        item["senti_inputs_params_dict"][sentiment_variable_str] = False
            #cancel out unneeded variable for comparition of single and no topic models
            for item in [copy_A, copy_B]:
                if item["senti_inputs_params_dict"]['topic_qty'] <= 1:
                    item["senti_inputs_params_dict"]['topic_model_alpha']    = None
                    item["senti_inputs_params_dict"]['apply_IDF']            = None
                    item["senti_inputs_params_dict"]['enforced_topics_dict_name'] = None
                    item["senti_inputs_params_dict"]['enforced_topics_dict'] = None
                    item["senti_inputs_params_dict"]['topic_weight_square_factor'] = None
                    item["senti_inputs_params_dict"]['factor_topic_volume'] = None
                if item["senti_inputs_params_dict"]['topic_qty'] <= 0:
                    item["senti_inputs_params_dict"]['topic_training_tweet_ratio_removed'] = None
                    item["senti_inputs_params_dict"]['relative_lifetime'] = None
                    item["senti_inputs_params_dict"]['relative_halflife'] = None
                    item["senti_inputs_params_dict"]['regenerate_cleaned_tweets_for_subject_discovery'] = None
                    item["senti_inputs_params_dict"]['inc_new_combined_stopwords_list'] = None
                    item["senti_inputs_params_dict"]['factor_tweet_attention'] = None
                    
                           
            if not copy_A == copy_B:
                print("ZZZZZZZZZZ differences in input dicts with identical hashcodes:{} found {}".format(str(folder_name), compare_dicts(copy_A, copy_B)))
                return False
            else:
                print("predictor match OK")
                return True
        
        # otherwise load the file
        print("xxx loading predictor{}".format(folder_name))
        with open(os.path.join(predictor_location_folder_path,"additional_assets.pkl"), "rb") as file:
            additional_assets_dict = pickle.load(file)
        for filename in os.listdir(predictor_location_folder_path):
            if filename.endswith(".h5"):
                file_path = os.path.join(predictor_location_folder_path, filename)
                #each component model needs its dropout cols individually populated
                file_num = filename.replace("component_model_","")
                file_num = int(file_num.replace(".h5",""))
                single_estimator = load_model(file_path)
                single_estimator.dropout_cols = additional_assets_dict["dropout_cols"][file_num]
                self.estimators_ += [single_estimator]

        # load additional factors
        for attr in ["scaler_X", "training_scores_dict", "validation_scores_dict", "additional_validation_dict"]:
            if attr in additional_assets_dict.keys():
                setattr(self, attr, additional_assets_dict[attr])

        



def load_RNN_predictor(input_dict, predictor_location_folder_path, only_return_viability=False):
        predictor = initiate_model(input_dict, hash_name=os.path.split(os.path.split(predictor_location_folder_path)[0])[-1])
        if only_return_viability == True:
            return predictor.load(predictor_location_folder_path, only_return_viability=only_return_viability)
        predictor.load(predictor_location_folder_path)
        return predictor

def return_lookback_appropriate_index_andor_data(df_x_input, lookbacks, scaler=None, dropout_cols=None):
     # this method, according to result bools, returns the index and input data so that time 
    # periods spanning two days are removed
    df_x = copy.copy(df_x_input)
    output_input, output_index = [], []
    trim_from_indexes = lookbacks-1
    ori_index = df_x.index
    if not scaler == None:
        df_x = pd.DataFrame(scaler.transform(df_x), index=df_x.index, columns=df_x.columns)

    if not dropout_cols == None:
        df_x.loc[:,dropout_cols] = 0

    days = ori_index.day.values

    for ts0, ts1 in zip(ori_index[:-trim_from_indexes], ori_index[trim_from_indexes:]):
        if ts0.day == ts1.day:
            #output_index += [ts1]
            #output_input += [df_x.loc[ts0:ts1,:].values]
            output_index.append(ts1)
            output_input.append(df_x.loc[ts0:ts1,:].values)
            #output_input += [list(ori_index[ts0:ts1].values)]

    output_index = np.array(output_index)
    output_input = np.array(output_input)
    

    return output_index, output_input
    

def return_columns_to_remove(columns_list, model=None):
    
    columns_to_remove = list(columns_list.copy())
    retain_cols = []
    
    retain_dict = model.model_hyper_params["cohort_retention_rate_dict"]
    general_adjusting_square_factor = model.model_hyper_params["general_adjusting_square_factor"]

    #max_features = self.max_features
    stock_strings_list = []
    columns_list = list(columns_list)

    for key in retain_dict:
        cohort = []
        target_string = key
        #if len(fnmatch.filter([key], "STOCK_NAME*")) > 0:
        #    for stock in stock_strings_list:
        #        target_string = key
        #        target_string = target_string.replace("STOCK_NAME", stock)
        #        cohort = cohort + fnmatch.filter(columns_list, target_string)
        #else:
        if "~senti_*" == target_string:
            target_string = "~sent*"
        cohort = cohort + fnmatch.filter(columns_list, target_string)
        for col in cohort:
                columns_list.remove(col)
        if '~sent_topic_W0' in cohort and target_string != "*":
            num_lim = int(0.5 * len(cohort))
            retain_nums = list(np.random.choice(range(num_lim), math.ceil(num_lim * (retain_dict[key] ** general_adjusting_square_factor)), replace=False))
            for x in retain_nums:
                cohort_temp = cohort
                retain_cols = retain_cols + [(col_str) for col_str in cohort_temp if ("W{}".format(x) in col_str or "t{}".format(x) in col_str)]
        elif len(cohort) > 0:
            retain_cols = retain_cols + list(np.random.choice(cohort, math.ceil(len(cohort) * (retain_dict[key] ** general_adjusting_square_factor)), replace=False))

    for value in retain_cols:
        try:
            columns_to_remove.remove(value)
        except:
            a=1
    
    return columns_to_remove


#%% Module - Model Testing and Standard Reporting

def return_testing_scores_and_testing_time_series(predictor, input_dict):
    testing_scores, X_testing, y_testing, Y_preds_testing = generate_testing_scores(predictor, input_dict, return_time_series=True)
    return testing_scores, X_testing, y_testing, Y_preds_testing

def generate_testing_scores(predictor, input_dict, return_time_series=False):
                            
    temporal_params_dict    = input_dict["temporal_params_dict"]
    fin_inputs_params_dict  = input_dict["fin_inputs_params_dict"]
    senti_inputs_params_dict = input_dict["senti_inputs_params_dict"]
    outputs_params_dict     = input_dict["outputs_params_dict"]
    model_hyper_params      = input_dict["model_hyper_params"]
    reporting_dict          = input_dict["reporting_dict"]
    
        
    # step 1a: generate testing input data (finanical)
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 1a: generate testing input data")
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="testing")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"], fin_inputs_params_dict["financial_value_scaling"])

    # step 1b: generate testing input data (sentiment)
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 1b: generate testing input data (sentiment)")
    df_sentiment_data = retrieve_or_generate_sentiment_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="testing")

    # step 2: generate testing output data
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 2: generate testing output data")
    X_testing, y_testing = create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
        
    # step 3: generate y_pred
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 3: generate y_pred")
    Y_preds_testing = predictor.predict_ensemble(X_testing) 
    
    # step 4: generate score
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 4: generate score")
    testing_scores, additional_results_dict = predictor.evaluate_results(y_testing, Y_preds_testing, outputs_params_dict, reporting_dict, input_dict["fin_inputs_params_dict"]["financial_value_scaling"])
    predictor.save(Y_preds_testing=Y_preds_testing, y_testing=y_testing)
    if return_time_series == False:
        return testing_scores
    else:
        return testing_scores, X_testing, y_testing, Y_preds_testing


#%% main support methods

def retrieve_model_and_training_scores(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict, only_return_viability=False):
    #predictor_location_file = "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\predictive_model\\aapl_ps04_06_18_000000_pe01_09_20_000000_ts_sec300_tm_qty7_r_lt3600_r_hl3600_tm_alpha1_IDF-True_t_ratio_r100000.pred"
    input_dict = return_input_dict(temporal_params_dict = temporal_params_dict, fin_inputs_params_dict = fin_inputs_params_dict, senti_inputs_params_dict = senti_inputs_params_dict, outputs_params_dict = outputs_params_dict, model_hyper_params = model_hyper_params, reporting_dict = reporting_dict)
    if only_return_viability == True:
        return load_RNN_predictor(input_dict, predictor_location_folder_path, only_return_viability=only_return_viability)
    
    model = load_RNN_predictor(input_dict, predictor_location_folder_path)
    

    print("loading predictor: ", predictor_location_folder_path)
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="training")
    #training_score = edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="load")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"], fin_inputs_params_dict["financial_value_scaling"])
    df_sentiment_data = retrieve_or_generate_sentiment_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
    
    X_train, y_train   = create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
    model, training_scores_dict, validation_scores_dict, additional_validation_dict = model.evaluate_ensemble(X_train, y_train, outputs_params_dict["pred_steps_ahead"], confidences_before_betting_PC=reporting_dict["confidence_thresholds"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
    
    return model, training_scores_dict, validation_scores_dict, additional_validation_dict

def generate_model_and_validation_scores(temporal_params_dict,
    fin_inputs_params_dict,
    senti_inputs_params_dict,
    outputs_params_dict,
    model_hyper_params,
    reporting_dict,
    hash_name=None):
    #desc
    
    
    #general parameters
    global global_index_cols_list, global_input_cols_to_include_list
    input_dict = return_input_dict(temporal_params_dict = temporal_params_dict, fin_inputs_params_dict = fin_inputs_params_dict, senti_inputs_params_dict = senti_inputs_params_dict, outputs_params_dict = outputs_params_dict, model_hyper_params = model_hyper_params, reporting_dict = reporting_dict)
    #stock market data prep
    print(datetime.now().strftime("%H:%M:%S") + " - importing and prepping financial data")
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="training")
    print(datetime.now().strftime("%H:%M:%S") + " - populate_technical_indicators")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"], fin_inputs_params_dict["financial_value_scaling"])

    if senti_inputs_params_dict["topic_qty"] >= 1:
        #sentiment data prep
        print(datetime.now().strftime("%H:%M:%S") + " - importing or prepping sentiment data")
        df_sentiment_data = retrieve_or_generate_sentiment_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
    elif senti_inputs_params_dict["topic_qty"] == 0:
        df_sentiment_data = None
        
    #model training - create regressors
    X_train, y_train   = create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
    model              = initiate_model(input_dict, hash_name = hash_name)
    model, training_scores_dict, validation_scores_dict, additional_validation_dict = model.fit_ensemble(X_train, y_train)
    #report timings
    print(datetime.now().strftime("%H:%M:%S") + " - complete generating model")
    global global_start_time
    total_run_secs      = (datetime.now() - global_start_time).total_seconds()
    total_run_hours     = total_run_secs // 3600
    total_run_minutes   = (total_run_secs % 3600) // 60
    total_run_seconds   = total_run_secs % 60
    report = f"{int(total_run_hours)} hours, {int(total_run_minutes)} minutes, {int(total_run_seconds)} seconds"
    print(report)
    return model, training_scores_dict, validation_scores_dict, additional_validation_dict




#%% main line

def custom_hash(data):
    # Use a cryptographic hash function (e.g., SHA-256)
    hash_object = hashlib.sha256()
    hash_object.update(str(data).encode('utf-8'))
    return hash_object.hexdigest()

def retrieve_or_generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict):
    
    #global values
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    
    #general parameters
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    train_period_start  = temporal_params_dict["train_period_start"]
    train_period_end    = temporal_params_dict["train_period_end"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    time_step_seconds   = temporal_params_dict["time_step_seconds"]
    topic_model_qty     = senti_inputs_params_dict["topic_qty"]
    rel_lifetime        = senti_inputs_params_dict["relative_lifetime"]
    rel_hlflfe          = senti_inputs_params_dict["relative_halflife"]
    apply_IDF           = senti_inputs_params_dict["apply_IDF"]
            
    #search for predictor
    predictor_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["predictive_model"]
    predictor_name_entry = company_symbol, train_period_start, train_period_end, time_step_seconds, topic_model_qty, rel_lifetime, rel_hlflfe, topic_model_alpha, apply_IDF, tweet_ratio_removed
    predictor_name = return_predictor_name(return_input_dict(temporal_params_dict = temporal_params_dict, fin_inputs_params_dict = fin_inputs_params_dict, senti_inputs_params_dict = senti_inputs_params_dict, outputs_params_dict = outputs_params_dict, model_hyper_params = model_hyper_params, reporting_dict = reporting_dict))
    predictor_location_folder_path = predictor_folder_location_string + custom_hash(predictor_name) + "//"
    print(str(temporal_params_dict["train_period_end"]) + "_____" + str(temporal_params_dict["test_period_start"])+ "_____" + str(temporal_params_dict["test_period_end"]))
    if os.path.exists(predictor_location_folder_path):
        model_matches = retrieve_model_and_training_scores(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict, only_return_viability=True)    
        if model_matches == True:
            predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = retrieve_model_and_training_scores(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
    else:
        print(datetime.now().strftime("%H:%M:%S") + " - generating model and testing scores")
        predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = generate_model_and_validation_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict, hash_name=custom_hash(predictor_name))
        predictor.save()
    return predictor, training_scores_dict, validation_scores_dict, additional_validation_dict

if __name__ == '__main__':
    predictor, traditional_training_scores, validation_dict = retrieve_or_generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
    testing_scores, Y_preds_testing                         = generate_testing_scores(predictor, input_dict)
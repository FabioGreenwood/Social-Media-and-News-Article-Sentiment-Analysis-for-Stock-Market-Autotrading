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
from stock_indicators import PeriodSize, PivotPointType 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Bidirectional, LSTM, Dense, GRU
import keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import timeseries_dataset_from_array
from config import global_strptime_str, global_strptime_str_filename, global_precalculated_assets_locations_dict, global_financial_history_folder_path, global_error_str_1, global_financial_history_folder_path, global_df_stocks_list_file, global_random_state, global_strptime_str_2, global_index_cols_list, global_input_cols_to_include_list, global_start_time, global_financial_history_folder_path
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import Dense
import hashlib
from tensorflow.keras.callbacks import EarlyStopping


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

def return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor):
    global global_strptime_str, global_strptime_str_filename
    if topic_model_qty == 1 or topic_model_qty == 0:
        weighted_topics, topic_weight_square_factor, topic_model_alpha = "NA", "NA", "NA"
    name = company_symbol + "_ps" + train_period_start.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_pe" + train_period_end.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_" + str(weighted_topics) + "_twsf" + str(topic_weight_square_factor) + "_"
    name = name + return_topic_model_name(topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc)
    return name

def return_sentiment_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe):
    global global_strptime_str, global_strptime_str_filename
    name = return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor)
    name = name + "_ts_sec" + str(time_step_seconds) + "_r_lt" + str(rel_lifetime) + "_r_hl" + str(rel_hlflfe)
    return name

def return_predictor_name(input_dict):
    global global_strptime_str, global_strptime_str_filename
    company_symbol = input_dict["outputs_params_dict"]["output_symbol_indicators_tuple"][0]
    train_period_start  = input_dict["temporal_params_dict"]["train_period_start"]
    train_period_end    = input_dict["temporal_params_dict"]["train_period_end"]
    weighted_topics     = input_dict["senti_inputs_params_dict"]["weighted_topics"]
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
    financial_value_scaling          = input_dict["fin_inputs_params_dict"]["financial_value_scaling"]

    name = return_sentiment_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe)
    predictor_hash = ""
    if not financial_value_scaling == None:
        predictor_hash += "_" + str(financial_value_scaling)
    predictor_hash += "_" + str(input_dict["model_hyper_params"]["n_estimators_per_time_series_blocking"]) + "_" + str(input_dict["model_hyper_params"]["testing_scoring"]) + "_" + str(input_dict["model_hyper_params"]["estimator__alpha"]) + "_" + str(input_dict["model_hyper_params"]["estimator__activation"]) + "_" + str(input_dict["model_hyper_params"]["cohort_retention_rate_dict"]) + "_" + str(input_dict["model_hyper_params"]["general_adjusting_square_factor"]) + "_" + str(input_dict["model_hyper_params"]["epochs"]) + "_" + str(input_dict["model_hyper_params"]["lookbacks"]) + "_" + str(input_dict["model_hyper_params"]["shuffle_fit"]) + "_" + str(input_dict["model_hyper_params"]["K_fold_splits"]) + "_" + str(pred_steps_ahead) + "_" + str(input_dict["model_hyper_params"]["estimator__alpha"]) + "_" + str(input_dict["model_hyper_params"]["general_adjusting_square_factor"]) + "_" + str(input_dict["model_hyper_params"]["lookbacks"]) + "_" + str(input_dict["model_hyper_params"]["batch_ratio"]) + "_" + str(input_dict["model_hyper_params"]["scaler_cat"])
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


def retrieve_or_generate_then_populate_technical_indicators(df, tech_indi_dict, match_doji, target_file_path, fin_data_scale):
    global global_strptime_str_filename
    quotes_list = [
        Quote(d,o,h,l,c,v) 
        for d,o,h,l,c,v 
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
                #calculate indicators
                if key == "sma":
                    results = indicators.get_sma(quotes_list, int(value))
                    attr = ["sma"]
                    col_pref = ""
                elif key == "ema":
                    results = indicators.get_ema(quotes_list, int(value))
                    attr = ["ema"]
                    col_pref = ""
                elif key == "macd":
                    if not len(value) == 3:
                        raise ValueError("the entries for macd, must be lists of a 3 value length")
                    results = indicators.get_macd(quotes_list, int(value[0]), int(value[1]), int(value[2]))
                    attr = ["macd", "signal", "histogram", "fast_ema", "slow_ema"]
                    col_pref = "macd_"
                elif key == "BollingerBands":
                    if not len(value) == 2:
                        raise ValueError("the entries for BollingerBands, must be lists of a 2 value length")
                    results = indicators.get_bollinger_bands(quotes_list, int(value[0]), int(value[1]))
                    attr = ["upper_band", "lower_band", "percent_b", "z_score", "width"]
                    col_pref = "Bollinger_"
                elif key == "PivotPoints":
                    results = indicators.get_pivot_points(quotes_list, PeriodSize.MONTH, PivotPointType.WOODIE);
                    attr = ["r3", "r2", "r1", "pp", "s1", "s2", "s3"]
                    col_pref = "PivotPoints"
                else:
                    raise ValueError("technical indicator " + key + " not programmed")
                # populate indicator
                for at in attr:
                    col_str = "$_" + col_pref + at + "_" + str(value)
                    df[col_str] = ""
                    for r in results:
                        df.at[r.date, col_str] = getattr(r,at)
                    print("tech indy")    
                    print("tech indy: " + str(key) + " " + str(value) +  " " + str(at) + " done " + datetime.now().strftime(global_strptime_str_filename))
            
        if match_doji == True:
            results = indicators.get_doji(quotes_list)
            df["match!_doji"] = ""
            for r in results:
                mat = r.match
                if mat == Match.BULL_CONFIRMED:
                    val = 3
                elif mat == Match.BULL_SIGNAL:
                    val = 2
                elif mat == Match.BULL_BASIS:
                    val = 1
                elif mat == Match.NEUTRAL:
                    val = 0
                elif mat == Match.NONE:
                    val = 0
                elif mat == Match.BEAR_BASIS:
                    val = -1
                elif mat == Match.BEAR_SIGNAL:
                    val = -2
                elif mat == Match.BEAR_CONFIRMED:
                    val = -3
                else:
                    raise ValueError(str(mat) + "not found in list")
                df.at[r.date, "match!_doji"] = val
        df.to_csv(technical_indicators_location_file)      

    df = rescale_financial_data_if_needed(df, fin_data_scale)

    return df


#%% SubModule – Sentiment Data Prep

def retrieve_or_generate_sentiment_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training"):
    #desc
    
    #general parameters
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    time_step_seconds   = temporal_params_dict["time_step_seconds"]
    topic_model_qty     = senti_inputs_params_dict["topic_qty"]
    rel_lifetime        = senti_inputs_params_dict["relative_lifetime"]
    rel_hlflfe          = senti_inputs_params_dict["relative_halflife"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    apply_IDF           = senti_inputs_params_dict["apply_IDF"]
    weighted_topics     = senti_inputs_params_dict["weighted_topics"]
    pred_steps          = outputs_params_dict["pred_steps_ahead"]
    enforced_topic_model_nested_list = senti_inputs_params_dict["enforced_topics_dict"]
    new_combined_stopwords_inc = senti_inputs_params_dict["inc_new_combined_stopwords_list"]
    topic_weight_square_factor       = senti_inputs_params_dict["topic_weight_square_factor"]
    
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
    sentiment_data_name = return_sentiment_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor, time_step_seconds, rel_lifetime, rel_hlflfe)
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



def generate_sentiment_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, repeat_timer=10, training_or_testing="training", hardcode_df_annotated_tweets=None):
    
    #general parameters
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    weighted_topics     = senti_inputs_params_dict["weighted_topics"]
    num_topics          = senti_inputs_params_dict["topic_qty"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    seconds_per_time_steps  = temporal_params_dict["time_step_seconds"]
    relavance_lifetime      = senti_inputs_params_dict["relative_lifetime"]
    relavance_halflife      = senti_inputs_params_dict["relative_halflife"]
    weighted_topics         = senti_inputs_params_dict["weighted_topics"]
    apply_IDF               = senti_inputs_params_dict["apply_IDF"]
    new_combined_stopwords_inc          = senti_inputs_params_dict["inc_new_combined_stopwords_list"]
    enforced_topic_model_nested_list    = senti_inputs_params_dict["enforced_topics_dict"]
    topic_weight_square_factor          = senti_inputs_params_dict["topic_weight_square_factor"]

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
    annotated_tweets_name = return_annotated_tweets_name(company_symbol, period_start, period_end, weighted_topics, num_topics, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, new_combined_stopwords_inc, topic_weight_square_factor)
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
    
    #generate sentiment data from topic model and annotate tweets
    #index = generate_datetimes(train_period_start, train_period_end, seconds_per_time_steps)
    #df_sentiment_scores = pd.DataFrame()
    columns = []
    for i in range(num_topics):
        columns = columns + ["~senti_score_t" + str(i)]
    df_sentiment_scores = pd.DataFrame(index=index, columns=columns)
    
    #create the initial cohort of tweets to be looked at in a time window
    epoch_time          = datetime(1970, 1, 1)
    #tweet_cohort_start  = (train_period_start - epoch_time) - timedelta(seconds=relavance_lifetime)
    #tweet_cohort_end    = (train_period_start - epoch_time)
    #tweet_cohort_start  = tweet_cohort_start.total_seconds()
    #tweet_cohort_end    = tweet_cohort_end.total_seconds()
    #tweet_cohort        = return_tweet_cohort_from_scratch(df_annotated_tweets, df_annotated_tweets, tweet_cohort_start, tweet_cohort_end)
    
    #index_list = list(index)
    #print(type(index_list))
    #rep_num = int(len(index_list) / 10)
    
    tweet_cohort_t1 = pd.DataFrame(index=index, columns=["tweet_cohort_start_post", "tweet_cohort_end_post", "tweet_cohort_t1", "tweet_cohort"])
    tweet_cohort_t1["tweet_cohort_start_post"]  = tweet_cohort_t1.index
    tweet_cohort_t1["tweet_cohort_end_post"]    = tweet_cohort_t1["tweet_cohort_start_post"].apply(lambda datetime: ((datetime - epoch_time)).total_seconds())
    tweet_cohort_t1["tweet_cohort_start_post"]  = tweet_cohort_t1["tweet_cohort_start_post"].apply(lambda datetime: ((datetime - epoch_time) - timedelta(seconds=relavance_lifetime)).total_seconds())
    
    def process_row_2(row, df_annotated_tweets=df_annotated_tweets, topic_num=num_topics, relavance_halflife=relavance_halflife):
        tweet_cohort_start_post = row["tweet_cohort_start_post"]
        tweet_cohort_end_post = row["tweet_cohort_end_post"]
        tweet_cohort = return_tweet_cohort_from_scratch(df_annotated_tweets, tweet_cohort_start_post, tweet_cohort_end_post)

        pre_calc_time_overall = np.exp((-3 / relavance_halflife) * (tweet_cohort["post_date"] - tweet_cohort_start_post)) * tweet_cohort["~sent_overall"]

        senti_scores = []
        for topic_num in range(num_topics):
            score_numer = np.sum(pre_calc_time_overall * tweet_cohort[f"~sent_topic_W{topic_num}"])
            score_denom = np.sum(np.exp((-3 / relavance_halflife) * (tweet_cohort["post_date"] -  tweet_cohort_start_post)) * tweet_cohort[f"~sent_topic_W{topic_num}"])
            if score_denom > 0:
                senti_scores = senti_scores + [score_numer / score_denom]
            elif score_denom == 0:
                senti_scores = senti_scores + [0]
            else:
                senti_scores = senti_scores + [2]
        
        return senti_scores
        
    data = tweet_cohort_t1.apply(process_row_2, axis=1)
    for indy in tweet_cohort_t1.index:
        df_sentiment_scores.loc[indy, :] = data[indy]

            
    
    return df_sentiment_scores

def generate_annotated_tweets(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, repeat_timer=10, training_or_testing="training", hardcode_location_for_topic_model="", k_fold_textual_sources=None):
    
    
    #general parameters
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    weighted_topics     = senti_inputs_params_dict["weighted_topics"]
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
    
    # Calculate sentiment scores for all tweets
    sentiment_scores = df_annotated_tweets['body'].apply(lambda text: sentiment_method.polarity_scores(text)['compound'])
    sentiment_scores.name = "~sent_overall"

    # Calculate topic weights for all tweets
    topic_weights = df_annotated_tweets['body'].apply(lambda text: return_topic_weight(text, topic_model_dict, num_topics))
    topic_weights = topic_weights.apply(lambda topic_tuples: [t[1] for t in topic_tuples] if len(topic_tuples) == num_topics else list(np.zeros(num_topics)))
    if topic_weight_square_factor != 1:
        topic_weights = topic_weights.apply(lambda x: adjust_topic_weights(x, topic_weight_square_factor))

    
    # Combine sentiment and topic weights
    temp_list = []
    for indy in topic_weights.index: temp_list = temp_list + [topic_weights[indy]]
    topic_weights_prepped = pd.DataFrame(temp_list, index=sentiment_scores.index, columns=sentiment_col_strs)
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
    #quick fix to reduce the amount of wasted time cleaning tweets
    if reclean_tweets == False and os.path.exists(senti_inputs_params_dict["cleaned_tweet_file_location"]):
        df_prepped_tweets_company_agnostic = pd.read_csv(senti_inputs_params_dict["cleaned_tweet_file_location"])
    else:
        df_prepped_tweets = pd.read_csv(senti_inputs_params_dict["tweet_file_location"])
        print(len(df_prepped_tweets))
        tweets_list = list(df_prepped_tweets["body"])
        df_prepped_tweets_company_agnostic = prep_and_save_twitter_text_for_subject_discovery(tweets_list[::int(len(tweets_list)/5e2)], df_stocks_list_file=global_df_stocks_list_file, save_location=senti_inputs_params_dict["cleaned_tweet_file_location"], inc_new_combined_stopwords_list=inc_new_combined_stopwords_list)
        #with open(senti_inputs_params_dict["tweet_file_location"], 'rb') as file:
        #    df_prepped_tweets_company_agnostic = pickle.load(file)
    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Creating Subject Keys --------------------------------")
    wordcloud, topic_model_dict, visualisation  = return_subject_keys(df_prepped_tweets_company_agnostic, topic_qty = num_topics, topic_model_alpha=topic_model_alpha, apply_IDF=apply_IDF,
                                                                      enforced_topics_dict=enforced_topics_dict, return_LDA_model=True, return_png_visualisation=True, return_html_visualisation=True)
    save_topic_clusters(wordcloud, topic_model_dict, visualisation, file_location_wordcloud, file_location_topic_model_dict, file_location_visualisation)
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


def prep_and_save_twitter_text_for_subject_discovery(input_list, df_stocks_list_file=None, 
                                                     save_location=None, 
                                                     make_company_agnostic=True, inc_new_combined_stopwords_list=False):
    global global_precalculated_assets_locations_dict, combined_stop_words_list_file_path, custom_stop_words_list_file_path
    from config import global_combined_stopwords_list_path
    #prep parameters
    death_characters    = ["$", "amazon", "apple", "goog", "tesla", "http", "@", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "compile", "www","(", ")"]
    stocks_list         = list(df_stocks_list_file["Name"].map(lambda x: x.lower()).values)
    tickers_list        = list(df_stocks_list_file["Ticker"].map(lambda x: x.lower()).values)
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
    stocks_list_shortened_dict  = update_shortened_company_file(global_df_stocks_list_file, corp_stopwords)
    stocks_list_shortened       = list(stocks_list_shortened_dict.values())
    
    if make_company_agnostic==False:
        #death_characters        = 
        stocks_list             = []
        tickers_list            = []
        #stopwords_english       = 
        #corp_stopwords          = 
        #misc_stopwords          = 
        #stocks_list_shortened   = 
    
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
        copy_of_tweet = copy.deepcopy(tweet)
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
    df_output.to_csv(save_location)
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



def return_subject_keys(df_prepped_tweets_company_agnostic, topic_qty=10, enforced_topics_dict=None, stock_names_list=None, words_to_remove=None, 
                        return_LDA_model=True, return_png_visualisation=False, return_html_visualisation=False, 
                        topic_model_alpha=0.1, apply_IDF=True, cores=2, passes=60, iterations=800, return_perplexity=False):
    output = []

    data = df_prepped_tweets_company_agnostic[::40]
    print("topic model trained from {} tweets".format(len(data)))
    data_words = list(sent_to_words(data))
    
    if return_LDA_model < return_html_visualisation:
        raise ValueError("You must return the LDA visualisation if you return the LDA model")

    if return_png_visualisation:
        # Create a long string for word cloud visualization
        long_string = "start"
        for w in data_words:
            long_string = long_string + ',' + ','.join(w)
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
        wordcloud.generate(long_string)
        wordcloud.to_image()
        output = output + [wordcloud]
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

def return_tweet_cohort_from_scratch(df_annotated_tweets_temp, tweet_cohort_start, tweet_cohort_end):
    maskA           = df_annotated_tweets_temp["post_date"] >= tweet_cohort_start
    maskB           = df_annotated_tweets_temp["post_date"] <= tweet_cohort_end
    mask            = maskA & maskB
    tweet_cohort    = df_annotated_tweets_temp[mask]
    return tweet_cohort

#def return_single_topic_sentiment_vector_overtime(df_annotated_tweets_temp):
#    seconds_per_time_steps
#    relavance_lifetime


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

def create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list, pred_steps_ahead, financial_value_scaling):
    #this method populates each row with the next X output results, this is done so that, each time step can be trained
    #to predict the value of the next X steps
    
    new_col_str = "{}_{}"
    list_of_new_columns = []
    nan_values_replaced = 0
    train_test_split = 1
    if isinstance(df_sentiment_data, pd.DataFrame):
        df_sentiment_data.index = pd.to_datetime(df_sentiment_data.index)
        df_sentiment_data = df_sentiment_data.loc[list(df_financial_data.index)]
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

    X = copy.deepcopy(data)
    y = copy.deepcopy(data[list_of_new_columns])
    
    for col in list_of_new_columns:
        X = X.drop(col, axis=1)
    
    return X, y

#create model

def initiate_model(input_dict):
    if input_dict["model_hyper_params"]["name"] == "RandomSubspace_RNN_Regressor":
        estimator = DRSLinRegRNN(base_estimator=Sequential(),
            input_dict = input_dict)
    else:
        raise ValueError("the model type: " + str(input_dict["model_hyper_params"]["name"]) + " was not found in the method")
    
    return estimator

def return_RNN_ensamble_estimator(model_hyper_params, global_random_state, n_features, dropout_cols):
    
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
        kwargs["kernel_regularizer"] = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])
        kwargs["bias_regularizer"] = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])
        kwargs["activity_regularizer"] = tf.keras.regularizers.L1(model_hyper_params["estimator__alpha"])    
        # add layer 
        if layer[0] == "simple":
            ensemble_estimator.add(SimpleRNN(**kwargs))            
        elif layer[0] == "GRU":
            ensemble_estimator.add(GRU(**kwargs))
        elif layer[0] == "LSTM":
            ensemble_estimator.add(LSTM(**kwargs))
        
    ensemble_estimator.add(Dense(units=1, activation='linear'))
    ensemble_estimator.compile(optimizer='adam', loss='mae')
    ensemble_estimator.random_state = global_random_state
    ensemble_estimator.dropout_cols_ = dropout_cols
    
    return ensemble_estimator


class DRSLinRegRNN():
    def __init__(self, base_estimator=Sequential(),
                 input_dict=None):
        #expected keys: training_time_splits, max_depth, max_features, random_state,        
        self.estimator_info_pack = {}
        for key in input_dict["model_hyper_params"]: #FG_action: ensure this is aligned
           setattr(self, key, input_dict["model_hyper_params"][key])
        self.model_hyper_params   = input_dict["model_hyper_params"]
        self.input_dict           = input_dict
        self.base_estimator       = base_estimator
        self.estimators_          = []

    def return_single_ensable_model_fitted(self, model, X, Y):
        X_indy, X = return_lookback_appropriate_index_andor_data(X, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_X)
        Y_indy, Y = return_lookback_appropriate_index_andor_data(Y, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_y)
        model.fit(X, Y, epochs=self.epochs, shuffle=self.shuffle_fit)
        return model

    def return_single_ensable_model_fitted_with_early_stopping(self, model, X, Y, X_val, Y_val):
        X_indy, X = return_lookback_appropriate_index_andor_data(X, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_X)
        Y_indy, Y = return_lookback_appropriate_index_andor_data(Y, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_y)
        X_indy_val, X_val = return_lookback_appropriate_index_andor_data(X_val, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_X)
        Y_indy_val, Y_val = return_lookback_appropriate_index_andor_data(Y_val, self.lookbacks, return_index=True, return_input=True, scaler=self.scaler_y)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model with early stopping
        history = model.fit(X, Y, epochs=100, validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=0)
        model.train_loss = history.history['loss']
        model.val_loss   = history.history['val_loss']
        
        return model, history.history['loss'][-1], history.history['val_loss'][-1]
             

    def evaluate_ensemble(self, df_X, df_y, pred_steps_value, confidences_before_betting_PC):
        count = 0
        global global_random_state
        global_random_state = 42
        # variables
        training_scores_dict_list, validation_scores_dict_list, additional_validation_dict_list = [], [], []
        kf = KFold(n_splits=self.K_fold_splits, shuffle=False)

        print(datetime.now().strftime("%H:%M:%S") + " - evaluating model")
        if self.scaler_cat > 0:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            col = df_X.columns
            scaler_X.fit(df_X)
            #scaler_y.fit(df_y.values.reshape(-1, 1))
            scaler_y.fit(df_y)
            if self.scaler_cat == 2:
                scaler_X = self.adjust_scaler_to_accept_distant_pricings(scaler_X, df_X)
                scaler_y = self.adjust_scaler_to_accept_distant_pricings(scaler_y, df_y)
            self.scaler_X, self.scaler_y = scaler_X, scaler_y
            del scaler_X, scaler_y
        else:
            self.scaler_X, self.scaler_y = None, None

        for train_index, val_index in kf.split(df_X):
            print(datetime.now().strftime("%H:%M:%S") + "-" + str(count))
            for i_random in range(self.n_estimators_per_time_series_blocking):
                n_features = df_X.shape[1]
                dropout_cols = return_columns_to_remove(columns_list=df_X.columns, self=self)

                # data prep
                X_train = df_X.loc[df_X.index[train_index].values].copy()
                X_train.loc[:, dropout_cols] = 0
                y_train = df_y.loc[df_y.index[train_index].values].copy()

                X_val = df_X.loc[df_X.index[val_index].values].copy()
                X_val.loc[:, dropout_cols] = 0
                y_val = df_y.loc[df_y.index[val_index].values].copy()

                single_estimator = self.estimators_[count]
                
                # produce standard training scores
                y_pred_train = self.custom_single_predict(X_train, single_estimator)
                y_pred_val = self.custom_single_predict(X_val, single_estimator)
                

                # collect training, validation, and validation additional analysis scores
                training_scores_dict_list_new, additional_training_dict_list_new        = self.evaluate(y_train, y_pred_train, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"])
                validation_scores_dict_list_new, additional_validation_dict_list_new    = self.evaluate(y_val, y_pred_val, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"])
                training_scores_dict_list += [training_scores_dict_list_new]
                validation_scores_dict_list += [validation_scores_dict_list_new]
                additional_validation_dict_list += [additional_validation_dict_list_new]
                self.estimators_ = self.estimators_ + [single_estimator]
                count += 1

        training_scores_dict = average_list_of_identical_dicts(training_scores_dict_list)
        validation_scores_dict = average_list_of_identical_dicts(validation_scores_dict_list)
        additional_validation_dict = average_list_of_identical_dicts(additional_validation_dict_list)
        

        return self, training_scores_dict, validation_scores_dict, additional_validation_dict

    def fit_ensemble(self, df_X, df_y, pred_steps_value, confidences_before_betting_PC):
        count = 0
        global global_random_state
        global_random_state = 42
        # variables
        training_scores_dict_list, validation_scores_dict_list, additional_validation_dict_list = [], [], []
        kf = KFold(n_splits=self.K_fold_splits, shuffle=False)

        print(datetime.now().strftime("%H:%M:%S") + " - training model")
        if self.scaler_cat > 0:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            col = df_X.columns
            scaler_X.fit(df_X)
            #scaler_y.fit(df_y.values.reshape(-1, 1))
            scaler_y.fit(df_y)
            if self.scaler_cat == 2:
                scaler_X = self.adjust_scaler_to_accept_distant_pricings(scaler_X, df_X)
                scaler_y = self.adjust_scaler_to_accept_distant_pricings(scaler_y, df_y)
            self.scaler_X, self.scaler_y = scaler_X, scaler_y
            del scaler_X, scaler_y
        else:
            self.scaler_X, self.scaler_y = None, None

        for train_index, val_index in kf.split(df_X):
            count += 1
            print(datetime.now().strftime("%H:%M:%S") + "-" + str(count))
            for i_random in range(self.n_estimators_per_time_series_blocking):
                n_features = df_X.shape[1]
                dropout_cols = return_columns_to_remove(columns_list=df_X.columns, self=self)

                # data prep
                X_train = df_X.loc[df_X.index[train_index].values].copy()
                X_train.loc[:, dropout_cols] = 0
                y_train = df_y.loc[df_y.index[train_index].values].copy()

                X_val = df_X.loc[df_X.index[val_index].values].copy()
                X_val.loc[:, dropout_cols] = 0
                y_val = df_y.loc[df_y.index[val_index].values].copy()

                
                # initialising and prepping
                single_estimator = return_RNN_ensamble_estimator(self.model_hyper_params, global_random_state, n_features, dropout_cols)
                global_random_state += 1
                single_estimator, train_loss, val_loss = self.return_single_ensable_model_fitted_with_early_stopping(single_estimator, X_train, y_train, X_val, y_val)
                
                # produce standard training scores
                y_pred_train = self.custom_single_predict(X_train, single_estimator)
                y_pred_val = self.custom_single_predict(X_val, single_estimator)
                

                # collect training, validation, and validation additional analysis scores
                training_scores_dict_list_new, additional_training_dict_list_new        = self.evaluate(y_train, y_pred_train, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"])
                validation_scores_dict_list_new, additional_validation_dict_list_new    = self.evaluate(y_val, y_pred_val, self.input_dict["outputs_params_dict"], self.input_dict["reporting_dict"])
                training_scores_dict_list += [training_scores_dict_list_new]
                validation_scores_dict_list += [validation_scores_dict_list_new]
                additional_validation_dict_list += [additional_validation_dict_list_new]
                self.estimators_ = self.estimators_ + [single_estimator]

        training_scores_dict = average_list_of_identical_dicts(training_scores_dict_list)
        validation_scores_dict = average_list_of_identical_dicts(validation_scores_dict_list)
        additional_validation_dict = average_list_of_identical_dicts(additional_validation_dict_list)
        

        return self, training_scores_dict, validation_scores_dict, additional_validation_dict

    def adjust_scaler_to_accept_distant_pricings(self, scaler, df):
        lowest_expected_close_price  = 2.5
        largest_expected_close_price = 132

        df_new = pd.DataFrame(columns=df.columns)
        #create min line
        min_dict = dict()
        max_dict = dict()
        for col in df.columns:
            if any(substring in col for substring in ["£_", "$_"]) and not "macd" in col:
                min_dict[col] = lowest_expected_close_price
                max_dict[col] = largest_expected_close_price
            else:
                min_dict[col] = min(df.loc[:,col])
                max_dict[col] = max(df.loc[:,col])
        df_new = pd.concat([df_new, pd.DataFrame([min_dict.values()], columns=min_dict.keys())], axis=0)
        df_new = pd.concat([df_new, pd.DataFrame([max_dict.values()], columns=max_dict.keys())], axis=0)
        scaler.fit(df_new)

        return scaler


    def custom_single_predict(self, df_X, single_estimator, return_df=False, independent_scaling=False, df_y=None):
        
        if independent_scaling == True:
            scaler_X=MinMaxScaler()
            scaler_X=scaler_X.fit(df_X)
            scaler_y=MinMaxScaler()
            scaler_y=scaler_y.fit(df_y)
        else:
            scaler_X=self.scaler_X
            scaler_y=self.scaler_y

        index, input_data   = return_lookback_appropriate_index_andor_data(df_X, self.lookbacks, return_index=True, return_input=True, scaler=scaler_X)
        y_pred_values       = single_estimator.predict(input_data, verbose=0)
        if not scaler_X == None:
            y_pred_values       = scaler_y.inverse_transform(y_pred_values)
        y_pred_values = pd.DataFrame(y_pred_values, index=index)

        return y_pred_values
        
    def evaluate(self, y_test, y_pred, outputs_params_dict, reporting_dict):
        
        pred_steps_value              = outputs_params_dict["pred_steps_ahead"]
        confidences_before_betting_PC = reporting_dict["confidence_thresholds"]

        #align indices
        if isinstance(y_pred, pd.Series):
            a = copy.deepcopy(y_pred)
            y_pred = pd.DataFrame(y_pred)
        merged_df = pd.merge(y_test, y_pred, left_index=True, right_index=True, how='inner')
        y_test = y_test.loc[merged_df.index]
        y_pred = y_pred.loc[merged_df.index]
        
        traditional_scores_dict_list = {"r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred)}
        additional_results_dict_list = FG_additional_reporting.return_results_X_min_plus_minus_accuracy(y_pred, y_test, pred_steps_value, confidences_before_betting_PC=confidences_before_betting_PC)
        return traditional_scores_dict_list, additional_results_dict_list
        
    def save(self, general_save_dir = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["predictive_model"], Y_preds_testing=None, y_testing=None):
        model_name = return_predictor_name(self.input_dict)
        folder_path = os.path.join(general_save_dir, custom_hash(model_name) + "\\")
        extension = ".h5"
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
            model.save(os.path.join(folder_path, f'ensemble_model_{i}.{extension}'))
        with open(os.path.join(folder_path,"input_dict.pkl"), "wb") as file:
                pickle.dump(self.input_dict, file)
        #save predictions if specified
        if isinstance(Y_preds_testing, pd.Series) or isinstance(Y_preds_testing, pd.DataFrame):
            Y_preds_testing = Y_preds_testing.shift(self.input_dict["outputs_params_dict"]["pred_steps_ahead"])
            Y_preds_testing.to_csv(os.path.join(folder_path, 'Y_preds_testing.csv'))
        if isinstance(y_testing, pd.Series) or isinstance(y_testing, pd.DataFrame):
            y_testing.to_csv(os.path.join(folder_path, 'y_testing.csv'))
   
    def predict_ensemble(self, X, y): #FG_action: This is where the new error is
        
        y_ensemble = pd.DataFrame()
        if self.model_hyper_params["scaler_cat"] == 3:
            independent_scaling = True
        else:
            independent_scaling = False
        
        for i, single_estimator in enumerate(self.estimators_):
            # randomly select features to drop out
            y_ensemble[i] = self.custom_single_predict(X, single_estimator, independent_scaling=independent_scaling, df_y=y)
        
        output = y_ensemble.mean(axis=1)

        return output
    
    def load(self, predictor_location_folder_path):
        folder_name = custom_hash(return_predictor_name(self.input_dict))
        folder_path = os.path.join(predictor_location_folder_path, folder_name+ "\\")
        print("xxx loading predictor")
        for filename in os.listdir(predictor_location_folder_path):
            if filename.endswith(".h5"):
                file_path = os.path.join(predictor_location_folder_path, filename)
                self.estimators_ += [load_model(file_path)]
        # check that the input parameters are the same
        with open(os.path.join(predictor_location_folder_path,"input_dict.pkl"), "rb") as file:
            save_input_dict = pickle.load(file)
        copy_A, copy_B = copy.deepcopy(save_input_dict), copy.deepcopy(self.input_dict)
        del copy_A["senti_inputs_params_dict"]["sentiment_method"], copy_B["senti_inputs_params_dict"]["sentiment_method"]
        del copy_A["temporal_params_dict"]['test_period_start'], copy_A["temporal_params_dict"]['test_period_end'], copy_B["temporal_params_dict"]['test_period_start'], copy_B["temporal_params_dict"]['test_period_end']
        if 'financial_value_scaling' in list(copy_A["fin_inputs_params_dict"].keys()) and copy_A["fin_inputs_params_dict"]['financial_value_scaling'] == None:
            del copy_A["temporal_params_dict"]['test_period_start']
            if 'financial_value_scaling' in list(copy_B["fin_inputs_params_dict"].keys()) and copy_B["fin_inputs_params_dict"]['financial_value_scaling'] == None:
                del copy_B["temporal_params_dict"]['test_period_start']
        if not copy_A == copy_B:
            raise ValueError("input dicts dont match")


def load_RNN_predictor(input_dict, predictor_location_folder_path):
        predictor = initiate_model(input_dict)
        predictor.load(predictor_location_folder_path)
        return predictor

def return_lookback_appropriate_index_andor_data(df_x, lookbacks, return_index=False, return_input=False, scaler=None):
    # this method, according to result bools, returns the index and input data so that time 
    # periods spanning two days are removed
    if return_index == False and return_input == False:
        raise ValueError("this method should at least request one of the outputs")

    output_input, output_index = [], []
    trim_from_indexes = lookbacks-1
    ori_index = df_x.index
    if not scaler == None:
        df_x = pd.DataFrame(scaler.transform(df_x), index=df_x.index, columns=df_x.columns)

    if return_index==True:
        for ts0, ts1 in zip(ori_index[:-trim_from_indexes], ori_index[trim_from_indexes:]):
            if ts0.day == ts1.day:
                output_index += [ts1]
        output_index = np.array(output_index)
        output_single = np.array(output_index)
    
    if return_input==True:
        for ts0, ts1 in zip(ori_index[:-trim_from_indexes], ori_index[trim_from_indexes:]):
            if ts0.day == ts1.day:
                output_input += [list(df_x.loc[ts0:ts1,:].values)]
                #output_input += [list(ori_index[ts0:ts1].values)]
        output_input = np.array(output_input)
        output_single = np.array(output_input)

    if return_index==True and return_input==True:
        return output_index, output_input
    else:
        return output_single


def return_filtered_batches_that_dont_cross_two_days(training_generator, datetime_generator):
    mask = np.array([np.datetime_as_string(batch_x[0][0], unit='D')[-2:] == np.datetime_as_string(batch_x[0][-1], unit='D')[-2:] for batch_x, _ in datetime_generator])
    mask = np.concatenate([mask,[False] * int(training_generator.data.shape[0] - len(mask))])
    new_data = training_generator.data[mask]
    new_targets = training_generator.targets[mask]

    ## Replace removed batches with random batches
    #for i in range(sum(mask), len(mask)):
    #    random_index = random.randint(0, sum(mask)-1)
    #    new_data = np.append(new_data, [new_data[random_index]], axis=0)
    #    new_targets = np.append(new_targets, [new_targets[random_index]], axis=0)

    # Transfer final batches directly
    new_data = np.append(new_data, training_generator.data[len(new_data):], axis=0)
    new_targets = np.append(new_targets, training_generator.targets[len(new_targets):], axis=0)

    training_generator.data = new_data
    training_generator.targets = new_targets
    return training_generator

def return_columns_to_remove(columns_list, self):
    
    columns_to_remove = list(copy.deepcopy(columns_list))
    retain_cols = []
    
    retain_dict = self.model_hyper_params["cohort_retention_rate_dict"]
    general_adjusting_square_factor = self.model_hyper_params["general_adjusting_square_factor"]

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
        cohort = cohort + fnmatch.filter(columns_list, target_string)
        if len(cohort) > 0:
            for col in cohort:
                columns_list.remove(col)
            retain_cols = retain_cols + list(np.random.choice(cohort, math.ceil(len(cohort) * (retain_dict[key] ** general_adjusting_square_factor)), replace=False))

    for value in retain_cols:
        columns_to_remove.remove(value)
    
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
    Y_preds_testing = predictor.predict_ensemble(X_testing, y_testing) # the y testing is used only for scaling and is justified as in the real world each day would be scaled off the previous day, instead of scaling outputs on 6 month old data like a standard timeline extrapolation
    
    # step 4: generate score
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 4: generate score")
    testing_scores, additional_results_dict = predictor.evaluate(y_testing, Y_preds_testing, outputs_params_dict, reporting_dict)
    predictor.save(Y_preds_testing=Y_preds_testing, y_testing=y_testing)
    if return_time_series == False:
        return testing_scores
    else:
        return testing_scores, X_testing, y_testing, Y_preds_testing


#%% main support methods

def retrieve_model_and_training_scores(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict):
    #predictor_location_file = "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\predictive_model\\aapl_ps04_06_18_000000_pe01_09_20_000000_ts_sec300_tm_qty7_r_lt3600_r_hl3600_tm_alpha1_IDF-True_t_ratio_r100000.pred"
    input_dict = return_input_dict(temporal_params_dict = temporal_params_dict, fin_inputs_params_dict = fin_inputs_params_dict, senti_inputs_params_dict = senti_inputs_params_dict, outputs_params_dict = outputs_params_dict, model_hyper_params = model_hyper_params, reporting_dict = reporting_dict)
    model = load_RNN_predictor(input_dict, predictor_location_folder_path)
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="training")
    #training_score = edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="load")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"], fin_inputs_params_dict["financial_value_scaling"])
    df_sentiment_data = retrieve_or_generate_sentiment_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
    
    X_train, y_train   = create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
    model, training_scores_dict, validation_scores_dict, additional_validation_dict = model.evaluate_ensemble(X_train, y_train, outputs_params_dict["pred_steps_ahead"], confidences_before_betting_PC=reporting_dict["confidence_thresholds"])
    
    return model, training_scores_dict, validation_scores_dict, additional_validation_dict

def generate_model_and_validation_scores(temporal_params_dict,
    fin_inputs_params_dict,
    senti_inputs_params_dict,
    outputs_params_dict,
    model_hyper_params,
    reporting_dict):
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
    model              = initiate_model(input_dict)
    model, training_scores_dict, validation_scores_dict, additional_validation_dict = model.fit_ensemble(X_train, y_train, outputs_params_dict["pred_steps_ahead"], confidences_before_betting_PC=reporting_dict["confidence_thresholds"])
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
    if os.path.exists(predictor_location_folder_path):
        predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = retrieve_model_and_training_scores(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
    else:
        print(datetime.now().strftime("%H:%M:%S") + " - generating model and testing scores")
        predictor, training_scores_dict, validation_scores_dict, additional_validation_dict = generate_model_and_validation_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
        predictor.save()
    return predictor, training_scores_dict, validation_scores_dict, additional_validation_dict

if __name__ == '__main__':
    predictor, traditional_training_scores, validation_dict = retrieve_or_generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
    testing_scores, Y_preds_testing                         = generate_testing_scores(predictor, input_dict)
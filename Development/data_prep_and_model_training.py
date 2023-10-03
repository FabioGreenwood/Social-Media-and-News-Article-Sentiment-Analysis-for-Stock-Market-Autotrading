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
from datetime import datetime
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import os
import warnings
import sys
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
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
from stock_indicators import PeriodSize, PivotPointType 

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



temporal_params_dict    = {
    "train_period_start"    : datetime.strptime('04/06/18 00:00:00', global_strptime_str),
    "train_period_end"      : datetime.strptime('01/09/20 00:00:00', global_strptime_str),
    "time_step_seconds"     : 5*60, #5 mins,
    "test_period_start"     : datetime.strptime('01/09/20 00:00:00', global_strptime_str),
    "test_period_end"       : datetime.strptime('01/01/21 00:00:00', global_strptime_str),
}

fin_inputs_params_dict      = {
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

senti_inputs_params_dict    = {
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
    "tweet_file_location"   : r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\data\twitter data\Tweets about the Top Companies from 2015 to 2020\Tweet.csv\Tweet.csv",
    "regenerate_cleaned_tweets_for_subject_discovery" : False
}

outputs_params_dict         = {
    "output_symbol_indicators_tuple"    : ("aapl", "close"),
    "pred_steps_ahead"                  : 1,
}

cohort_retention_rate_dict_strat1 = {
            "£_close" : 1, #output value
            "£_*": 1, #other OHLCV values
            "$_*" : 0.5, # technical indicators
            "match!_*" : 0.8, #pattern matchs
            "~senti_*" : 0.5, #sentiment analysis
            "*": 0.5} # other missed values

model_hyper_params          = {
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
    "cohort_retention_rate_dict"       : cohort_retention_rate_dict_strat1}

input_dict = {
    "temporal_params_dict"      : temporal_params_dict,
    "fin_inputs_params_dict"    : fin_inputs_params_dict,
    "senti_inputs_params_dict"  : senti_inputs_params_dict,
    "outputs_params_dict"       : outputs_params_dict,
    "model_hyper_params"        : model_hyper_params
    }

global_precalculated_assets_locations_dict = {
    "root" : "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\",
    "topic_models"              : "topic_models\\",
    "annotated_tweets"          : "annotated_tweets\\",
    "predictive_model"          : "predictive_model\\",
    "sentimental_data"          : "sentimental_data\\",
    "technical_indicators"      : "technical_indicators\\",
    "experiment_records"        : "experiment_records\\",
    "clean_tweets"              : "cleaned_tweets_ready_for_subject_discovery\\tweets.pkl"
    }



#%% misc methods
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
            
    
    

def return_topic_model_name(topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list):
    file_string = "tm_qty" + str(topic_model_qty) + "_tm_alpha" + str(topic_model_alpha) + "_IDF-" + str(apply_IDF) + "_t_ratio_r" + str(tweet_ratio_removed) + return_topic_mode_seed_hash(enforced_topic_model_nested_list)
    return file_string

def return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list):
    global global_strptime_str, global_strptime_str_filename
    name = company_symbol + "_ps" + train_period_start.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_pe" + train_period_end.strftime(global_strptime_str_filename).replace(":","").replace(" ","_") + "_" + str(weighted_topics) + "_"
    name = name + return_topic_model_name(topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list)
    return name

def return_sentimental_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, time_step_seconds, rel_lifetime, rel_hlflfe):
    global global_strptime_str, global_strptime_str_filename
    name = return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list)
    name = name + "_ts_sec" + str(time_step_seconds) + "_r_lt" + str(rel_lifetime) + "_r_hl" + str(rel_hlflfe)
    return name

def return_predictor_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, time_step_seconds, rel_lifetime, rel_hlflfe, pred_steps_ahead, hidden_layers, estm_alpha, model_hyper_params):
    global global_strptime_str, global_strptime_str_filename
    name = return_sentimental_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, time_step_seconds, rel_lifetime, rel_hlflfe)
    name = name + "_steps" + str(pred_steps_ahead) + "_hiddenL" + str(hidden_layers) + "estm_A" + str(estm_alpha) + "estimHASH" + str(hash(str(model_hyper_params)))
    return name
    


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
            output_string += " : etc: " + (start_time + (((1 - proportion_done) / proportion_done) * time_past)).strftime("%H:%M:%S")
        print(output_string)
        
    return curr_iter_num + 1



#%%SubModule – Stock Market Data Prep 

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
    df_financial_data = pd.read_csv(target_file_path)
    df_financial_data["date"] = df_financial_data["date"].str.replace("T", " ")
    df_financial_data["date"] = df_financial_data["date"].str.replace("Z", "")
    df_financial_data["date"] = df_financial_data["date"].str.replace(".000", "")
    df_financial_data["date"] = pd.to_datetime(df_financial_data["date"], format='%Y-%m-%d %H:%M:%S')
    df_financial_data.set_index("date", inplace=True)
    df_financial_data.index.names = ['datetime']
    index = list(df_financial_data.index)
    
    #check for faulty time windows
    if temporal_params_dict["train_period_start"] < min(index) - timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["test_period_start"] < min(index)  - timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["train_period_end"] > max(index)   + timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    if temporal_params_dict["test_period_end"] > max(index)    + timedelta(seconds=24*60*60):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    
    #check for the wrong timestep
    if not index[1] - index[0] - timedelta(seconds=5*60) == timedelta(0):
        raise ValueError("the financial data provided doesn't cover the experiment time window")
    
    #trim for time window
    mask_a = pd.to_datetime(index) > period_start
    mask_b = pd.to_datetime(index) < period_end
    mask = mask_a & mask_b
    df_financial_data[mask]
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
    


def retrieve_or_generate_then_populate_technical_indicators(df, tech_indi_dict, match_doji, target_file_path):
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


    return df


#%% SubModule – Sentiment Data Prep

def retrieve_or_generate_sentimental_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training"):
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
    
    #set period start/ends
    if training_or_testing == "training" or training_or_testing == "train":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    else:
        raise ValueError("the input " + str(training_or_testing) + " is wrong for the input training_or_testing")
       
    #method
    #search for predictor
    sentimental_data_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["sentimental_data"]
    #FG_action: check that the train_period_start is only used and not the test version, that is an error
    sentimental_data_name = return_sentimental_data_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, time_step_seconds, rel_lifetime, rel_hlflfe)
    sentimental_data_location_file = sentimental_data_folder_location_string + sentimental_data_name + ".csv"
    if os.path.exists(sentimental_data_location_file):
        df_sentimental_data = pd.read_csv(sentimental_data_location_file)
        df_sentimental_data.set_index(df_sentimental_data.columns[0], inplace=True)
        df_sentimental_data.index.name = "datetime"
    elif not os.path.exists(sentimental_data_location_file) and topic_model_qty > 0:
        df_sentimental_data = generate_sentimental_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing=training_or_testing)
        df_sentimental_data.to_csv(sentimental_data_location_file)
    else:
        df_sentimental_data = None
    return df_sentimental_data

def retrieve_sentimental_data():
    raise ValueError('needs writing') 
    return None

def generate_sentimental_data(index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, repeat_timer=10, training_or_testing="training", hardcode_df_annotated_tweets=None):
    
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
    start_time              = datetime.now()
    enforced_topic_model_nested_list = senti_inputs_params_dict["enforced_topics_dict"]

    if training_or_testing == "training" or training_or_testing == "train":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    else:
        global global_error_str_1
        raise ValueError(global_error_str_1.format(str(training_or_testing)))
    
    #search for annotated tweets
    annotated_tweets_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["annotated_tweets"]
    annotated_tweets_name = return_annotated_tweets_name(company_symbol, train_period_start, train_period_end, weighted_topics, num_topics, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list)
    annotated_tweets_location_file = annotated_tweets_folder_location_string + annotated_tweets_name + ".csv"
    print("approaching gate")
    if isinstance(hardcode_df_annotated_tweets, pd.DataFrame):
        print("gate works")
        df_annotated_tweets = hardcode_df_annotated_tweets
        print(datetime.now().strftime("%H:%M:%S") + " - generating sentimental data")
    elif os.path.exists(annotated_tweets_location_file):
        df_annotated_tweets = pd.read_csv(annotated_tweets_location_file)
        df_annotated_tweets.set_index(df_annotated_tweets.columns[0], inplace=True)
        df_annotated_tweets.index.name = "datetime"
    else:
        df_annotated_tweets = generate_annotated_tweets(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
        df_annotated_tweets.to_csv(annotated_tweets_location_file)
        print(datetime.now().strftime("%H:%M:%S") + " - generating sentimental data")
    
    #generate sentimental data from topic model and annotate tweets
    #index = generate_datetimes(train_period_start, train_period_end, seconds_per_time_steps)
    #df_sentiment_scores = pd.DataFrame()
    columns = []
    for i in range(num_topics):
        columns = columns + ["~senti_score_t" + str(i)]
    df_sentiment_scores = pd.DataFrame(index=index, columns=columns)
    
    #create the initial cohort of tweets to be looked at in a time window
    epoch_time          = datetime(1970, 1, 1)
    tweet_cohort_start  = (train_period_start - epoch_time) - timedelta(seconds=relavance_lifetime)
    tweet_cohort_end    = (train_period_start - epoch_time)
    tweet_cohort_start  = tweet_cohort_start.total_seconds()
    tweet_cohort_end    = tweet_cohort_end.total_seconds()
    tweet_cohort        = pd.DataFrame(columns=df_annotated_tweets.columns)
    tweet_cohort, df_annotated_tweets  = update_tweet_cohort(tweet_cohort, df_annotated_tweets, tweet_cohort_start, tweet_cohort_end)
    index_list = list(index)
    print(type(index_list))
    rep_num = int(len(index_list) / 10)
    
    count = 0; counter_len = len(index) # FG_Counter
    for time_step in index:
        #if index_list.index(time_step) % rep_num == 0:
        #    print(str(index_list.index(time_step)/ (rep_num * 10)))
        #    print(datetime.now().strftime("%H:%M:%S"))
            
        senti_scores = list(np.zeros(num_topics))
        tweet_cohort["post_date"] = tweet_cohort["post_date"].astype(float)
        tweet_cohort["~sent_overall"] = tweet_cohort["~sent_overall"].astype(float)
        pre_calc_time_overall = np.exp((- 3 / relavance_lifetime) * (tweet_cohort.loc[:, "post_date"] - tweet_cohort_start)) * tweet_cohort.loc[:, "~sent_overall"]
        for topic_num in range(num_topics):
            score_numer = sum(pre_calc_time_overall * tweet_cohort.loc[:, "~sent_topic_W" + str(topic_num)])
            score_denom = sum(np.exp((- 3 / relavance_lifetime) * (tweet_cohort.loc[:, "post_date"] - tweet_cohort_start)) * tweet_cohort.loc[:, "~sent_topic_W" + str(topic_num)])
            if score_numer > 0 and score_denom > 0:
                senti_scores[topic_num] = score_numer / score_denom
        #update table
        df_sentiment_scores.loc[time_step, :] = senti_scores
        #update tweet cohort
        tweet_cohort_start += seconds_per_time_steps
        tweet_cohort_end   += seconds_per_time_steps
        tweet_cohort, df_annotated_tweets = update_tweet_cohort(tweet_cohort, df_annotated_tweets, tweet_cohort_start, tweet_cohort_end)
        #timer
        if repeat_timer>0:
            count = fg_timer(count, counter_len, repeat_timer, task_name="sentimental_tweets", start_time=start_time)
            
    
    return df_sentiment_scores

def generate_annotated_tweets(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, repeat_timer=10, training_or_testing="training", hardcode_location_for_topic_model=""):
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
    start_time          = datetime.now()
    
    if training_or_testing == "training" or training_or_testing == "train":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    elif training_or_testing == "testing" or training_or_testing == "test":
        train_period_start  = temporal_params_dict["train_period_start"]
        train_period_end    = temporal_params_dict["train_period_end"]
    else:
        global global_error_str_1
        raise ValueError(global_error_str_1.format(str(training_or_testing)))
    
    
    
    #search for topic_model
    topic_model_folder_folder = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["topic_models"]
    
    topic_model_name = return_topic_model_name(num_topics, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list)
    topic_model_location_file = topic_model_folder_folder + topic_model_name
    if not hardcode_location_for_topic_model=="":
        with open(hardcode_location_for_topic_model, 'rb') as file:
            topic_model_dict = pickle.load(file)
    elif os.path.exists(topic_model_location_file + "topic_model_dict_" + topic_model_name + ".pkl"):
        with open(topic_model_location_file + "topic_model_dict_" + topic_model_name + ".pkl", "rb") as file:
            topic_model_dict = pickle.load(file)
    else:
        wordcloud, topic_model_dict, visualisation = generate_and_save_topic_model(topic_model_name, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
  
    
    #generate annotated tweets
    print(datetime.now().strftime("%H:%M:%S") + " - generating annotated tweets")
    df_annotated_tweets   = import_twitter_data_period(senti_inputs_params_dict["tweet_file_location"], train_period_start, train_period_end, relavance_lifetime, tweet_ratio_removed)
    columns_to_add = ["~sent_overall"]
    for num in range(num_topics):
        columns_to_add = columns_to_add + ["~sent_topic_W" + str(num)]
    
    df_annotated_tweets[columns_to_add] = float("nan")
    count = 0; counter_len = len(df_annotated_tweets.index) # FG_Counter
    
    for tweet_id in df_annotated_tweets.index:
        
        text = df_annotated_tweets["body"][tweet_id]
        sentiment_value = sentiment_method.polarity_scores(text)["compound"] #FG_Action: this needs to be checked 
        topic_tuples = return_topic_weight(text, topic_model_dict, num_topics)
        if len(topic_tuples) == num_topics:
            topic_weights = [t[1] for t in topic_tuples]
        else:
            topic_weights = list(np.zeros(num_topics))
            for tup in topic_tuples:
                topic_weights[tup[0]] = tup[1]
        sentiment_analysis = [sentiment_value] + topic_weights
        df_annotated_tweets.loc[tweet_id, columns_to_add] = sentiment_analysis
        #timer
        if repeat_timer>0:
            count = fg_timer(count, counter_len, repeat_timer, task_name="annotated_data", start_time=start_time)
    
    return df_annotated_tweets

def return_topic_weight(text_body, topic_model_dict, num_topics):
    if num_topics > 1:
        id2word = topic_model_dict["id2word"]
        lda_model = topic_model_dict["lda_model"]
        bow_doc = id2word.doc2bow(text_body.split(" "))
        doc_topics = lda_model.get_document_topics(bow_doc)
    else:
        doc_topics = [(0,1)]
    return doc_topics

def generate_and_save_topic_model(run_name, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params):
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
    global global_df_stocks_list_file
    df_prepped_tweets_company_agnostic_file_path = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["clean_tweets"]
    if reclean_tweets == True:
        df_prepped_tweets_company_agnostic          = prep_and_save_twitter_text_for_subject_discovery(df_prepped_tweets["body"], df_stocks_list_file=global_df_stocks_list_file, df_prepped_tweets_company_agnostic_file_path=df_prepped_tweets_company_agnostic_file_path)
    else:
        with open(df_prepped_tweets_company_agnostic_file_path, 'rb') as file:
            df_prepped_tweets_company_agnostic = pickle.load(file)
    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Creating Subject Keys --------------------------------")
    wordcloud, topic_model_dict, visualisation  = return_subject_keys(df_prepped_tweets_company_agnostic, topic_qty = num_topics, topic_model_alpha=topic_model_alpha, apply_IDF=apply_IDF,
                                                                      enforced_topics_dict=enforced_topics_dict, return_LDA_model=True, return_png_visualisation=True, return_html_visualisation=True)
    save_topic_clusters(wordcloud, topic_model_dict, visualisation, file_location_wordcloud, file_location_topic_model_dict, file_location_visualisation)
    print(datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------- Completed Subject Keys --------------------------------")
    print(datetime.now().strftime("%H:%M:%S"))
    return wordcloud, topic_model_dict, visualisation

def import_twitter_data_period(target_file, period_start, period_end, relavance_lifetime, tweet_ratio_removed):
    #prep data
    input_df = pd.read_csv(target_file)
    epoch_time  = datetime(1970, 1, 1)
    period_start -= timedelta(seconds=relavance_lifetime)
    epoch_start = (period_start - epoch_time).total_seconds()
    epoch_end   = (period_end - epoch_time).total_seconds()
    
    #trim according to time window    
    input_df = input_df[input_df["post_date"]>epoch_start]
    input_df = input_df[input_df["post_date"]<epoch_end]
    
    if tweet_ratio_removed > 1:
        new_index = input_df.index[::tweet_ratio_removed]
        input_df = input_df.loc[new_index]
    
    return input_df

from datetime import datetime, timedelta
def convert_epoch_time_to_datetime_string(post_date):
    period = datetime(1970, 1, 1) + timedelta(seconds=post_date)
    return period.strftime('%d/%m/%y %H:%M:%S')

from datetime import datetime, timedelta
def return_time(post_date):
    period = datetime(1970, 1, 1) + timedelta(seconds=post_date)
    return period.strftime('%d/%m/%y %H:%M:%S')


def prep_and_save_twitter_text_for_subject_discovery(input_list, df_stocks_list_file=None, df_prepped_tweets_company_agnostic_file_path=None):
    global global_precalculated_assets_locations_dict
    #prep parameters
    death_characters    = ["$", "amazon", "apple", "goog", "tesla", "http", "@", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "'s", "compile", "www"]
    stocks_list         = list(df_stocks_list_file["Name"].map(lambda x: x.lower()).values)
    tickers_list        = list(df_stocks_list_file["Ticker"].map(lambda x: x.lower()).values)
    stopwords_english   = stopwords.words('english')
    #these are words are removed from company names to create additional shortened versions of those names. This is so these version can be eliminated from the tweets to make the subjects agnostic
    corp_stopwords      = [".com", "company", "corp", "froup", "fund", "gmbh", "global", "incorporated", "inc.", "inc", "tech", "technology", "technologies", "trust", "limited", "lmt", "ltd"]
    #these are words are directly removed from tweets
    misc_stopwords      = ["iphone", "airpods", "jeff", "bezos", "#microsoft", "#amzn", "volkswagen", "microsoft", "amazon's", "tsla", "androidwear", "ipad", "amzn", "iphone", "tesla", "TSLA", "elon", "musk", "baird", "robert", "pm", "androidwear", "android", "robert", "ab", "ae", "dlvrit", "https", "iphone", "inc", "new", "dlvrit", "py", "twitter", "cityfalconcom", "aapl", "ing", "ios", "samsung", "ipad", "phones", "cityfalconcom", "us", "bitly", "utmmpaign", "etexclusivecom", "cityfalcon", "owler", "com", "stock", "stocks", "buy", "bitly", "dlvrit", "alexa", "zprio", "billion", "seekalphacom", "et", "alphet", "seekalpha", "googl", "zprio", "trad", "jt", "windows", "adw", "ifttt", "ihadvfn", "nmona", "pphppid", "st", "bza", "twits", "biness", "tim", "ba", "se", "rat", "article"]


    #prep stocks_list_shortened
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
    for tweet in split_tweets:
        for word in reversed(tweet):
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
    
    
    if not df_prepped_tweets_company_agnostic_file_path == None:
        with open(df_prepped_tweets_company_agnostic_file_path, "wb") as file:
            pickle.dump(output, file)
    
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



def return_subject_keys(df_prepped_tweets_company_agnostic, topic_qty = 10, enforced_topics_dict=None, stock_names_list=None, words_to_remove = None, 
                        return_LDA_model=True, return_png_visualisation=False, return_html_visualisation=False, 
                        topic_model_alpha=0.1, apply_IDF=True, cores=2, passes=20, iterations=400):
    output = []

    data = df_prepped_tweets_company_agnostic
    data_words = list(sent_to_words(data))
    if return_LDA_model < return_html_visualisation:
        raise ValueError("You must return the LDA visualisation if you return the LDA model")

       
    if return_png_visualisation==True:
        long_string = "start"
        for w in data_words:
            long_string = long_string + ',' + ','.join(w)
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
        wordcloud.generate(long_string)
        wordcloud.to_image()
        output = output + [wordcloud]
    else:
        output = output + [None]
    
    if return_LDA_model==True:
        # Create Dictionary
        id2word = corpora.Dictionary(data_words)

        # Create Corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        #translate the enforced_topics_dict input
        if enforced_topics_dict != None:
            if len(id2word.id2token) == 0:
                #manual transfer to id2token, required.
                for key in id2word.token2id:
                    id2word.id2token[id2word.token2id[key]] = key
            eta = return_initial_eta(topic_qty, enforced_topics_dict, id2word)
        else:
            eta='auto' # Let Gensim estimate eta (topic-word distribution)
        #eta = np.zeros(len(id2word.id2token)) #FG_action: remove
        #apply IDF
        if apply_IDF == True:
            # create tfidf model
            tfidf = TfidfModel(corpus)

            # apply tfidf to corpus
            corpus = tfidf[corpus]
        
        # Build LDA model
        lda_model = gensim.models.LdaModel(corpus=corpus,
        #lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topic_qty,
                                                alpha = topic_model_alpha, # controls topic sparsity:- Set a low alpha to make topics more focused
                                                eta=eta, # sets the initial topic-word distribution
                                                passes=passes,  # Number of training iterations
                                                iterations=iterations,  # Number of iterations per pass
                                                random_state=42)
        
        # Print the Keyword in the 10 topics
        #pprint(lda_model.print_topics())
        #eta = lda_model.expElogbeta
        doc_lda = lda_model[corpus]
        topic_model_dict = {"lda_model" : lda_model, "doc_lda" : doc_lda, "corpus" : corpus, "id2word" : id2word}
        output = output + [topic_model_dict]
    else:
        output = output + [None]
            
    if return_html_visualisation==True:
        pyLDAvis.enable_notebook
        if topic_qty > 1:
            LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
        else:
            LDAvis_prepared = None
        output = output + [LDAvis_prepared]
    else:
        output = output + [None]
    
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
        pyLDAvis.save_html(visualisation, file_location_visualisation)

def edit_scores_csv(predictor_name_entry, is_training_or_testing, score_types, mode="save", training_scores=None):
    global global_scores_database, global_strptime_str_filename
    DB_strptime_str = '%d/%m/%Y %H:%M:%S'

    ID_str          = "ID"
    first_col_str   = "company_symbol (ps)"
    last_col_str    = "tweet_ratio_removed (t_ratio_r)"
    ERROR_multi_input = "There are multiple entries of the following experiment: {}"
    ERROR_mode_input_wrong = "'mode' input should be 'save' or 'load', recieved {}"
    ERROR_is_training_or_testing_input_wrong= "'is_training_or_testing' input should be 'training' or 'testing', recieved {}"
    score_col_str    =  is_training_or_testing + "_{}" # + score_type
    
    #expection handling
    if not is_training_or_testing == "training" and not is_training_or_testing == "testing":
        raise ValueError(ERROR_is_training_or_testing_input_wrong.format(is_training_or_testing))
        
    #format values
    company_symbol, train_period_start, train_period_end, time_step_seconds, topic_model_qty, rel_lifetime, rel_hlflfe, topic_model_alpha, apply_IDF, tweet_ratio_removed = predictor_name_entry
    train_period_start  = train_period_start.strftime(DB_strptime_str)
    train_period_end    = train_period_end.strftime(DB_strptime_str)
    apply_IDF           = str(apply_IDF)
    predictor_name_entry = company_symbol, train_period_start, train_period_end, time_step_seconds, topic_model_qty, rel_lifetime, rel_hlflfe, topic_model_alpha, apply_IDF, tweet_ratio_removed
    
    #prep DB, try using dont touch
    try:
        df_scores_database = pd.read_csv(global_scores_database + "DONTTOUCH", index_col=ID_str)
    except:
        df_scores_database = pd.DataFrame(columns=[ID_str, "company_symbol (ps)", "train_period_start (ps)", "train_period_end (pe)", "time_step_seconds (ts_sec)", "topic_model_qty (tm_qty)", "rel_lifetime (r_lt)", "rel_hlflfe (r_hl)", "topic_model_alpha (tm_alpha)", "apply_IDF (IDF)", "tweet_ratio_removed (t_ratio_r)", "training_r2", "training_mse", "training_mae", "testing_r2", "testing_mse",  "testing_mae"])
        df_scores_database = df_scores_database.set_index("ID")
                
    #format datetimes
    df_scores_database['train_period_start (ps)'] = df_scores_database['train_period_start (ps)'].apply(lambda x: str(x))
    df_scores_database['train_period_end (pe)'] = df_scores_database['train_period_end (pe)'].apply(lambda x: str(x))
        
    #find matches
    value_qty = len(df_scores_database.loc[:,first_col_str:last_col_str].columns)
    previous_values = df_scores_database.loc[:,"company_symbol (ps)":"tweet_ratio_removed (t_ratio_r)"]==predictor_name_entry
    previous_entries_indexes = list((previous_values[previous_values.sum(axis=1) == value_qty]).index)
    
    
    if mode=="load":
        output = dict()
        if len(previous_entries_indexes) == 1:
            for val in score_types:
                output[val] = df_scores_database.loc[previous_entries_indexes[0], score_col_str.format(val)]
            return output
        elif len(previous_entries_indexes) > 1:
            raise ValueError(ERROR_multi_input.format(str(predictor_name_entry)))
        else:
            output["na"] = math.nan
            return output
        
    elif mode=="save":
        #assign correct ID, given matches
        if len(previous_entries_indexes) == 1:
            ID = previous_entries_indexes[0]
        elif len(previous_entries_indexes) > 1:
            raise ValueError(ERROR_multi_input.format(str(predictor_name_entry)))
        else:
            ID = df_scores_database.index.max() + 1
            if math.isnan(ID):
                ID = 0
            df_scores_database.loc[ID,first_col_str:last_col_str] = list(predictor_name_entry)

        #populate score
        for val in score_types:
            df_scores_database.loc[ID, score_col_str.format(val)] = training_scores[val]

        #save
        try:
            df_scores_database.to_csv(global_scores_database)
            df_scores_database.to_csv(global_scores_database + "DONTTOUCH")
        except:
            df_scores_database.to_csv(global_scores_database + "DONTTOUCH")
            
            
    else:
        ERROR_mode_input_wrong.format(mode)
    
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def update_tweet_cohort(tweet_cohort, df_annotated_tweets_temp, tweet_cohort_start, tweet_cohort_end):
    epoch_time          = datetime(1970, 1, 1)
    #delete old tweets
    tweet_cohort                = tweet_cohort[tweet_cohort["post_date"] <= tweet_cohort_start]
    
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
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def create_step_responces(df_financial_data, df_sentimental_data, pred_output_and_tickers_combos_list, pred_steps_ahead):
    #this method populates each row with the next X output results, this is done so that, each time step can be trained
    #to predict the value of the next X steps
    
    new_col_str = "{}_{}"
    list_of_new_columns = []
    nan_values_replaced = 0
    train_test_split = 1
    if isinstance(df_sentimental_data, pd.DataFrame):
        df_sentimental_data.index = pd.to_datetime(df_sentimental_data.index)
        df_sentimental_data = df_sentimental_data.loc[list(df_financial_data.index)]
    data = pd.concat([df_financial_data, df_sentimental_data], axis=1, ignore_index=False)
    #this method has been removed cos its doesn't work with negative values
    #df_financial_data, nan_values_replaced = current_infer_values_method(df_financial_data)
    
    #create regressors
    symbol, old_col = pred_output_and_tickers_combos_list
    old_col = "£_" + old_col
    
    if isinstance(pred_steps_ahead,list):
        raise ValueError("only runs with singluar pred_steps_list values are allow, no lists")
        
    new_col = new_col_str.format(old_col, pred_steps_ahead)
    list_of_new_columns = list_of_new_columns + [new_col]
    data[new_col] = data[old_col].shift(-pred_steps_ahead)

    #split regressors and responses
    #Features = 6

    data = data.dropna(inplace=False)

    X = copy.deepcopy(data)
    y = copy.deepcopy(data[list_of_new_columns])
    
    for col in list_of_new_columns:
        X = X.drop(col, axis=1)
    
    return X, y

def current_infer_values_method(df):
    raise BrokenPipeError("This method isn't fit for current purposes")
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


#create model

def initiate_model(outputs_params_dict, model_hyper_params):
    if model_hyper_params["name"] == "RandomSubspace_MLPRegressor":
            estimator = DRSLinReg(base_estimator=MLPRegressor(
                hidden_layer_sizes  = model_hyper_params["estimator__hidden_layer_sizes"],
                activation          = model_hyper_params["estimator__activation"],
                alpha               = model_hyper_params["estimator__alpha"]
                ),
                model_hyper_params=model_hyper_params, 
                ticker_name=outputs_params_dict["output_symbol_indicators_tuple"][0])
    #elif model_hyper_params["name"] == "SimpleRNN":
    #    estimator = Sequential()
    #    estimator.add(SimpleRNN(units=32, activation='relu', input_shape=(sequence_length, input_dim)))
    #    estimator.add(Dense(output_dim))
    
    elif model_hyper_params["name"] == "ElasticNet": 
            raise ValueError("OLD SYSTEM WILL NOT WORK")
            keys = ["estimator__alpha", "estimator__l1_ratio"]
            #check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = BaggingRegressor(
                MultiOutputRegressor(
                ElasticNet(
                    alpha       = model_hyper_params["ElasticNet"]["estimator__alpha"],
                    l1_ratio    = model_hyper_params["ElasticNet"]["estimator__l1_ratio"],
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
    elif model_hyper_params["name"] == "MLPRegressor":
            raise ValueError("OLD SYSTEM WILL NOT WORK")
            keys = ["estimator__hidden_layer_sizes", "estimator__activation"]
            #check_dict_keys_for_build_model(keys, input_dict, type_str)
            estimator = MLPRegressor(
                activation          = model_hyper_params["MLPRegressor"]["estimator__activation"],
                hidden_layer_sizes  = model_hyper_params["MLPRegressor"]["estimator__hidden_layer_sizes"],
                alpha=0.001,
                random_state=20,
                early_stopping=False)
    else:
        raise ValueError("the model type: " + str(model_hyper_params["name"]) + " was not found in the method")
    
    return estimator

class custom_simpleRNN():
    def __init__(self, base_estimator=MLPRegressor(),
                 model_hyper_params=model_hyper_params, 
                 ticker_name=outputs_params_dict["output_symbol_indicators_tuple"][0]):
        #expected keys: training_time_splits, max_depth, max_features, random_state,        
        for key in model_hyper_params:
           setattr(self, key, model_hyper_params[key])
        self.model_hyper_params = model_hyper_params
        self.ticker_name = ticker_name
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.training_time_splits = model_hyper_params["time_series_split_qty"]
        self.n_estimators = 1 #this is a hard coding as the n estimators is set by a random loop. this ensures that each version of the input (provided by the 'remove columns' function), is used to train just a single model

class DRSLinReg():
    def __init__(self, base_estimator=MLPRegressor(),
                 model_hyper_params=model_hyper_params, 
                 ticker_name=outputs_params_dict["output_symbol_indicators_tuple"][0]):
        #expected keys: training_time_splits, max_depth, max_features, random_state,        
        for key in model_hyper_params:
           setattr(self, key, model_hyper_params[key])
        self.model_hyper_params = model_hyper_params
        self.ticker_name = ticker_name
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.training_time_splits = model_hyper_params["time_series_split_qty"]
        self.n_estimators = 1 #this is a hard coding as the n estimators is set by a random loop. this ensures that each version of the input (provided by the 'remove columns' function), is used to train just a single model
        
    def fit(self, X, y):
        tscv = BlockingTimeSeriesSplit(n_splits=self.training_time_splits)
        count = 0
        global global_random_state
        for train_index, _ in tscv.split(X):
            #these are the base values that will be updated if there isn't a passed value in the input dict
            estimator = BaggingRegressor(base_estimator=self.base_estimator,
                                          #the assignment of "one" estimator is overwritten by the rest of the method
                                          n_estimators=self.n_estimators,
                                          max_samples=1.0,
                                          max_features=1.0,
                                          bootstrap=True,
                                          bootstrap_features=False)
                                          #random_state=self.random_state SET ELSEWHERE
            for key in self.model_hyper_params:
                setattr(estimator, key, self.model_hyper_params[key])
            
            estimator.base_estimator = self.base_estimator
            
            count += 1
            for i_random in range(model_hyper_params["n_estimators_per_time_series_blocking"]):
                # randomly select features to drop out
                n_features = X.shape[1]
                dropout_cols = return_columns_to_remove(columns_list=X.columns, self=self)
                X_sel = X.loc[X.index[train_index].values].copy()
                X_sel.loc[:, dropout_cols] = 0
                y_sel= y.loc[y.index[train_index].values].copy()
                #add max depth if it is a decision tree
                estimator.random_state = global_random_state
                global_random_state += 1
                estimator.dropout_cols_ = dropout_cols
                estimator.fit(X_sel, y_sel)
                self.estimators_ = self.estimators_ + [estimator]
            
            #self.estimators_.append(estimator)
        return self

    def predict_ensemble(self, X, output_name=None):
        
        if output_name==None:
            raise ValueError ("please ensure that the outputs are labelled")
            output_name = ["output"]
        
        y_ave = []
        y_ensemble = []
        
        for i, estimator_local in enumerate(self.estimators_):
            # randomly select features to drop out
            y_ensemble.insert(len(y_ensemble), estimator_local.predict(X))
        
        y_ensemble = np.array(y_ensemble)
        for ts in range(len(y_ensemble[0])):
            y_ave = y_ave + [y_ensemble[:,ts].mean()]
        
        output = pd.DataFrame(y_ave, columns=[output_name], index=X.index)
        
        return output
    
    def evaluate(self, X_test=None, y_test=None, y_pred=None, method=["r2"], return_high_good=False):
        
        output = dict()
        
        if y_pred is None:
            y_pred = self.predict_ensemble(X_test, output_name=["test output"]).values
        
        for val in method:
            if val == "r2":
                output[val] = r2_score(y_test, y_pred)
            elif val == "mse":
                output[val] = mean_squared_error(y_test, y_pred)
            elif val == "mae":
                output[val] = mean_absolute_error(y_test, y_pred)
            else:
                raise ValueError("passed method string not found")
        return output

def return_columns_to_remove(columns_list, self):
    
    columns_to_remove = list(copy.deepcopy(columns_list))
    retain_cols = []
    retain_dict = self.model_hyper_params["cohort_retention_rate_dict"]
    #max_features = self.max_features
    stock_strings_list = []
    columns_list = list(columns_list)
    #for a in self.ticker_name:
    #    stock_strings_list = stock_strings_list + [a[0]]
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
            retain_cols = retain_cols + list(np.random.choice(cohort, math.ceil(len(cohort) * retain_dict[key]), replace=False))

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
    
    # step 1a: generate testing input data (finanical)
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 1a: generate testing input data")
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="testing")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"])

    # step 1b: generate testing input data (sentimental)
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 1b: generate testing input data (sentimental)")
    df_sentimental_data = retrieve_or_generate_sentimental_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="testing")

    # step 2: generate testing output data
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 2: generate testing output data")
    X_testing, y_testing   = create_step_responces(df_financial_data, df_sentimental_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"])
        
    # step 3: generate y_pred
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 3: generate y_pred")
    Y_preds_testing = predictor.predict_ensemble(X_testing, output_name=["77777777"])
    
    # step 4: generate score
    print(datetime.now().strftime("%H:%M:%S") + " - testing - step 4: generate score")
    testing_scores = predictor.evaluate(y_test=y_testing, y_pred=Y_preds_testing, method=model_hyper_params["testing_scoring"])
    if return_time_series == False:
        return testing_scores
    else:
        return testing_scores, X_testing, y_testing, Y_preds_testing



#%% main support methods

def retrieve_model_and_training_scores(predictor_location_file, predictor_name_entry, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params):
    #predictor_location_file = "C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\predictive_model\\aapl_ps04_06_18_000000_pe01_09_20_000000_ts_sec300_tm_qty7_r_lt3600_r_hl3600_tm_alpha1_IDF-True_t_ratio_r100000.pred"
    with open(predictor_location_file, 'rb') as file:
        predictor = pickle.load(file)
    training_score = edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="load")
    
    if any(math.isnan(value) for value in list(training_score.values())):
    #if training_score is None:
        training_score = quick_training_score_rerun(predictor, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
        edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="save", training_scores=training_score)

    return predictor, training_score

def generate_model_and_training_scores(temporal_params_dict,
    fin_inputs_params_dict,
    senti_inputs_params_dict,
    outputs_params_dict,
    model_hyper_params):
    #desc
    
    
    #general parameters
    global global_index_cols_list, global_input_cols_to_include_list
    
    
    #stock market data prep
    print(datetime.now().strftime("%H:%M:%S") + " - importing and prepping financial data")
    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="training")
    print(datetime.now().strftime("%H:%M:%S") + " - populate_technical_indicators")
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"])

    if senti_inputs_params_dict["topic_qty"] >= 1:
        #sentiment data prep
        print(datetime.now().strftime("%H:%M:%S") + " - importing or prepping sentimental data")
        df_sentimental_data = retrieve_or_generate_sentimental_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
    elif senti_inputs_params_dict["topic_qty"] == 0:
        df_sentimental_data = None
    
    #model training - time series blocking
    if model_hyper_params["time_series_blocking"] == "btscv":
        time_series_spliting = BlockingTimeSeriesSplit(n_splits=model_hyper_params["time_series_split_qty"])
    else:
        raise ValueError("model_hyper_params['time_series_blocking'], not recognised")
        
    #model training - create regressors
    X_train, y_train   = create_step_responces(df_financial_data, df_sentimental_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"])
    model              = initiate_model(outputs_params_dict, model_hyper_params)
    model.fit(X_train, y_train)
    training_scores = model.evaluate(X_test=X_train , y_test=y_train, method=model_hyper_params["testing_scoring"])
      
    
    #report timings
    print(datetime.now().strftime("%H:%M:%S") + " - complete generating model")
    global global_start_time
    total_run_secs      = (datetime.now() - global_start_time).total_seconds()
    total_run_hours     = total_run_secs // 3600
    total_run_minutes   = (total_run_secs % 3600) // 60
    total_run_seconds   = total_run_secs % 60
    report = f"{int(total_run_hours)} hours, {int(total_run_minutes)} minutes, {int(total_run_seconds)} seconds"
    print(report)
    return model, training_scores


def quick_training_score_rerun(model, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params):

    df_financial_data = import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="training")
    
    df_financial_data = retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"])

    #sentiment data prep
    print(datetime.now().strftime("%H:%M:%S") + " - importing or prepping sentimental data")
    df_sentimental_data = retrieve_or_generate_sentimental_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
    
    
    #model training - time series blocking
    if model_hyper_params["time_series_blocking"] == "btscv":
        time_series_spliting = BlockingTimeSeriesSplit(n_splits=model_hyper_params["time_series_split_qty"])
    else:
        raise ValueError("model_hyper_params['time_series_blocking'], not recognised")
        
    #model training - create regressors
    X_train, y_train   = create_step_responces(df_financial_data, df_sentimental_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"])
    training_scores = model.evaluate(X_test=X_train , y_test=y_train, method=model_hyper_params["testing_scoring"])
    return training_scores


#%% main line

def retrieve_or_generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params):
    
    #global values
    global global_financial_history_folder_path, global_precalculated_assets_locations_dict
    
    #general parameters
    company_symbol      = outputs_params_dict["output_symbol_indicators_tuple"][0]
    pred_steps          = outputs_params_dict["pred_steps_ahead"]
    train_period_start  = temporal_params_dict["train_period_start"]
    train_period_end    = temporal_params_dict["train_period_end"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    time_step_seconds   = temporal_params_dict["time_step_seconds"]
    topic_model_qty     = senti_inputs_params_dict["topic_qty"]
    rel_lifetime        = senti_inputs_params_dict["relative_lifetime"]
    rel_hlflfe          = senti_inputs_params_dict["relative_halflife"]
    topic_model_alpha   = senti_inputs_params_dict["topic_model_alpha"]
    tweet_ratio_removed = senti_inputs_params_dict["topic_training_tweet_ratio_removed"]
    apply_IDF           = senti_inputs_params_dict["apply_IDF"]
    weighted_topics     = senti_inputs_params_dict["weighted_topics"]
    hidden_layers       = model_hyper_params["estimator__hidden_layer_sizes"]
    estm_alpha          = model_hyper_params["estimator__alpha"]
    pred_steps_ahead    = outputs_params_dict["pred_steps_ahead"]
    enforced_topic_model_nested_list = senti_inputs_params_dict["enforced_topics_dict"]
            
    #search for predictor
    predictor_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["predictive_model"]
    predictor_name_entry = company_symbol, train_period_start, train_period_end, time_step_seconds, topic_model_qty, rel_lifetime, rel_hlflfe, topic_model_alpha, apply_IDF, tweet_ratio_removed
    predictor_name = return_predictor_name(company_symbol, train_period_start, train_period_end, weighted_topics, topic_model_qty, topic_model_alpha, apply_IDF, tweet_ratio_removed, enforced_topic_model_nested_list, time_step_seconds, rel_lifetime, rel_hlflfe, pred_steps_ahead, hidden_layers, estm_alpha, model_hyper_params)
    predictor_location_file = predictor_folder_location_string + predictor_name + ".pred"
    #previous_score = edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="load")
    if os.path.exists(predictor_location_file):
        predictor, training_scores = retrieve_model_and_training_scores(predictor_location_file, predictor_name_entry, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
    else:
        print(datetime.now().strftime("%H:%M:%S") + " - generating model and testing scores")
        predictor, training_scores = generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
        with open(predictor_location_file, "wb") as file:
            pickle.dump(predictor, file)
        edit_scores_csv(predictor_name_entry, "training", model_hyper_params["testing_scoring"], mode="save", training_scores=training_scores)

    return predictor, training_scores

if __name__ == '__main__':
    predictor, training_scores      = retrieve_or_generate_model_and_training_scores(temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params)
    testing_scores, Y_preds_testing = generate_testing_scores(predictor, input_dict)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import data_prep_and_model_training as FG_model_training
import additional_reporting_and_model_trading_runs as FG_additional_reporting
import experiment_handler as FG_experiment_handler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
global_strptime_str = '%d/%m/%y %H:%M:%S'



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

save_location = [r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\always_up_down_results"]

df_financial_data = FG_model_training.import_financial_data(
        target_file_path          = fin_inputs_params_dict["historical_file"], 
        input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
        temporal_params_dict = temporal_params_dict, training_or_testing="testing")
X_test = df_financial_data
design_history_dict = dict()
design_space_dict = design_space_dict = {"bet": {"direction" : ["up", "down"]}}
df_designs_record = pd.DataFrame()
pred_steps_list = [1,3,5,15]

for delta, dir_str, id in zip([1,-1], ["up", "down"], range(2)):

    
    design_history_dict[id] = dict()
    design_history_dict[id]["X"] = [dir_str]
    design_history_dict[id]["additional_results_dict"] = dict()
    
    for pred_steps in pred_steps_list:
        preds = X_test["£_close"] + delta
        #preds = preds.shift(-pred_steps)
        results_tables_dict = FG_additional_reporting.run_additional_reporting(
            X_test = X_test,
            preds = preds,
            pred_steps_list = [pred_steps],
            pred_output_and_tickers_combos_list = None,
            DoE_orders_dict = None,
            model_type_name = "always_up",
            outputs_path = None,
            model_start_time = None,
            confidences_before_betting_PC=[0])
        #integrate the predictions for the given time step in the larger results
        for result_type in results_tables_dict.keys():
            if not result_type in design_history_dict[id]["additional_results_dict"].keys():
                design_history_dict[id]["additional_results_dict"][result_type] = dict()
            for pred_step in results_tables_dict[result_type].keys():
                design_history_dict[id]["additional_results_dict"][result_type][pred_step] = results_tables_dict[result_type][pred_step]

        
        
        
    
    df_designs_record   = FG_experiment_handler.update_df_designs_record(df_designs_record, design_history_dict, design_space_dict, skip_column_check=True)
    FG_experiment_handler.save_designs_record_csv_and_dict(save_location, df_designs_record=df_designs_record, design_history_dict=design_history_dict, optim_run_name="always_up_down")



print("hello")

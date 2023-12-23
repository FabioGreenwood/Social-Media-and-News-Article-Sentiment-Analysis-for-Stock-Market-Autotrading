import pandas as pd
from datetime import datetime
import os


global_general_folder = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\"
global_outputs_folder = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\outputs\\"


global_input_cols_to_include_list = ["<CLOSE>", "<HIGH>"]
global_index_cols_list = ["<DATE>","<TIME>"]
global_index_col_str = "datetime"
global_target_file_folder_path = ""
global_feature_qty = 6
global_outputs_folder_path = ".\\outputs\\"
global_financial_history_folder_path = "FG action, do I need to update this?"
global_df_stocks_list_file           = pd.read_csv(os.path.join(global_general_folder, r"data\support_data\stock_info.csv"))
global_start_time = datetime.now()
global_error_str_1 = "the input {} is wrong for the input training_or_testing"
global_random_state = 1
global_scores_database = os.path.join(global_general_folder, r"outputs\scores_database.csv")
global_strptime_str = '%d/%m/%y %H:%M:%S'
global_strptime_str_2 = '%d/%m/%y %H:%M'
global_strptime_str_filename = '%d_%m_%y %H:%M:%S'
global_precalculated_assets_locations_dict = {
    "root" : os.path.join(global_general_folder, "precalculated_assets\\"),
    "topic_models"              : "topic_models\\",
    "annotated_tweets"          : "annotated_tweets\\",
    "predictive_model"          : "predictive_model\\",
    "sentiment_data"          : "sentiment_data\\",
    "technical_indicators"      : "technical_indicators\\",
    "experiment_records"        : "experiment_records\\",
    }
global_tweets = os.path.join(global_general_folder, r"data\twitter_data\apple.csv")
global_cleaned_tweets = os.path.join(global_general_folder, r"data\twitter_data\apple_cleaned_for_topic_discovery.csv")
global_combined_stopwords_list_path = os.path.join(global_general_folder, r"data\support_data\combined_stop_words.txt")
global_designs_record_final_columns_list = ["experiment_timestamp", "training_r2", "training_mse", "training_mae", "validation_r2", "validation_mse", "validation_mae", "testing_r2", "testing_mse", "testing_mae", "profitability", "predictor_names"]


SECS_IN_A_DAY = 60*60*24
SECS_IN_AN_HOUR = 60*60
FIVE_MIN_TIME_STEPS_IN_A_DAY = SECS_IN_A_DAY / (5*60)
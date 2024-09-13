
SECS_IN_AN_HOUR = 60*60


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



DoE_dict = {
    "senti_inputs_params_dict" : {
        "topic_qty" : [5,7,9],
        "topic_model_alpha" : [0.5, 1.0],
        "weighted_topics"       : [True, False],
        "relative_halflife" : [SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR]
    },
    "model_hyper_params" : {
        #"estimator__hidden_layer_sizes" : ["100_10", "50_20_10", "20_10"]
        "estimator__hidden_layer_sizes" : ["100_10", "15_10", "20_10"]
        #"estimator__hidden_layer_sizes" : [10, 10, 10]
    }
}
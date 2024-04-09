import data_prep_and_model_training as FG_model_training
import additional_reporting_and_model_trading_runs as FG_additional_reporting
from datetime import datetime, date, time
import pickle
import os
from config import global_strptime_str, global_strptime_str_filename, global_exclusively_str, global_precalculated_assets_locations_dict, global_financial_history_folder_path, global_error_str_1, global_financial_history_folder_path, global_df_stocks_list_file, global_random_state, global_strptime_str_2, global_index_cols_list, global_input_cols_to_include_list, global_start_time, global_financial_history_folder_path

list_of_model_subfolder_names = ["dfb17d5b073d46354f34c02da10684d793cc50b63e88d66a49b556a63a1688a2", "d8e5d23702377334027cc07da486827f7782c696543f586ac961acc9f3bb7136", "964981d6af03bc93f33ebd47ecac7b2783d0b41a2a0fa9afa4d4b3b1bd28c1c6", "57421bcb640295ee7090731f98cb541426ab0ed8c7f67172856c65832b0eaed5", "0629ef5162363d1bac9a316d846d250f780e5e20eefadffaf2dff92c4535d742", "13d56f6d8d8b5e2707a1ec817a441b1a330c8780b28b2972c2ae146105ee9403", "1ed5697ebc3809a5aac3bda996fb74cfd6481a8f621bb285fb3c52c13a9d56f6"]
folder_location = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\precalculated_assets\\predictive_model\\"
count = 0

def update_tracker_notepad_file(folder_path, work_completed=False):
    #this function checks of the folder has a tracker notepad file inside and edits the files and reports back accordingly
    work_started_substring = "work_started_"
    work_completed_substring = "work_completed_"


    # if there is a tracker file there, just report back there is a tracker file
    if work_completed==False:
        files = os.listdir(folder_path)
        for file in files:
            if file.startswith(work_started_substring) or file.startswith(work_completed_substring):
                return True    
        with open(os.path.join(folder_path, f'{work_started_substring}{datetime.now().strftime("%Y%m%d %H%M%S")}'), 'w') as f:
            f.write("BLANK FILE")
        return False
    elif work_completed==True:
        files = os.listdir(folder_path)
        for file in files:
            if file.startswith(work_started_substring):
                os.remove(file)
                with open(os.path.join(folder_path, f'{work_completed_substring}{datetime.now().strftime("%Y%m%d %H%M%S")}'), 'w') as f:
                    f.write("BLANK FILE")
                return True
                




for folder_name in list_of_model_subfolder_names:
    print(folder_name)
    work_started = update_tracker_notepad_file(os.path.join(folder_location, folder_name))
    if work_started == False:
        with open(os.path.join(folder_location, folder_name, "input_dict.pkl"), 'rb') as file:
            input_dict = pickle.load(file)
        with open(os.path.join(folder_location, folder_name, "additional_assets.pkl"), 'rb') as file:
            additional_assets = pickle.load(file)
        print("hello")

        temporal_params_dict     = input_dict["temporal_params_dict"]
        fin_inputs_params_dict   = input_dict["fin_inputs_params_dict"]
        senti_inputs_params_dict = input_dict["senti_inputs_params_dict"]
        outputs_params_dict      = input_dict["outputs_params_dict"]
        model_hyper_params       = input_dict["model_hyper_params"]
        reporting_dict           = input_dict["reporting_dict"]

        df_financial_data = FG_model_training.import_financial_data(
            #target_file_path          = fin_inputs_params_dict["historical_file"], 
            target_file_path          = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\data\\financial_data\\firstratedata\\AAPL_full_5min_adjsplit.txt",
            input_cols_to_include_list  = fin_inputs_params_dict["cols_list"],
            temporal_params_dict = temporal_params_dict, training_or_testing="training")
        print(datetime.now().strftime("%H:%M:%S") + " - populate_technical_indicators")
        df_financial_data = FG_model_training.retrieve_or_generate_then_populate_technical_indicators(df_financial_data, fin_inputs_params_dict["fin_indi"], fin_inputs_params_dict["fin_match"]["Doji"], fin_inputs_params_dict["historical_file"], fin_inputs_params_dict["financial_value_scaling"])

        if senti_inputs_params_dict["topic_qty"] >= 1:
            #sentiment data prep
            print(datetime.now().strftime("%H:%M:%S") + " - importing or prepping sentiment data")
            senti_inputs_params_dict["tweet_file_location"] = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\data\\twitter_data\\apple.csv"
            df_sentiment_data = FG_model_training.retrieve_or_generate_sentiment_data(df_financial_data.index, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, training_or_testing="training")
        elif senti_inputs_params_dict["topic_qty"] == 0:
            df_sentiment_data = None

        #model training - create regressors
        X_train, y_train   = FG_model_training.create_step_responces(df_financial_data, df_sentiment_data, pred_output_and_tickers_combos_list = outputs_params_dict["output_symbol_indicators_tuple"], pred_steps_ahead=outputs_params_dict["pred_steps_ahead"], financial_value_scaling=fin_inputs_params_dict["financial_value_scaling"])
        # reload old completed or semi-completed model if it already exists
        predictor_folder_location_string = global_precalculated_assets_locations_dict["root"] + global_precalculated_assets_locations_dict["predictive_model"]
        predictor_name = FG_model_training.return_predictor_name(FG_model_training.return_input_dict(temporal_params_dict = temporal_params_dict, fin_inputs_params_dict = fin_inputs_params_dict, senti_inputs_params_dict = senti_inputs_params_dict, outputs_params_dict = outputs_params_dict, model_hyper_params = model_hyper_params, reporting_dict = reporting_dict))
        predictor_location_folder_path = predictor_folder_location_string + FG_model_training.custom_hash(predictor_name) + "//"
        if os.path.exists(predictor_location_folder_path):
            model = FG_model_training.retrieve_model(predictor_location_folder_path, temporal_params_dict, fin_inputs_params_dict, senti_inputs_params_dict, outputs_params_dict, model_hyper_params, reporting_dict)
        else:
            raise ValueError("estimator not found" + str(count) + " " + str(predictor_name) + " " + str(FG_model_training.custom_hash(predictor_name)))
        model = FG_model_training.pop_scaler_if_required(model, X_train)
        a = list(model.additional_validation_dict["results_x_mins_weighted"].keys())[0]
        print(model.additional_validation_dict["results_x_mins_weighted"][a][0.9])
        model, training_scores_dict, validation_scores_dict, additional_validation_dict = model.regenerate_validation_scores(X_train, y_train)
        print(model.additional_validation_dict["results_x_mins_weighted"][a][0.9])
        update_tracker_notepad_file(os.path.join(folder_location, folder_name), work_completed=True)
        print(datetime.now().strftime("%H:%M:%S") + " - " + str(count) + " - XXXXXXXXXXXXXXXXXXXX")

#%% Import Methods
import data_prep_and_model_training as FG_model_training

import numpy as np
import pandas as pd
import GPyOpt
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
    os.environ["PyTHONWARNINGS"] = "ignore"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
global_master_folder_path = "placeholder"


#%% methods

def return_realign_plus_minus_table(preds, y_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True,global_master_folder_path=global_master_folder_path):
    input_col_str = "{}_{}"
    output_col_str = "{}_{}_{}"
    reaglined_plus_minus_dict = dict()
    for ticker, output in pred_output_and_tickers_combos_list:
        #initial output
        input_col_str_curr = input_col_str.format(ticker, output)
        df_temp = pd.DataFrame(y_test[input_col_str_curr])
        #pop the preditions
        output_col_str_2 = output_col_str.format(ticker, output, {})
    
        for step_backs in pred_steps_list:
            df_temp[output_col_str_2.format(step_backs)+"_before"] = preds[output_col_str_2.format(step_backs)].shift(step_backs)
            if make_relative == True:
                df_temp[output_col_str_2.format(step_backs)+"_before"] -= df_temp[input_col_str_curr]
        
        reaglined_plus_minus_dict[(ticker, output)] = df_temp
        df_temp.to_csv(pd.read_csv(os.path.join(global_master_folder_path,r"temp.csv")))
        preds.to_csv(pd.read_csv(os.path.join(global_master_folder_path, +r"\preds.csv")))
    
    return reaglined_plus_minus_dict

def return_results_X_min_plus_minus_accuracy(y_preds_input, y_test_input, pred_steps_list, confidences_before_betting_PC=[0, 0.01], financial_value_scaling=None, seconds_per_time_steps=7):
    
    # variables and checks
    if financial_value_scaling==None:
        raise ValueError("financial_value_scaling must be set")
    
    
    results_dict = {
        "results_bets_with_confidence_proportion" : dict(), 
        "results_x_mins_PC" : dict(), 
        "results_x_mins_score" : dict(), 
        "results_x_mins_weighted" : dict()
        }
    
    if type(pred_steps_list) == int:
        pred_steps_list = [pred_steps_list]
    
    
    # logic
    for steps_back in pred_steps_list:
        #initialise variables
        results_dict["results_bets_with_confidence_proportion"][steps_back] = dict()
        results_dict["results_x_mins_PC"][steps_back]                       = dict()
        results_dict["results_x_mins_score"][steps_back]                    = dict()
        results_dict["results_x_mins_weighted"][steps_back]                 = dict()
        
        y_preds, y_test = copy.deepcopy(y_preds_input), copy.deepcopy(y_test_input)
        y_original_val  = copy.deepcopy(y_test_input)
        # this time delta is reversed as now the test(actual), pred(predicted) & original value for a give prediction all share the same index value
        y_original_val.index += timedelta(seconds=steps_back*seconds_per_time_steps)

        # trim dfs so they all share the same index
        if isinstance(y_preds, pd.Series):
            y_preds_inputs = copy.deepcopy(y_preds)
            y_preds = pd.DataFrame(y_preds)
        merged_df = pd.merge(y_test, y_preds, left_index=True, right_index=True, how='inner')
        merged_df = pd.merge(merged_df, y_original_val, left_index=True, right_index=True, how='inner')
        y_test = y_test.loc[merged_df.index]
        y_preds = y_preds.loc[merged_df.index]
        y_original_val = y_original_val.loc[merged_df.index]
        
        #adjust if the data input is not in the "price delta form"
        if financial_value_scaling != "delta_scaled":
            df_x_values_delta, df_y_values_delta = pd.DataFrame(columns=["delta_price"]), pd.DataFrame(columns=["delta_price"])
            test_col_temp, pred_col_temp = y_test.columns[0], y_preds.columns[0]
            for prediction_index in merged_df.index[steps_back:]:
                df_x_values_delta.loc[prediction_index, "delta_price"] = y_test.loc[prediction_index,test_col_temp]  - y_original_val.loc[prediction_index,test_col_temp]
                df_y_values_delta.loc[prediction_index, "delta_price"] = y_preds.loc[prediction_index,pred_col_temp] - y_original_val.loc[prediction_index,test_col_temp]
        else: # no convertion needed
            df_x_values_delta = y_test
            df_y_values_delta = y_preds
                
        for confidence_threshold_key in confidences_before_betting_PC:
            
            #refresh values
            x = copy.deepcopy(df_x_values_delta)
            y = copy.deepcopy(df_y_values_delta)
            x.columns, y.columns = ["delta_price"], ["delta_price"]
            # confidence threshold is now taken from the percentile
            confidence_threshold_adjusted = np.percentile(abs(y),confidence_threshold_key * 100)

            ## proportion of bets taken
            # filter out bets not confident to make
            y["delta_price"] = y["delta_price"].apply(lambda y: y if abs(y) > confidence_threshold_adjusted else 0)
            results_dict["results_bets_with_confidence_proportion"][steps_back][confidence_threshold_key] = max(0,(abs(y["delta_price"]) > 0).sum() / len(y["delta_price"]))

            if results_dict["results_bets_with_confidence_proportion"][steps_back][confidence_threshold_key] != 0:
                ## proportion up/down correctly bet
                z = x * y
                results_dict["results_x_mins_PC"][steps_back][confidence_threshold_key] = (z["delta_price"] > 0).sum() / (abs(z["delta_price"]) > 0).sum()

                ## first scoring method
                # give one weighting to all accepted bets
                stake_a = pd.DataFrame()
                stake_a["delta_price"] = y["delta_price"].apply(lambda y: y/abs(y) if abs(y) > confidence_threshold_adjusted else 0)
                score_correct = (stake_a*x).sum()
                score_both = abs(stake_a*x).sum()
                results_dict["results_x_mins_score"][steps_back][confidence_threshold_key] = score_correct.values[0] / score_both.values[0]

                ## second scoring method FG_action: replace
                # calc stake in all accepted bets
                stake_b = pd.DataFrame()
                stake_b["delta_price"] = y["delta_price"].apply(lambda y: y - 0.5 * confidence_threshold_adjusted if y > confidence_threshold_adjusted else (y + 0.5 * confidence_threshold_adjusted if -y > confidence_threshold_adjusted else 0))
                score_correct = (stake_b*x).sum()
                score_both = abs(stake_b*x).sum()
                results_dict["results_x_mins_weighted"][steps_back][confidence_threshold_key] = score_correct.values[0] / score_both.values[0]
            else:
                results_dict["results_x_mins_PC"][steps_back][confidence_threshold_key]         = 0.0
                results_dict["results_x_mins_score"][steps_back][confidence_threshold_key]      = 0.0
                results_dict["results_x_mins_weighted"][steps_back][confidence_threshold_key]   = 0.0
    
    return results_dict

def return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = "", model_type_name="", model_start_time = "", outputs_folder_path = ".//outputs//tables//", timestamp = False):
    
    outputs_folder_path = outputs_folder_path + "//" + model_type_name + "//"
    single_levelled_tables = ["results_x_min_plus_minus_PC"]
    double_levelled_tables = ["results_x_min_plus_minus_PC_confindence", "results_x_min_plus_minus_score_confidence", "results_x_min_plus_minus_score_confidence_weighted"]
    fig_i = 0
    
    for table_name in single_levelled_tables:
        fig = plt.figure(fig_i)
        figure = plt.figure(fig_i)
        target_dict = results_tables_dict[table_name]
        table = pd.DataFrame.from_dict(target_dict, orient="columns")
        fig = sns.heatmap(table, cmap="vlag_r", annot=True, vmin=0, vmax=1)
        fig.set_title(table_name)
        fig.get_figure()
        figure.tight_layout()
        figure.savefig(outputs_folder_path+model_type_name+"_"+table_name+".png")
        table.to_csv(outputs_folder_path+model_type_name+"_"+table_name+".csv")
        fig_i += 1
           
    #prep info for double layered lists
    tickers_list    = list(results_tables_dict  [double_levelled_tables[0]].keys())
    steps_list      = list(results_tables_dict  [double_levelled_tables[0]][tickers_list[0]].keys())
    confidences_list = list(results_tables_dict [double_levelled_tables[0]][tickers_list[0]][steps_list[0]].keys())
    tickers_confidences_combos = []
    for con in confidences_list:
        for tic in tickers_list:
            tickers_confidences_combos = tickers_confidences_combos + [str(tic) + "_" + str(con)]
    
    #print double layered list
    for table_name in double_levelled_tables:
        #del fig, figure, new_table, target_dict, new_dict
        new_dict = dict()
        target_dict = results_tables_dict[table_name]
        new_table = pd.DataFrame(columns=tickers_confidences_combos)
        for step in steps_list:
            new_dict[step] = dict()
            for con in confidences_list:
                for tic in tickers_list:
                    new_dict[step][(tic, con)] = target_dict[tic][step][con]
                    #new_table.iloc[str(tic) + "_" + str(con), step] = target_dict[tic][step][con]
                    #new_table.loc[step, str(tic) + "_" + str(con)] = target_dict[tic][step][con]
                    tt = pd.DataFrame.from_dict(new_dict, orient="columns")
                    
        fig = plt.figure(fig_i)
        figure = plt.figure(fig_i)
        new_table = pd.DataFrame.from_dict(new_dict, orient="columns")
        if "PC" in table_name:
            fig = sns.heatmap(tt, cmap="vlag_r", annot=True, vmin=0, vmax=1)
        else:
            fig = sns.heatmap(tt, cmap="vlag_r", annot=True, vmin=-4, vmax=4)
        fig.set_title(table_name)
        figure = fig.get_figure()
        figure.tight_layout()
        figure.savefig(outputs_folder_path+model_type_name+"_"+table_name+".png")
        new_table.to_csv(outputs_folder_path+model_type_name+"_"+table_name+".csv")
        fig_i += 1
        del fig, figure, new_table, target_dict, new_dict
        
    results_x_min_plus_minus_PC                         = results_tables_dict["results_x_min_plus_minus_PC"]
    results_x_min_plus_minus_PC_confindence             = results_tables_dict["results_x_min_plus_minus_PC_confindence"]
    results_x_min_plus_minus_score_confidence           = results_tables_dict["results_x_min_plus_minus_score_confidence"]
    results_x_min_plus_minus_score_confidence_weighted  = results_tables_dict["results_x_min_plus_minus_score_confidence_weighted"]

    return plt, df_realigned_dict


#%% main line


def run_additional_reporting(preds=None,
                            y_testing = None, 
                            pred_steps_list = None,
                            confidences_before_betting_PC=None,
                            financial_value_scaling=None,
                            seconds_per_time_steps=7
                            ):
    
    #df_realigned_dict                   = return_realign_plus_minus_table(preds, y_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True)
    results_tables_dict                 = return_results_X_min_plus_minus_accuracy(preds, y_testing, pred_steps_list, confidences_before_betting_PC=confidences_before_betting_PC, financial_value_scaling=financial_value_scaling, seconds_per_time_steps=seconds_per_time_steps)
    #plt, df_realigned_dict              = return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = DoE_orders_dict["name"], model_type_name=model_type_name, model_start_time = model_start_time, outputs_folder_path = outputs_path, timestamp = False)
    
    return results_tables_dict#, plt, df_realigned_dict





















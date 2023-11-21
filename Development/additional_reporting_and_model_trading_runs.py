
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
import matplotlib.pyplot as plt



#%% methods

def return_realign_plus_minus_table(preds, Y_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True):
    input_col_str = "{}_{}"
    output_col_str = "{}_{}_{}"
    reaglined_plus_minus_dict = dict()
    for ticker, output in pred_output_and_tickers_combos_list:
        #initial output
        input_col_str_curr = input_col_str.format(ticker, output)
        df_temp = pd.DataFrame(Y_test[input_col_str_curr])
        #pop the preditions
        output_col_str_2 = output_col_str.format(ticker, output, {})
    
        for step_backs in pred_steps_list:
            df_temp[output_col_str_2.format(step_backs)+"_before"] = preds[output_col_str_2.format(step_backs)].shift(step_backs)
            if make_relative == True:
                df_temp[output_col_str_2.format(step_backs)+"_before"] -= df_temp[input_col_str_curr]
        
        reaglined_plus_minus_dict[(ticker, output)] = df_temp
        df_temp.to_csv("C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\" +"temp" + ".csv")
        preds.to_csv("C:\\Users\\Fabio\\OneDrive\\Documents\\Studies\\Final Project\\" +"preds" + ".csv")
    
    return reaglined_plus_minus_dict

def return_results_X_min_plus_minus_accuracy(y_preds, Y_test, pred_steps_list, confidences_before_betting_PC=[0, 0.01]):
        
    if len(Y_test.columns) > 1:
        raise ValueError("Y_test should only be one columns wide")
    
    df_temp = pd.DataFrame()
    pred_str = dict()
    ticker = dict()
    actual_values = dict()
    #df_realigned_temp = copy.deepcopy(df_realigned)
    if type(pred_steps_list) == int:
        pred_steps_list = [pred_steps_list]
    
    time_series_index = Y_test.index
    
    results_bets_with_confidence_proportion             = dict()
    results_x_min_plus_minus_PC                         = dict()
    results_x_min_plus_minus_PC_confindence             = dict()
    results_x_min_plus_minus_score_confidence           = dict()
    results_x_min_plus_minus_score_confidence_weighted  = dict()
    
    
    
    count_bets_with_confidence               = dict()
    count_correct_bets_with_confidence       = dict()
    count_correct_bets_with_confidence_score = dict()
    count_correct_bets_with_confidence_score_weight = dict()
    count_correct_bets_with_confidence_score_weight_total = dict()
    
    time_series_index = Y_test.index
    
    for steps_back in pred_steps_list:
        #initialise variables
        
        # count values
        count           = 0
        count_correct   = 0
        
        count_bets_with_confidence[steps_back]               = dict()
        count_correct_bets_with_confidence[steps_back]       = dict()
        count_correct_bets_with_confidence_score[steps_back] = dict()
        count_correct_bets_with_confidence_score_weight[steps_back] = dict()
        count_correct_bets_with_confidence_score_weight_total[steps_back] = dict()
        
        results_x_min_plus_minus_PC[steps_back]                      = dict()
        results_bets_with_confidence_proportion[steps_back]          = dict()
        results_x_min_plus_minus_PC_confindence[steps_back]          = dict()
        results_x_min_plus_minus_score_confidence[steps_back]        = dict()
        results_x_min_plus_minus_score_confidence_weighted[steps_back] = dict()
        
        
        for confidence_threshold in confidences_before_betting_PC:
            count_bets_with_confidence              [steps_back][confidence_threshold] = 0
            count_correct_bets_with_confidence      [steps_back][confidence_threshold] = 0
            count_correct_bets_with_confidence_score[steps_back][confidence_threshold] = 0
            count_correct_bets_with_confidence_score_weight[steps_back][confidence_threshold] = 0
            count_correct_bets_with_confidence_score_weight_total[steps_back][confidence_threshold] = 0

        
        # values
        max_pred_value = max(pred_steps_list)
        x_values = list(Y_test.iloc[:,0].values)
        if isinstance(y_preds, np.ndarray):
            y_values = list(y_preds.reshape((1,-1))[0])
        else:
            y_values = list(y_preds.values.reshape((1,-1))[0])
        
        for time_step in range(max(pred_steps_list), len(Y_test.index)):
            
            #basic values
            
            original_value      = x_values[time_step - steps_back]
            expected_difference = y_values[time_step] - x_values[time_step]
            actual_difference   = x_values[time_step] - x_values[time_step - steps_back]
            relative_confidence = expected_difference / original_value
            if not actual_difference * expected_difference == 0:
                count += 1
            
            
            #basic count scoring 
            if abs(actual_difference * expected_difference) > 0:
                count_correct += 1
            
            #bets with confidence scoring
            for confidence_threshold in confidences_before_betting_PC:
                if   abs(relative_confidence) > confidence_threshold and actual_difference * expected_difference > 0:
                    count_bets_with_confidence              [steps_back][confidence_threshold] += 1
                    count_correct_bets_with_confidence      [steps_back][confidence_threshold] += 1
                    count_correct_bets_with_confidence_score[steps_back][confidence_threshold] += abs(actual_difference)
                    count_correct_bets_with_confidence_score_weight[steps_back][confidence_threshold] += abs(actual_difference) * (abs(relative_confidence) - confidence_threshold)
                    count_correct_bets_with_confidence_score_weight_total[steps_back][confidence_threshold] += (abs(relative_confidence) - confidence_threshold)
                    
                elif abs(relative_confidence) > confidence_threshold and actual_difference * expected_difference < 0:
                    count_bets_with_confidence              [steps_back][confidence_threshold] += 1
                    count_correct_bets_with_confidence      [steps_back][confidence_threshold] += 0
                    count_correct_bets_with_confidence_score[steps_back][confidence_threshold] -= abs(actual_difference)
                    count_correct_bets_with_confidence_score_weight[steps_back][confidence_threshold] -= abs(actual_difference) * (abs(relative_confidence) - confidence_threshold)
                    count_correct_bets_with_confidence_score_weight_total[steps_back][confidence_threshold] += (abs(relative_confidence) - confidence_threshold)
                
                
                    

        #Total scores for ticker steps_back conbination
        results_x_min_plus_minus_PC[steps_back] = count_correct / count
        for confidence_threshold in confidences_before_betting_PC:
            results_bets_with_confidence_proportion           [steps_back][confidence_threshold] = count_bets_with_confidence[steps_back][confidence_threshold] / count
            if count_bets_with_confidence                           [steps_back][confidence_threshold] > 0:
                results_x_min_plus_minus_PC_confindence           [steps_back][confidence_threshold] = count_correct_bets_with_confidence             [steps_back][confidence_threshold] / count_bets_with_confidence                           [steps_back][confidence_threshold]
                results_x_min_plus_minus_score_confidence         [steps_back][confidence_threshold] = count_correct_bets_with_confidence_score       [steps_back][confidence_threshold] / count_bets_with_confidence                           [steps_back][confidence_threshold]
                results_x_min_plus_minus_score_confidence_weighted[steps_back][confidence_threshold] = count_correct_bets_with_confidence_score_weight[steps_back][confidence_threshold] / count_correct_bets_with_confidence_score_weight_total[steps_back][confidence_threshold]
            else:
                results_x_min_plus_minus_PC_confindence           [steps_back][confidence_threshold] = 0
                results_x_min_plus_minus_score_confidence         [steps_back][confidence_threshold] = 0
                results_x_min_plus_minus_score_confidence_weighted[steps_back][confidence_threshold] = 0
                
                
    
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results_dict = dict()
    results_dict["results_bets_with_confidence_proportion"] = results_bets_with_confidence_proportion
    results_dict["results_x_mins_PC"]        = results_x_min_plus_minus_PC_confindence
    results_dict["results_x_mins_score"]     = results_x_min_plus_minus_score_confidence 
    results_dict["results_x_mins_weighted"]  = results_x_min_plus_minus_score_confidence_weighted
    
    return results_dict

def return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = "", model_type_name="", model_start_time = "", outputs_folder_path = ".//outputs//tables//", timestamp = False):
    
    outputs_folder_path = outputs_folder_path + "\\" + model_type_name + "\\"
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
                            Y_test = None, 
                            pred_steps_list = None, 
                            DoE_orders_dict = None, 
                            model_type_name = None, 
                            outputs_path = None, 
                            model_start_time = None,
                            confidences_before_betting_PC=None
                            ):
    
    #df_realigned_dict                   = return_realign_plus_minus_table(preds, Y_test, pred_steps_list, pred_output_and_tickers_combos_list, make_relative=True)
    results_tables_dict                 = return_results_X_min_plus_minus_accuracy(preds, Y_test, pred_steps_list, confidences_before_betting_PC=confidences_before_betting_PC)
    #plt, df_realigned_dict              = return_model_performance_tables_figs(df_realigned_dict, preds, pred_steps_list, results_tables_dict, DoE_name = DoE_orders_dict["name"], model_type_name=model_type_name, model_start_time = model_start_time, outputs_folder_path = outputs_path, timestamp = False)
    
    return results_tables_dict#, plt, df_realigned_dict





















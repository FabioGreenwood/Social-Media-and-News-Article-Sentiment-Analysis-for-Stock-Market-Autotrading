import numpy as np
import pandas as pd
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import openpyxl

#%% parameters
# parameters

confidences_list    = [0, 0.01, 0.02, 0.035, 0.05, 0.1]
pred_steps_list     = [1, 3, 5, 15]
model_type_designator_list = {"Full" : "multi_topic", "No Topics" : "no_topics", "No Sentiment" : "no_sentiment"}
df_columns = ["Time Steps (Mins)", "Confidence"] + list(model_type_designator_list.keys()) + ["Bet Up Every Time", "Bet Down Every Time"]
target_columns_string_dict = {"bets_proportion" : "bets_with_confidence_proportion_sX_c{}", "precision" : "x_mins_PC_sX_c{}", "score" : "x_mins_score_sX_c{}", "weighted_score" : "x_mins_weighted_sX_c{}"}
up_down_outputs_cols_to_records_cols_dict = {"Bet Up Every Time" : "up", "Bet Down Every Time" : "down"}

# importing of data

results_csv_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\outputs\kfold_v3_global_results.csv"
df_results = pd.read_csv(results_csv_file_path)
always_up_down_csv_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\always_up_down_results\always_up_down.csv"
df_always_up_down_results = pd.read_csv(always_up_down_csv_file_path)
df_always_up_down_results.set_index("bet_direction", inplace=True)

# exporting data

outputs_folder_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\human readable results"
overall_results_file = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\human readable results\Final Project Excel Results.xlsx"

#%% scraps of code

"""cols_strings_ori = results_csv.columns
pattern = re.compile(r's\d+_')

cols_strings_new = []
for i, input_string in enumerate(cols_strings_ori):
    matches = pattern.findall(input_string)
    replaced_string = pattern.sub('sX_', input_string)
    if not replaced_string in cols_strings_new:
        cols_strings_new = cols_strings_new + [replaced_string]"""
    

#%% methods

def to_scientific_notation(number, sf):
    number = float(number)
    format_string = "{{:.{}e}}".format(sf)
    scientific_notation = format_string.format(number)
    return scientific_notation

def return_pareto_results_and_ID(df_results, target_output_col, model_type, pred_steps, output_tables_dict_single, target_columns_string_dict=target_columns_string_dict):
    df_results = df_results[df_results["run_name"].str.contains(model_type_designator_list[model_type], case=False)]
    df_results = df_results[df_results["pred_steps"] == pred_steps]
    if action_min in output_tables_dict_single["special_actions"]:
        opt_id      = df_results[target_output_col].idxmin()
    else:
        opt_id      = df_results[target_output_col].idxmax()
    opt_value   = df_results.at[opt_id, target_output_col]
        
    return opt_id, opt_value 

def return_col_for_always_up_down_dict(input_str, pred_steps):
    
    input_str = input_str[:input_str.rfind("c")+1] + "0"
    input_str = input_str[:input_str.rfind("sX")+1] + str(int(pred_steps)) + input_str[input_str.rfind("sX")+2:]
    
    return input_str

def return_target_column(output_tables_dict_single, con):
    if "mae" in output_tables_dict_single["col_str"]:
        output = output_tables_dict_single["group"] + "_" + output_tables_dict_single["col_str"]
    else:
        output = str(output_tables_dict_single["group"] + "_" + output_tables_dict_single["col_str"]).format(con)
        
    return output

def return_column_string_2(input_string, con):
    if "{}" in input_string:
        output = input_string.format(con)
    else:
        output = input_string
    return output
        
    
    

def return_pareto_table_and_max_ids(output_tables_dict_single, max_ids_and_vals_dict, sf=1):
    global action_no_pre_ID, action_pop_ID, action_n_updn
    
    # the function of the input bools can be found at the large if-else gate
    df_output = pd.DataFrame(columns=df_columns)
    
    for pred_steps in pred_steps_list:
        for con in confidences_list:
            id = len(df_output.index)
            df_output.loc[id, ["Time Steps (Mins)", "Confidence"]] = [pred_steps, con]
            for model_type in model_type_designator_list:
                # this if gate controls what result is populated. See each if statement for more info FG_ACTION_UPDATE NPTE
                if action_no_pre_ID in output_tables_dict_single["special_actions"] and action_no_val in output_tables_dict_single["special_actions"]:
                    raise ValueError("can't have both no_pre_ID and no_val in special actions")
                target_col = return_target_column(output_tables_dict_single, con)
                if action_no_pre_ID in output_tables_dict_single["special_actions"]:
                    val_id, val = return_pareto_results_and_ID(df_results, target_col, model_type, pred_steps, output_tables_dict_single, output_tables_dict_single)
                    df_output.loc[id, model_type] = val
                    # update new max_ids_and_vals_dict value
                    max_ids_and_vals_dict[pred_steps, con, model_type] = (val_id, val)
                elif not action_no_pre_ID in output_tables_dict_single["special_actions"] and not action_no_val in output_tables_dict_single["special_actions"]:
                    val = df_results.loc[max_ids_and_vals_dict[pred_steps, con, model_type][0], return_column_string_2(target_col, con)]
                    df_output.loc[id, model_type] = val
                elif action_no_val in output_tables_dict_single["special_actions"]:
                    df_output.loc[id, model_type] = max_ids_and_vals_dict[pred_steps, con, model_type][0]
                
                # populate always up/down results
                if not action_no_up_down in output_tables_dict_single["special_actions"]:
                    #records_col = target_columns_string_dict["target_output_name"].format(pred_steps, 0)
                    target_updown_col =  return_col_for_always_up_down_dict(target_col, pred_steps)
                    for output_col in up_down_outputs_cols_to_records_cols_dict:
                        records_row = up_down_outputs_cols_to_records_cols_dict[output_col]
                        val = df_always_up_down_results.loc[records_row, target_updown_col[target_updown_col.find("_")+1:]]
                        df_output.loc[id, output_col] = val
                
    df_output = df_output.apply(pd.to_numeric, errors='coerce')

    
    return df_output, max_ids_and_vals_dict
    

def update_excel(df, start_cell, excel_file_path):
    workbook = openpyxl.load_workbook(excel_file_path)
    sheet = workbook.active
    for row_num, row in enumerate(df.values):
        for col_num, value in enumerate(row):
            cell = sheet.cell(row=row_num + sheet[start_cell].row, column=col_num + sheet[start_cell].column)
            cell.value = value
    workbook.save(excel_file_path)

#%% main line

action_no_pre_ID    = "dont_populate_according_to_previous_max_id"
action_no_val       = "populate_ID_value_not_val"
action_min          = "minimise_value_not_maximise"
action_no_up_down   = "dont_populate_up_down_values"

weighted_score_col_str  = "x_mins_weighted_sX_c{}"
absolute_score_col_str  = "x_mins_score_sX_c{}"
bets_proportion_col_str = "bets_with_confidence_proportion_sX_c{}"
precision_col_str       = "x_mins_PC_sX_c{}"

output_tables_dict = {"Optimum Validation MAE Score" : 
        {"col_str":"mae",
        "group" : "validation", "top_left_cell" : "B4",
        "special_actions" : [action_no_pre_ID, action_min, action_no_up_down]},
    "Validation Weighted Score (for Highest Validation MAE Score Design)":
        {"col_str":weighted_score_col_str,
        "group" : "validation", "top_left_cell" : "J4",
        "special_actions" : []},
    "Validation Absolute Score (for Highest Validation MAE Score Design)":
        {"col_str":absolute_score_col_str,
        "group" : "validation", "top_left_cell" : "R4",
        "special_actions" : []},
    "Validation Proportion of Correct Positions (for Highest Validation MAE Score Design)":
        {"col_str":precision_col_str,
        "group" : "validation", "top_left_cell" : "Z4",
        "special_actions" : []},
    "Validation Proportion of Potential Positions Taken (for Highest Validation MAE Score Design)":
        {"col_str":bets_proportion_col_str,
        "group" : "validation", "top_left_cell" : "AH4",
        "special_actions" : []},
    "Global Design IDs (for Highest Validation MAE Score Design)":
        {"col_str":"mae",
        "group" : "testing", "top_left_cell" : "AP4",
        "special_actions" : [action_no_val, action_no_up_down]},
    "Testing MAE Score (for Highest Validation MAE Score Design)":
        {"col_str":"mae",
        "group" : "testing", "top_left_cell" : "B31",
        "special_actions" : [action_no_up_down]},
    "Testing Weighted Score (for Highest Validation MAE Score Design)":
        {"col_str":weighted_score_col_str,
        "group" : "testing", "top_left_cell" : "J31",
        "special_actions" : []},
    "Testing Absolute Score (for Highest Validation MAE Score Design)":
        {"col_str":absolute_score_col_str,
        "group" : "testing", "top_left_cell" : "R31",
        "special_actions" : []},
    "Testing Proportion of Correct Positions (for Highest Validation MAE Score Design)":
        {"col_str":precision_col_str,
        "group" : "testing", "top_left_cell" : "Z31",
        "special_actions" : []},
    "Testing Proportion of Potential Positions Taken (for Highest Validation MAE Score Design)":
        {"col_str":bets_proportion_col_str,
        "group" : "testing", "top_left_cell" : "H31",
        "special_actions" : []}}
#%%



max_ids_and_vals_dict       = dict()
final_output_tables_list    = dict()

for name in output_tables_dict:
    df, max_ids_and_vals_dict = return_pareto_table_and_max_ids(output_tables_dict[name], max_ids_and_vals_dict, output_tables_dict)
    
    df.to_csv(outputs_folder_file_path + "\\" + name + ".csv")
    #save_table(df, generate_image_name(target_output_name), folder, colour_bool)
    update_excel(df, output_tables_dict[name]["top_left_cell"], overall_results_file)
    
    final_output_tables_list[name] = df
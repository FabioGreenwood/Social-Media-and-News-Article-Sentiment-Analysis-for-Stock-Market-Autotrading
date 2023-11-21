import numpy as np
import pandas as pd
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product


#%% parameters
# parameters

confidences_list    = [0, 0.01, 0.02, 0.05, 0.1]
pred_steps_list     = [1, 3, 5, 15]
model_type_designator_list = {"Full" : "multi_topic", "No Topics" : "no_topics", "No Sentiment" : "no_sentiment"}
df_columns = ["Time Steps (Mins)", "Confidence"] + list(model_type_designator_list.keys()) + ["Bet Up Every Time", "Bet Down Every Time"]
target_columns_string_dict = {"bets_proportion" : "bets_with_confidence_proportion_s{}_c{}", "precision" : "x_mins_PC_s{}_c{}", "score" : "x_mins_score_s{}_c{}", "weighted_score" : "x_mins_weighted_s{}_c{}"}
up_down_outputs_cols_to_records_cols_dict = {"Bet Up Every Time" : "up", "Bet Down Every Time" : "down"}

# importing of data

results_csv_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\outputs\non_seededv2_global_results.csv"
df_results = pd.read_csv(results_csv_file_path)
always_up_down_csv_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\always_up_down_results\always_up_down.csv"
df_always_up_down_results = pd.read_csv(always_up_down_csv_file_path)
df_always_up_down_results.set_index("bet_direction", inplace=True)

# exporting data

outputs_folder_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\human readable results"


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

def return_pareto_results_and_ID(df_results, target_output_name, model_type, pred_steps, con, target_columns_string_dict=target_columns_string_dict):
    
    output_col = target_columns_string_dict[target_output_name].format(pred_steps, con)
    
    df_results = df_results[df_results["run_name"].str.contains(model_type_designator_list[model_type], case=False)]
    df_results = df_results[df_results["pred_steps"] == pred_steps]
    
    max_id      = df_results[output_col].idxmax()
    max_value   = df_results.at[max_id, output_col]
        
    return max_id, max_value 

def return_pareto_table_and_max_ids(target_output_name, previous_max_ids_and_vals_dict,
                                    replace_previous_max_ids_and_vals_dict_BOOL=False, 
                                    populate_value_corresponding_to_previous_max_id=True, 
                                    populate_only_the_id_in_max_id_dict=False, 
                                    sf=1):
    # the function of the input bools can be found at the large if-else gate
    max_ids_and_vals_dict = dict()
    df_output = pd.DataFrame(columns=df_columns)
   
    for pred_steps in pred_steps_list:
        for con in confidences_list:
            id = len(df_output.index)
            df_output.loc[id, ["Time Steps (Mins)", "Confidence"]] = [pred_steps, con]
            for model_type in model_type_designator_list:
                # this if gate controls what result is populated. See each if statement for more info FG_ACTION_UPDATE NPTE
                
                if populate_value_corresponding_to_previous_max_id == True and populate_only_the_id_in_max_id_dict == False: # this table is only required to report its value for the optimal results for the optimial design of another variable FG_ACTION_UPDATE NPTE
                    output_col = target_columns_string_dict[target_output_name].format(pred_steps, con)
                    val = df_results.loc[previous_max_ids_and_vals_dict[pred_steps, con, model_type][0], output_col]
                    df_output.loc[id, model_type] = val
                    
                elif populate_value_corresponding_to_previous_max_id == False and populate_only_the_id_in_max_id_dict == False: # if it only needs to report its own pareto value FG_ACTION_UPDATE NPTE
                    val_id, val = return_pareto_results_and_ID(df_results, target_output_name, model_type, pred_steps, con)
                    df_output.loc[id, model_type] = val
                    
                elif populate_only_the_id_in_max_id_dict == True:
                    df_output.loc[id, model_type] = previous_max_ids_and_vals_dict[pred_steps, con, model_type][0]
                
                # populate always up/down results
                records_col = target_columns_string_dict[target_output_name].format(pred_steps, 0)
                for output_col in up_down_outputs_cols_to_records_cols_dict:
                    records_row = up_down_outputs_cols_to_records_cols_dict[output_col]
                    val = df_always_up_down_results.loc[records_row, records_col]
                    df_output.loc[id, output_col] = val
                
                                    
                # update new max_ids_and_vals_dict value
                if replace_previous_max_ids_and_vals_dict_BOOL == True:
                    max_ids_and_vals_dict[pred_steps, con, model_type] = (val_id, val)
    
    if replace_previous_max_ids_and_vals_dict_BOOL == False:
        max_ids_and_vals_dict = previous_max_ids_and_vals_dict
    
    df_output = df_output.apply(pd.to_numeric, errors='coerce')

    
    return df_output, max_ids_and_vals_dict
    

def generate_and_save_table(df, outputs_folder_file_path, target_output_name):
    
    
    
    return

    

#%% main line


output_tables_list = [ # table name, output name, replace max id and vals, populate according to previous max id, populate_only_the_id_in_max_id_dict
    ["Optimum Weighted Score", "weighted_score", True, False, False], 
    ["Proportion of Correct Positions (for Optimum Design)", "precision", False, True, False], 
    ["Absolute Score (for Optimum Design)", "score", False, True, False], 
    ["Proportion of Potential Positions Taken (for Optimum Design)", "bets_proportion", False, True, False],
    ["Global Design IDs (for Optimum Design)", "weighted_score",  False, False, True], 
    
    ["Proportion of Correct Positions (Highest Recorded)", "precision", True, False, False],
    ["Weighted Score (of Designs with Highest Correct Positions)", "weighted_score", False, True, False],
    
    ["Absolute Score (Highest Recorded)", "score", True, False, False],
    ["Weighted Score (for Highest Absolute Score Design)", "weighted_score", False, True, False], 
    ["Proportion of Correct Positions (for Highest Absolute Score Design)", "precision", False, True, False], 
    ["Proportion of Potential Positions Taken (for Highest Absolute Score Design)", "bets_proportion", False, True, False],
    ["Global Design IDs (for Highest Absolute Score)", "score",  False, False, True]
    ]
    

max_ids_and_vals_dict       = dict()
final_output_tables_list    = dict()

for table_name, target_output_name, replace_BOOL, previous_BOOL, id_BOOL in output_tables_list:
    df, max_ids_and_vals_dict = return_pareto_table_and_max_ids(target_output_name, max_ids_and_vals_dict, 
                                                                replace_previous_max_ids_and_vals_dict_BOOL=replace_BOOL, 
                                                                populate_value_corresponding_to_previous_max_id=previous_BOOL,
                                                                populate_only_the_id_in_max_id_dict=id_BOOL)
    df.to_csv(outputs_folder_file_path + "\\" + table_name + ".csv")
    #save_table(df, generate_image_name(target_output_name), folder, colour_bool)
    
    final_output_tables_list[target_output_name] = df
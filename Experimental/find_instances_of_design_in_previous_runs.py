"""
The purpose of this file is to analyse how often a set of designs has been repeated. It will output its results as a notepad and a printout

It will count the times the file was an explict copy, a copy with all the matching variables

"""

# some_file.py
import sys
import pandas as pd
import os
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, 'C:/Users/Public/fabio_uni_work/Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading/Development/')
#from experiment_handler import return_keys_within_2_level_dict 

# parameters
SECS_IN_AN_HOUR, global_exclusively_str = 1,2
design_space_dict = {
    "senti_inputs_params_dict" : {
        "topic_qty" : [5, 9, 13, 17, 25],
        "topic_model_alpha" : [0.3, 0.7, 1, 2, 3, 5, 7, 13],
        "relative_halflife" : [3*60, 0.25 * SECS_IN_AN_HOUR, 2*SECS_IN_AN_HOUR, 7*SECS_IN_AN_HOUR], 
        "apply_IDF" : [False, True],
        "topic_weight_square_factor" : [1, 2, 4],
        "factor_tweet_attention" : [False, True],
        "factor_topic_volume" : {0 : False, 1 : True, 2 : global_exclusively_str}
    },
    "model_hyper_params" : {
        "estimator__hidden_layer_sizes" : {
            0 : [("GRU", 50)],
            1 : [("GRU", 40), ("GRU", 30)],
            2 : [("LSTM", 50)],
            3 : [("LSTM", 50), ("LSTM", 30)],
            4 : [("LSTM", 50), ("GRU", 30), ("GRU", 20)],
            5 : [("LSTM", 60), ("GRU", 30), ("LSTM", 8)]
            },
        "general_adjusting_square_factor" : [3, 2, 1, 0],
        "estimator__alpha"                : [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-4], 
        "lookbacks"                       : [8, 10, 15, 20, 25, 50],
        "early_stopping" : [0, 5, 7, 9, 12]
        },
        
    "string_key" : {}
}

list_of_design_inputs = [
    [25, 13,  7200,  0, 4, 0, 0, 5, 1, 0.0001, 10, 7],  # 0
    [13, 3,   7200,  0, 2, 0, 0, 4, 0, 1e-11,  8,  9],  # 1
    [13, 7,   180,   1, 1, 1, 0, 4, 3, 1e-09,  25, 0],  # 2
    [25, 13,  7200,  0, 1, 1, 1, 2, 1, 1e-06,  10, 5],  # 3
    [13, 2,   900,   1, 4, 0, 0, 0, 0, 1e-10,  8,  12], # 4
    [9,  13,  900,   0, 1, 1, 2, 5, 0, 1e-08,  8,  0],  # 5
    [17, 3,   25200, 1, 1, 1, 1, 4, 0, 0.0001, 15, 12], # 6
    [25, 7,   25200, 0, 4, 1, 2, 2, 3, 1e-09,  15, 9],  # 7
    [17, 7,   7200,  0, 4, 1, 2, 3, 1, 1e-05,  8,  0],  # 8
    [5,  5,   25200, 1, 4, 1, 0, 3, 3, 1e-10,  15, 7],  # 9
    [17, 0.7, 900,   1, 2, 1, 1, 2, 3, 1e-07,  50, 5],  # 10
    [9,  2,   7200,  1, 4, 1, 0, 4, 0, 1e-11,  15, 5],  # 11
    
    [17, 7,   900,   1, 2, 1, 1, 5, 2, 1e-06, 20, 5],   # 12
    
    [17, 13,  900,   0, 4, 0, 1, 4, 1, 1e-07,  50, 9],  # 13
    [5,  0.3, 25200, 1, 4, 1, 2, 5, 2, 1e-09,  8,  0],  # 14
    [5,  13,  900,   0, 1, 1, 1, 4, 0, 1e-05,  20, 0],  # 15
    [17, 2,   7200,  0, 2, 1, 2, 0, 2, 0.0001, 8,  7],  # 16
    [13, 13,  900,   1, 2, 0, 0, 3, 0, 1e-05,  50, 9],  # 17
    [13, 13,  900,   0, 1, 1, 1, 1, 0, 1e-05,  10, 5],  # 18
    [25, 0.7, 900,   0, 2, 1, 2, 5, 1, 1e-05,  25, 7],  # 19
    [5,  0.3, 900,   1, 4, 0, 0, 3, 0, 1e-08,  8,  9]   # 20
]


def compare_design_dict_to_df(design_params_list, df, design_space_dict=design_space_dict, mode=None):
    
    # compare that all variables are there
    design_columns = return_keys_within_2_level_dict(design_space_dict)
    design = pd.Series(data=design_params_list, index=design_columns)
    for col in design_columns:
        if not col in df.columns:
            design = design.drop(labels=col)
    
    ####### add a check that more that X variables are still in the running
    if len(design) > 5:
        # now the relevent variables are prepared
        if mode == "explict":
            correct_rows = (df[design.index] == design).all(axis=1)
        elif mode == "implict":
            matching_values = (df[design.index] == design).sum(axis=1)
            correct_rows    = (matching_values - len(df[design.index].columns) > -4)
        else:
            raise ValueError("mode setting not recognized")
        if len(df[correct_rows]) > 0:
            return 1
        else:
            return 0
    else:
        print("short design found")
        return 0


def read_csvs_in_folder(folder_path, file_suffix=".csv"):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(file_suffix) and not file.endswith("T" + file_suffix) and os.path.isfile(os.path.join(folder_path, file))]
    
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        dataframe_name = os.path.splitext(file)[0]  # Get the name of the CSV file without extension
        dataframes[dataframe_name] = pd.read_csv(file_path)
    return dataframes

def return_keys_within_2_level_dict(input_dict):
    output_list = []
    for key in input_dict:
        if type(input_dict[key]) == dict:
            for subkey in input_dict[key]:
                output_list = output_list + [key + "_" + subkey]
        else:
            output_list = output_list + [key]
    return output_list

# collect dfs

folder_path = 'C:/Users/Public/fabio_uni_work/Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading/precalculated_assets/experiment_records'
folder_path = 'C:/Users/Public/fabio_uni_work/Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading/outputs'
file_suffix = '.csv'
list_dataframes = read_csvs_in_folder(folder_path, file_suffix)

# compare and count designs
for design_ID in range(len(list_of_design_inputs)):
    count_explict = 0
    count_implicit = 0
    history = []
    for df in list_dataframes:
        count_explict  += compare_design_dict_to_df(list_of_design_inputs[design_ID], list_dataframes[df], mode = "explict")
        count_implicit += compare_design_dict_to_df(list_of_design_inputs[design_ID], list_dataframes[df], mode = "implict")
        if compare_design_dict_to_df(list_of_design_inputs[design_ID], list_dataframes[df], mode = "implict")>0:
            history += [df]
    print("ID:{} explict:{} implicit:{} history:{}".format(design_ID, count_explict, count_implicit, str(history)))
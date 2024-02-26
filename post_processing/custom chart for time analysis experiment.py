# importing package 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ttest_rel
import itertools
import os


## constants

global_strptime_str = '%d/%m/%y %H:%M:%S'

profit_colours = ["darkgreen", "forestgreen", "mediumseagreen", "springgreen"]

low_profit_colours

brown
firebrick





colors = ['darkgreen', "blue", "violet", 'darkgoldenrod', "brown", 'red', "brown", "black", "pink", "violet", 'darkgreen', 'darkturquoise', 'indianred', 'yellowgreen']
original_scores_list = ["high", "high", "high", "mid", "low", "low", "Unknown", "Unknown", "Unknown", "Unknown"]
results_csv_file_home_path  = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\human_readable_results\\240125_shortertraining_a"
save_folder_path            = "C:\\Users\\Public\\fabio_uni_work\\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\\human_readable_results\\temporal_analysis"
base_analysis_filename_format_str = "run6_{}.csv"
#temporal_analysis_filename_format_str = "time_shift_test_of_" + base_analysis_filename_format_str
temporal_analysis_filename_format_str = "time_shift_test_of_" + "run13_shortertraining{}.csv"
#temporal_analysis_filename_format_str = "time_shift_test_of_" + "run10_{}.csv"
shards_studied = [3, 9]

temporal_label_dict_chart_label_str, temporal_label_dict_column_name_str, values_lookup_name_str = "month period end date", "temporal_params_dict_test_period_end", "test_period_end_study_dict"
variable_name_for_chart_str, variable_df_col_name_str = "testing_x_mins_weighted_sX_c0.9", "testing_x_mins_weighted_sX_c0.9"


default_temporal_params_dict        = {
    "train_period_start"    : datetime.strptime('01/01/16 00:00:00', global_strptime_str),
    "train_period_end"      : datetime.strptime('01/07/19 00:00:00', global_strptime_str), 
    "time_step_seconds"     : 5*60, #5 mins,
    "test_period_start"     : datetime.strptime('01/07/19 00:00:00', global_strptime_str),
    "test_period_end"       : datetime.strptime('01/01/20 00:00:00', global_strptime_str)
}

## special additions for time shift study
def dateshift(datetime_index=0, multiples=0, length_of_single_shift_months=1):
    list_of_datetimes = [default_temporal_params_dict["train_period_end"], default_temporal_params_dict["test_period_start"], default_temporal_params_dict["test_period_end"]]
    return list_of_datetimes[datetime_index] + timedelta(days=30*length_of_single_shift_months*multiples)
train_period_end_study_dict   = {0 : dateshift(0, -6)}
#test_period_start_study_dict  = {0 : dateshift(1, -3), 1 : dateshift(1, -2), 2 : dateshift(1, -1), 3 : dateshift(1, -0)}
#test_period_end_study_dict    = {0 : dateshift(2, -3), 1 : dateshift(2, -2), 2 : dateshift(2, -1), 3 : dateshift(2, -0)}

test_period_start_study_dict  = {
0 : dateshift(1, -6), 1  : dateshift(1, -5), 2 : dateshift(1, -4),
3 : dateshift(1, -3), 4  : dateshift(1, -2), 5 : dateshift(1, -1),
6 : dateshift(1, +0), 7  : dateshift(1, +1), 8 : dateshift(1, +2),
9 : dateshift(1, +3), 10 : dateshift(1, +4), 11: dateshift(1, +5)
}
test_period_end_study_dict    = {
0 : dateshift(2, -11), 1  : dateshift(2, -10), 2 : dateshift(2, -9),
3 : dateshift(2, -8),  4  : dateshift(2, -7),  5 : dateshift(2, -6), 
6 : dateshift(2, -5),  7  : dateshift(2, -4),  8 : dateshift(2, -3), 
9 : dateshift(2, -2),  10 : dateshift(2, -1),  11: dateshift(2, -0)
}
ts_lookup = locals()[values_lookup_name_str]

def return_model_type_name(params):
    target_string = params["run_name"]
    if "multi" in target_string:
        return "Full"
    elif "no_topics" in target_string:
        return "No Topics"
    elif "no_sentiment" in target_string:
        return "No Sentiment"
    else:
        raise ValueError("run_name: {} didn't return a known type of model".format(target_string))



# first we extract all our temporal studies, their parameters and results
data_set_list = []
for s in shards_studied:
    data_set_list_single_shard = []
    df_original_shard = pd.read_csv(os.path.join(results_csv_file_home_path, base_analysis_filename_format_str.format(s)))
    df_temporal_shard = pd.read_csv(os.path.join(results_csv_file_home_path, temporal_analysis_filename_format_str.format(s)))
    df_temporal_shard_cols_list = list(df_temporal_shard.columns)
    # find unique params
    param_cols_inc_temp_list = ["run_name", "pred_steps"] + [x for x in df_temporal_shard_cols_list if "param" in x ]
    param_cols_sans_temp_list = [x for x in param_cols_inc_temp_list if not "period" in x ]
    param_cols_sans_temp_list_or_run_name = [x for x in param_cols_sans_temp_list if not "run_nam" in x ]
    df_param_data = df_temporal_shard[param_cols_sans_temp_list].drop_duplicates()
    
    for row in df_param_data.index:
        param_data = df_param_data.loc[row, param_cols_sans_temp_list_or_run_name]

        original_results = df_original_shard[df_original_shard[param_cols_sans_temp_list_or_run_name].eq(param_data).all(axis=1)]
        temporal_results = df_temporal_shard[df_temporal_shard[param_cols_sans_temp_list_or_run_name].eq(param_data).all(axis=1)]
        local_ID                    = int(original_results["local_ID"].values[0])
        different_temporal_values   = dict()

        #extract temporal results
        for ts_param, testing_value in zip(temporal_results[temporal_label_dict_column_name_str], temporal_results[variable_df_col_name_str]):
            different_temporal_values[ts_lookup[ts_param]] = testing_value

        data_set_new = {
            "series_name" : "local ID: {},\n".format(local_ID, return_model_type_name(df_param_data.loc[row, param_cols_sans_temp_list])),
            "single_value" : original_results[variable_df_col_name_str].values[0],
            "different_temporal_values" : different_temporal_values}
        data_set_list_single_shard += [data_set_new]
    data_set_list += [data_set_list_single_shard]

    print("number of series: " + str(len(data_set_list_single_shard)))
    plt.figure(figsize=(7, 6))
    for i, data_set in enumerate(data_set_list_single_shard):
        x_values          = list(data_set["different_temporal_values"].keys())
        x_values.sort()
        y_values_temporal       = [data_set["different_temporal_values"][x] for x in x_values]
        y_values_temporal_mean  = [sum(y_values_temporal)/len(y_values_temporal) for x in x_values]
        y_values_original       = [data_set["single_value"] for x in x_values]
        plt.plot(x_values, y_values_temporal, label = data_set["series_name"] + "testing score: " + original_scores_list[i], color=colors[i], linestyle="-") 
        plt.plot(x_values, y_values_temporal_mean, color=colors[i], linestyle=":")
        plt.plot(x_values, y_values_original, color=colors[i], linestyle="--") 

    plt.title("Testing Profitability Per Month\n(5 Min Prediction Horizon)")
    plt.xticks([f'{i}' for i in x_values], rotation='vertical')
    plt.xlabel(temporal_label_dict_chart_label_str, labelpad=20)
    plt.ylabel('testing score', labelpad=20)

    plt.subplots_adjust(bottom=0.3)
    #plt.subplots_adjust(left=0.05)
    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.50, 0.5), loc='right')
    
    plt.savefig(os.path.join(save_folder_path, 'paired_t_test_results {} mins.png'.format(str(int(5 * param_data["pred_steps"])))))
    plt.close



# save numerical results
df_temporal_results_csv = pd.DataFrame()
row = 0
for data_set_single_shard in data_set_list:
    for entry in data_set_single_shard:
        df_temporal_results_csv.loc[row, "name"] = entry["series_name"]
        df_temporal_results_csv.loc[row, "original_test_value"] = entry["single_value"]
        for col, temp_result in enumerate(entry["different_temporal_values"].values()):
            df_temporal_results_csv.loc[row, col] = temp_result
        row += 1
df_temporal_results_csv.to_csv(os.path.join(save_folder_path, 'temporal_results.csv'))


combinations            = []
name_combos             = []
single_values_combos    = []
for data_set_single_shard in data_set_list:
    all_series = []
    names = []
    single_values = []
    for entry in data_set_single_shard:
    
        all_series += [list(entry["different_temporal_values"].values()) + [entry["single_value"]]]
        names += [entry["series_name"]]
        single_values += [entry["single_value"]]

    # Perform paired t-test for all combinations
    combinations += list(itertools.combinations(all_series, 2))
    name_combos += list(itertools.combinations(names, 2))
    single_values_combos += list(itertools.combinations(single_values, 2))


# Store results in a list of dictionaries
results = []

for i, (series_a, series_b) in enumerate(combinations):
    t_statistic, p_value = ttest_rel(series_a, series_b)
    
    result_dict = {
        'Comparison': name_combos[i],

        'original_value_a'   : single_values_combos[i][0],
        'temporal_average_a' : sum(series_a) / len(series_a),
        'original_value_b'   : single_values_combos[i][1],
        'temporal_average_b' : sum(series_b) / len(series_b),

        'T-statistic': t_statistic,
        'P-value': p_value,
        'Statistically significant': p_value < 0.05,
    }
    
    results.append(result_dict)

# Create a DataFrame from the results list
df_ttest = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_ttest.to_csv(os.path.join(save_folder_path, 'paired_t_test_results.csv'), index=False)

# Print the DataFrame

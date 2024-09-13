import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MDAL
df_file_location = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\outputs\kfold_v3_global_results.csv"
save_folder      = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\human readable results\validation checks"
df = pd.read_csv(df_file_location)
output_list_A = ["PC", "score", "weighted"]
output_list_B = ["r2", "mse", "mae"]
dpi = 550
dot_size=0.5
line_size=dot_size*0.3

# input
pred_steps = 5
run_name_str = "multi_topic"

target_column_confidence = 0.02
#x_lims_1 = [0.0, 2.5]
#y_lims_1 = [-1, 0.25]
#y_lims_2 = [0, 30]
target_output_list = ["score", "score"]

# methods
def return_update_title_string(title_str_temp, title_str):
    title_str_temp = column_str_format.format("", target_output_str, pred_steps, target_column_confidence)
    if title_str == "" or title_str_temp == title_str:
        title_str = title_str_temp
    else:
        title_str += "_vs_" + title_str_temp
    return title_str

def return_title_with_final_updates(title_str, run_name_str, pred_steps):
    
    return title_str


#filter rows
mask_a = df["pred_steps"] == pred_steps
mask_b = df["run_name"].str.contains(run_name_str)
mask = mask_a & mask_b
df = df[mask]

#filter columns
columns_list = []
title_str = ""
stages_list = ["validation", "testing"]
for target_output_str, stage_str in zip(target_output_list, stages_list):
    if target_output_str in output_list_A:
        column_str_format = "{}_x_mins_{}_s{}_c{}"
        columns_list += [column_str_format.format(stage_str, target_output_str, "X", target_column_confidence)]
        title_str = return_update_title_string(column_str_format.format("", target_output_str, pred_steps, target_column_confidence), title_str)
    elif target_output_str in output_list_B:
        column_str_format = "{}_{}"
        columns_list += [column_str_format.format(stage_str, target_output_str)]
        title_str = return_update_title_string(column_str_format.format("", target_output_str)[1:], title_str)
title_str = "{}_predSteps_{}".format(run_name_str, pred_steps) + title_str
df = df[columns_list]

# produce labels
print(title_str)

# scatter figure
plt.figure()
plt.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title(title_str)
if "x_lims_1" in globals():
    plt.xlim(x_lims_1[0], x_lims_1[1])
if "y_lims_1" in globals():
     plt.ylim(y_lims_1[0], y_lims_1[1])
for i, point_id in enumerate(range(len(df))):
    color = 'red' if point_id > 49 else 'blue'
    plt.scatter(df.iloc[:,0].iloc[i], df.iloc[:,1].iloc[i], label='Points', color=color, marker='o')
    plt.annotate(point_id, (df.iloc[:,0].iloc[i], df.iloc[:,1].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)
plt.savefig(save_folder +  "\\" + title_str + ".png", dpi=dpi)
print(save_folder + "\\" + title_str)


# line chart#
plt.figure()
plt.plot(range(len(df.iloc[:,0])), df.iloc[:,0], label='Line 1', color='blue', marker='o', linewidth=line_size)
plt.plot(range(len(df.iloc[:,0])), df.iloc[:,1], label='Line 2', color='green', marker='s', linewidth=line_size)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
if "y_lims_2" in globals():
     plt.ylim(y_lims_2[0], y_lims_2[1])
plt.title('Two-Line Line Chart')
plt.legend()
plt.savefig(save_folder +  "\\" + title_str + "_lines.png", dpi=dpi)

df.to_csv(save_folder +  "\\" + title_str + ".csv")

print("{:.2f}".format(min(df.iloc[:,0])),", ", "{:.2f}".format(max(df.iloc[:,0])))
print("{:.2f}".format(min(df.iloc[:,1])),", ", "{:.2f}".format(max(df.iloc[:,1])))
print(len(df))

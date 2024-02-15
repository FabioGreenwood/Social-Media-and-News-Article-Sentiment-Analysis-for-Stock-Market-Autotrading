import os
import pickle
import shutil

"""
This file is made for the notation of prediction folders and the delete of multi-topic models that include topic volume calcs, this is because an error was found with these calcs.

These functions are controlled by booleans and the deletion functality is doubled locked
"""


target_dir = r"C:\Users\Public\fabio_uni_work\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\predictive_model"

add_notation = True
delete_defective_multi_topic_folders = False


if add_notation == True:
    # list locations
    folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]
    for f in folders:
        with open(os.path.join(target_dir, f, "input_dict.pkl"), "rb") as file:
            input_dict = pickle.load(file)
        text = ""
        for key in input_dict.keys():
            text += " -- " + key + "\n"
            for subkey in input_dict[key].keys():
                text += " - " + subkey + ": " + str(input_dict[key][subkey]) + "\n"
            text += "\n"
        with open(os.path.join(target_dir, f, "inputs_{}.txt".format(f[:4])), 'w') as file:
            file.write(text)
        if delete_defective_multi_topic_folders == True:
            print("please note the delete predictor functionality has been commented out for user safety, please <<uncomment>> should you want to use it")
            #if input_dict["senti_inputs_params_dict"]["topic_qty"] > 1 and input_dict["senti_inputs_params_dict"]["factor_topic_volume"] != False:
            #    print("deletion of:")
            #    print(input_dict["senti_inputs_params_dict"])
            #    shutil.rmtree(os.path.join(target_dir, f))

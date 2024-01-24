import os
import data_prep_and_model_training as FG_model_training
#import additional_reporting_and_model_trading_runs as FG_additional_reporting
from config import global_general_folder, global_outputs_folder, global_input_cols_to_include_list, global_index_cols_list, global_index_col_str, global_target_file_folder_path, global_feature_qty, global_outputs_folder_path, global_financial_history_folder_path, global_df_stocks_list_file          , global_start_time, global_error_str_1, global_random_state, global_scores_database, global_strptime_str, global_strptime_str_2, global_strptime_str_filename, global_precalculated_assets_locations_dict, global_designs_record_final_columns_list, SECS_IN_A_DAY, SECS_IN_AN_HOUR, FIVE_MIN_TIME_STEPS_IN_A_DAY
import shutil
import pickle

#%% parameters

predictor_folder_path = os.path.join(global_general_folder, "precalculated_assets", "predictive_model")
temp_storage_folder_name = "rename_temp_storage"

#%% methods


def move_files_in_folder(source_folder, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        return
    
    original_walk = os.walk(source_folder)

    # Check if the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for foldername, subfolders, filenames in os.walk(os.path.join(source_folder, folder)):
            # create detination folder
            destination_subfolder = os.path.join(destination_folder, os.path.relpath(destination_folder, folder))
            if not os.path.exists(destination_subfolder):
                os.makedirs(destination_subfolder)
            
            # Move files
            for filename in filenames:
                source_file_path = os.path.join(foldername, filename)
                destination_file_path = os.path.join(destination_subfolder, filename)
                shutil.move(source_file_path, destination_file_path)
            shutil.rmtree(foldername)




def move_contents(source_folder, destination_folder, folders_to_exclude_list=[]):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        return

    folders = list_folders(source_folder)

    for exclusion in folders_to_exclude_list:
        folders.remove(exclusion)
    
    # Check if the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the original folders 
    for folder in folders:
        original_subfolder      = os.path.join(source_folder, folder)
        shutil.move(original_subfolder, destination_folder)


            

def list_folders(location):
    # Check if the location exists
    if not os.path.exists(location):
        return

    # Get a list of folders in the specified location
    return [f for f in os.listdir(location) if os.path.isdir(os.path.join(location, f))]
    
    




#%% main line

# move all predictors to subfolder
temp_store_folder = os.path.join(predictor_folder_path, temp_storage_folder_name)
move_contents(predictor_folder_path, temp_store_folder, folders_to_exclude_list=["rename_temp_storage"])

folders = list_folders(temp_store_folder)
folders.sort()
for folder in folders:
    i+=1
    predictor_folder_path_original = os.path.join(temp_store_folder, folder)
    with open(os.path.join(predictor_folder_path_original, "input_dict.pkl")) as file:
        input_dict = pickle.load(file)
    predictor_name_new = FG_model_training.return_predictor_name(input_dict)
    
    predictor_folder_path_new = os.path.join(predictor_folder_path, predictor_name_new)
    print("changing:{} to {}".format(folder, predictor_name_new))
    shutil.move(predictor_folder_path_original, predictor_folder_path_new)
    
    #file_names = os.listdir(predictor_folder_path_original)
#
    #for file_name in file_names:
    #    shutil.move(os.path.join(predictor_folder_path_original, file_name), target_dir)



    #move_contents(predictor_folder_path_original, predictor_folder_path_new)



#file_names = os.listdir(source_dir)
#    
#for file_name in file_names:
#    shutil.move(os.path.join(source_dir, file_name), target_dir)

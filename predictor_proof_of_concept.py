"""
Required actions:
 - upgrade analysis of up down betting with for loop
 - change all mentions of close to "output"
- rearrange method declaration as needed
- actions about the drop NA needs to be done, this will unalign my data
- I'm unsure if the finder function is wiping all information between iterations

Dev notes:
 - Cound potentially add the function to view the model parameters scores

"""
#%% Import Modules and Basic Parameters

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns 

import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.simplefilter('always', category=any)

import seaborn as sns
import copy

#questionable modules
#from sklearn import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings


##basic parameters

#stratergy params
Features = 6
STEPS = 4
time_series_split_qty = 5
training_error_measure_main = 'neg_mean_squared_error'
train_test_split = 0.7 #the ratio of the time series used for training
CV_Reps=30
time_step_notation_sting = "d" #day here is for a day, update as needed

#file params
output_str = "close"
date_str  ='date'
target_file_name = 'Bitfinex_ETHUSD_d.csv'
target_file_folder_path = ""
feature_qty = 6
outputs_folder_path = ".\\outputs\\"




#Blank Variables (to remove problem messages)
df = pd.DataFrame()
tscv = ""
bscv = ""

#generic strings
step_forward_string = "Fwd_Output"
pred_col_name_str = "{}_" + time_step_notation_sting + "_" + step_forward_string



#%% methods definition

"""import_data methods"""
def import_data(target_file_folder_path, target_file_name, index_col=date_str):
    df = pd.read_csv(target_file_folder_path + target_file_name, skiprows=1, parse_dates=True, index_col=index_col)
    df = df.sort_index().drop('symbol', axis=1)
    return df


"""create_step_responces methods"""
#this method populates each row with the next X output results, this is done so that, each time step can be trained
#to predict the value of the next X steps
def create_step_responces_and_split_training_test_set(output_col_name=output_str, df=df, STEPS=STEPS, feature_qty=feature_qty, train_test_split=train_test_split, pred_col_name_str=pred_col_name_str):
    #create regressors
    for i in np.arange(1 ,STEPS + 1):
        #col_name = '{}d_Fwd_Output'.format(i)
        
        df[pred_col_name_str.format(i)] = df[output_str].shift(-i)
        
    df = df.dropna()

    #split regressors and responses
    #Features = 6

    X = df.iloc[:, :feature_qty]
    y = df.iloc[:, feature_qty:]

    split = int(len(df) * train_test_split)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test

"""create_model methods"""
#Let’s define a method that creates an elastic net model from sci-kit learn
#forecast more than one future time step, we will use a multi-output regressor wrapper that trains a separate model for each target time step
def build_model(_alpha, _l1_ratio):
    estimator = ElasticNet(
        alpha=_alpha,
        l1_ratio=_l1_ratio,
        fit_intercept=True,
        #normalize=False,
        precompute=False,
        max_iter=16,
        copy_X=True,
        tol=0.1,
        warm_start=False,
        positive=False,
        random_state=None,
        selection='random'
    )
    return MultiOutputRegressor(estimator, n_jobs=4)


"""time_series_blocking methods"""
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]





"""CV_Analysis methods"""
#GridSearchCV works by exhaustively searching all the possible combinations of the model’s parameters
#21
##FG_action: Does this need to be given an output
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, best_params):
    
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title(f"Grid Search Best Params: {best_params}", fontsize=12, fontweight='medium')
    ax.set_xlabel(name_param_1, fontsize=12)
    ax.set_ylabel('CV Average Score', fontsize=12)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    ax.legend(bbox_to_anchor=(1.02, 1.02))
##FG_action: Below needs to be parameterised
params = {
    'estimator__alpha':(0.1, 0.3, 0.5, 0.7, 0.9),
    'estimator__l1_ratio':(0.1, 0.3, 0.5, 0.7, 0.9)
}

def return_best_scores_from_CV_analysis(CV_Reps=CV_Reps, cv=bscv): #CV_snalysis_script
    scores = []
    for i in range(CV_Reps):
        model = build_model(_alpha=1.0, _l1_ratio=0.3)

        finder = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='r2',
            n_jobs=4,
            #iid=False,
            refit=True,
            cv=cv,  # change this to the splitter subject to test
            verbose=0,
            pre_dispatch=8,
            error_score=-999,
            return_train_score=True
            )

        #warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        warnings.filterwarnings('ignore') 
        finder.fit(X_train, y_train)
        #warnings.filterwarnings("default", category=ConvergenceWarning)

        best_params = finder.best_params_
        best_score = round(finder.best_score_,4)
        scores.append(best_score)
    
    return scores, best_params, finder



"""training methods"""







"""testing methods"""

def visualiser_up_down_confidence_tester(preds, X_test, STEPS, output_str, range_to_show=range(0,20,2), make_relative=True, output_name="Test", outputs_folder_path = ".//outputs//", figure_name = "test_output", pred_col_name_str=pred_col_name_str):
    
    df_realigned = return_realign_plus_minus_table(preds, X_test, STEPS, output_str, range_to_show=range(0,20,2), make_relative=True)
    #column_names = [output_str]
    diagram_labels_X = ["Actual"]
    for i in np.arange(1 ,STEPS + 1):
        #column_names = column_names + [col_name.format(i)]
        diagram_labels_X = diagram_labels_X + [str(i)]
    
    
    fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(9,5))
    #sns.heatmap(df_realigned, cmap="vlag_r", annot=True, center=0.00, ax=ax3, xticklabels=df_section_3.columns, yticklabels=df_section_3.index[0:7])
    sns.heatmap(df_realigned, cmap="vlag_r", annot=True, center=0.00, ax=ax3, xticklabels=diagram_labels_X, yticklabels=df_realigned.index.astype(str))#, xticklabels=df_section_3.columns, yticklabels=df_section_3.index[0:7])
    fig.suptitle("Realigned Fig_FG_Action: Change")
    fig.savefig(outputs_folder_path + output_name + ".png")
    df_realigned.to_csv(path_or_buf=outputs_folder_path + output_name + ".csv")
    fig.show()
    
    return fig, df_realigned


def return_realign_plus_minus_table(preds, X_test, STEPS, output_str, range_to_show=range(0,20,2), make_relative=True):
    col_name = 'pred_{}_' + time_step_notation_sting + '_before'
    column_names = [output_str]
    for i in np.arange(1 ,STEPS + 1):
        column_names = column_names + [col_name.format(i)]
        
    df_realigned = pd.DataFrame(columns=column_names)
    for index in range_to_show:
    #for y_test_index, pred_index in zip(X_test.index[STEPS:], preds.index[STEPS:]): #FG_Action, the preditions and the real data must use the same index
        X_test_index    = X_test.index[STEPS:][index]
        pred_index      = preds.index[STEPS:][index]
        if make_relative == True:
            previous_output_value = X_test[output_str][X_test.index[STEPS:][index-1]]
        else:
            previous_output_value = 0
        df_realigned.loc[X_test_index, output_str] = X_test[output_str][X_test_index] - previous_output_value
        for step_backs in range(1, STEPS+1):
            df_realigned.loc[X_test_index, col_name.format(step_backs)] = preds[pred_col_name_str.format(step_backs)][pred_index - step_backs] - previous_output_value
    return df_realigned.astype(float)



def return_df_X_day_plus_minus_accuracy(preds, X_test, STEPS, output_str, confidences_before_betting=[0], fixed_bet_penality=0, save=True, output_name="Test2", outputs_folder_path = ".//outputs//", figure_name = "test_output2", pred_col_name_str=pred_col_name_str):
    #df_realigned_temp = copy.deepcopy(df_realigned)
    results_X_day_plus_minus_accuracy = dict()
    results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_count = dict()
    results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_score = dict()
    
    temp_range = preds.index[:-STEPS]
    df_temp = return_realign_plus_minus_table(preds, X_test, STEPS, output_str, range_to_show=temp_range, make_relative=True)
    
    count_bets_with_confidence               = dict()
    count_correct_bets_with_confidence       = dict()
    count_correct_bets_with_confidence_score = dict()
        
    for steps_back in range(1, STEPS + 1):
        #initialise variables
        col_name = df_temp.columns[steps_back]
        count           = 0
        count_correct   = 0
        count_bets_with_confidence[steps_back]               = dict()
        count_correct_bets_with_confidence[steps_back]       = dict()
        count_correct_bets_with_confidence_score[steps_back] = dict()
        results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_count[steps_back] = dict()
        results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_score[steps_back] = dict()
        
        for confidence_level in confidences_before_betting:
            count_bets_with_confidence[steps_back][confidence_level]               = 0
            count_correct_bets_with_confidence[steps_back][confidence_level]       = 0
            count_correct_bets_with_confidence_score[steps_back][confidence_level] = 0
        
        for row_index in df_temp.index:
            count += 1
            #basic count scoring 
            if df_temp[output_str][row_index] * df_temp[col_name][row_index] > 1:
                count_correct += 1
            #bets with confidence scoring
            for confidence_level in confidences_before_betting:
                if   abs(df_temp[col_name][row_index]) > confidence_level and df_temp[output_str][row_index] * df_temp[col_name][row_index] > 1:
                    count_bets_with_confidence[steps_back][confidence_level] += 1
                    count_correct_bets_with_confidence[steps_back][confidence_level] += 1
                    count_correct_bets_with_confidence_score[steps_back][confidence_level] += abs(df_temp[col_name][row_index]) - fixed_bet_penality
                elif abs(df_temp[col_name][row_index]) > confidence_level and df_temp[output_str][row_index] * df_temp[col_name][row_index] < 1:
                    count_bets_with_confidence[steps_back][confidence_level] += 1
                    count_correct_bets_with_confidence_score[steps_back][confidence_level] -= abs(abs(df_temp[col_name][row_index]) - fixed_bet_penality)
    
        results_X_day_plus_minus_accuracy[steps_back] = count_correct / count
        results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_count[steps_back][confidence_level] = count_correct_bets_with_confidence[steps_back][confidence_level]       / count_bets_with_confidence[steps_back][confidence_level]
        results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_score[steps_back][confidence_level] = count_correct_bets_with_confidence_score[steps_back][confidence_level] / count_bets_with_confidence[steps_back][confidence_level]
        
    if save == True:
        df_temp.to_csv(path_or_buf=outputs_folder_path + output_name + ".csv")
        
    return results_X_day_plus_minus_accuracy, results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_score, results_BASIC_X_day_plus_minus_accuracy_betting_score_with_confidence_count
            
    
    print("Hello")



"""print_results methods"""





#%% Main Script
"""

This is a placeholder for the basic model script

df                               = import_data(data_folder_location)
X_train, y_train, X_test, y_test = create_step_responces(df, Steps)


models_list                      = create_model(list_of_model_types)

##This object informs the cross_val_score (the training method) of the time splits
btscv                            = time_series_blocking(n_splits=time_series_split_qty)

##This object scans all the modelling parameters to fine the best combo for fit
scores, best_params, finder      = return_best_scores_from_CV_snalysis(models_list, model_params_list)

##this trains and tests the models
response_models_list             = training(X, Y, models_list, best_params)

testing_results                  = testing(response_models_list, df)
void                             = print_results(testing_results)
"""

def run_design_of_experiments(
    target_file_folder_path,
    target_strings_to_filter_for_list,
    stock_name_dict,
    existing_output_cols_list,
    technicial_indicators_to_add,
    
    
    
    
    
    
):


df                               = import_data(target_file_folder_path, target_file_name, index_col=date_str)
X_train, y_train, X_test, y_test = create_step_responces_and_split_training_test_set(output_col_name=output_str, df=df, feature_qty=feature_qty, train_test_split=train_test_split)
btscv                            = BlockingTimeSeriesSplit(n_splits=time_series_split_qty)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
scores, best_params, finder      = return_best_scores_from_CV_analysis(CV_Reps=CV_Reps, cv=btscv)
#FG_Question: what is a finder object?
#Final Training
preds                            = pd.DataFrame(finder.predict(X_test), columns=df.iloc[:, Features:].columns)
#Results Analysis
fig, df_realigned                = visualiser_up_down_confidence_tester(preds, X_test, STEPS, output_str, outputs_folder_path = ".//outputs//", figure_name = "test_output", make_relative=True)
results_X_day_plus_minus_accuracy= return_df_X_day_plus_minus_accuracy(preds, X_test, STEPS, output_str, output_name="Test2", outputs_folder_path = ".//outputs//", figure_name = "test_output2", pred_col_name_str=pred_col_name_str)


# %%




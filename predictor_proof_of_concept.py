"""
Config Control
0.0.1 - intial population

Required actions:
 - after basic population an additional analysis looking at the accuracy of the model to predict an up or down is required
 - change all mentions of close to "output"
- rearrange method declaration as needed
- actions about the drop NA needs to be done, this will unalign my data

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
import seaborn as sns

#questionable modules
#from sklearn import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings


##basic parameters

#stratergy params
Features = 6
STEPS = 9
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


#%% methods definition

"""import_data methods"""
def import_data(target_file_folder_path, target_file_name, index_col=date_str):
    df = pd.read_csv(target_file_folder_path + target_file_name, skiprows=1, parse_dates=True, index_col=index_col)
    df = df.sort_index().drop('symbol', axis=1)
    return df


"""create_step_responces methods"""
#this method populates each row with the next X output results, this is done so that, each time step can be trained
#to predict the value of the next X steps
def create_step_responces(output_col_name=output_str, df=df, feature_qty=feature_qty, train_test_split=train_test_split):
    #create regressors
    for i in np.arange(1 ,STEPS):
        #col_name = '{}d_Fwd_Output'.format(i)
        col_name = str(i) + "_" + time_step_notation_sting + "_" + step_forward_string
        df[col_name] = df[output_str].shift(-i)
        
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

def return_best_scores_from_CV_snalysis(CV_Reps=CV_Reps, cv=bscv): #CV_snalysis_script
    btscv = BlockingTimeSeriesSplit(n_splits=time_series_split_qty)
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
            verbose=-1,
            pre_dispatch=8,
            error_score=-999,
            return_train_score=True
            )

    #warnings.filterwarnings("ignore", category=ConvergenceWarning)
    finder.fit(X_train, y_train)
    #warnings.filterwarnings("default", category=ConvergenceWarning)

    best_params = finder.best_params_
    best_score = round(finder.best_score_,4)
    scores.append(best_score)
    
    return scores, best_params, finder



"""training methods"""







"""testing methods"""



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

df                               = import_data(target_file_folder_path, target_file_name, index_col=date_str)
X_train, y_train, X_test, y_test = create_step_responces(output_col_name=output_str, df=df, feature_qty=feature_qty, train_test_split=train_test_split)
btscv                            = BlockingTimeSeriesSplit(n_splits=time_series_split_qty)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
scores, best_params, finder      = return_best_scores_from_CV_snalysis(CV_Reps=CV_Reps, cv=btscv)
#FG_Question: what is a finder
#Final Training
preds                            = pd.DataFrame(finder.predict(X_test), columns=df.iloc[:, Features:].columns)

def visualiser_up_down_confidence_tester(preds, y_test, STEPS, output_str, outputs_folder_path = ".//outputs//", figure_name = "test_output", make_relative=True):
    
    column_names = [output_str]
    col_name = 'pred_{}_' + time_step_notation_sting + '_before'
    for i in np.arange(1 ,STEPS):
        column_names = column_names + [col_name.format(i)]
    
    df_realigned = pd.DataFrame(columns=column_names)
    for y_test_index, pred_index in zip(y_test.index[STEPS:], preds.index[STEPS:]): #FG_Action, the preditions and the real data must use the same index
        df_realigned.loc[pred_index, output_str] = y_test[output_str][y_test_index]
        for step_backs in range(1, STEPS+1):
            df_realigned.loc[pred_index, col_name.format(i)] = preds[output_str][pred_index - step_backs]
    
    fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(9,5))
    #sns.heatmap(df_realigned, cmap="vlag_r", annot=True, center=0.00, ax=ax3, xticklabels=df_section_3.columns, yticklabels=df_section_3.index[0:7])
    sns.heatmap(df_realigned, cmap="vlag_r", annot=True, center=0.00, ax=ax3)#, xticklabels=df_section_3.columns, yticklabels=df_section_3.index[0:7])
    fig.suptitle("Realigned Fig_FG_Action: Change")
    fig.savefig('books_read.png')
    df_realigned.to_csv(path_or_buf=outputs_folder_path + "test_output" + ".csv")
    fig.show()
    
    return fig, df_realigned

fig, df_realigned = visualiser_up_down_confidence_tester(preds, y_test, STEPS, output_str, outputs_folder_path = ".//outputs//", figure_name = "test_output", make_relative=True)


fig.savefig('books_read.png')


print("Compelete")


# %%

import pandas as pd
import matplotlib.pyplot as plt
from  ksm.SSMLKVForestPredictorMPSC import SSMLKVForestPredictor
from ksm.DownloadHelper import *
from random import random
from random import randint
from ksm.utils import multilabel_train_test_split, get_features, save_report,generate_explainable_datasets
import sys
import os
import platform
import warnings


from ksm.explainability import generate_explainability_files

import multiprocessing as mp
# warnings.filterwarnings("ignore")


prefix = './'

input_path = prefix + 'datasets'
output_path = os.path.join(prefix, 'results')
model_path = os.path.join(prefix, 'results')
training_path = input_path



def calculate_parameters(params_dict):
    '''
    Function used to generate a parameter combination out of a dictionary with parameter limits. 
    When analyzing every element in a dictionary, it will select a specific value out from the key content, i.e.:
    'trees_quantity':(int, 10,20), # normally 20- 130 

    This will select a new integer within 10 to 20 (inclusive). 

    The final dictionary out from calculate_parameters will hold the same 'trees_quantity' key but with a defined value. 

    The function is smart enough to avoid this procedure if the key has a value that is not a tuple of bounds (type,min,max)
    '''
    params = dict()

    for p in params_dict:
        
        item = params_dict[p]
        
        if(type(item) is tuple):

            tp = item[0]
            mn = item[1]
            mx = item[2]
            if(tp == int):
                params[p] = randint(mn,mx)
            if(tp == float):
                params[p] = mn + (mx-mn)*random()
        elif(type(item) is list):
            if(type(item[0]) is tuple): # is a list of random parameters
                params[p] = []
                for idx in range(len(item)):
                    tp = item[idx][0]
                    mn = item[idx][1]
                    mx = item[idx][2]
                    params[p].append( mn + (mx-mn)*random() )
            else:
                params[p] = item[randint(0,len(item)-1)]
        else:
            params[p] = item # fixed to a value
    return params

def train(cycles_to_do = 1, do_explain = False):
    """
    Main train function. Every dataset needs the amount of labels in the ds_configs dictionary. 
    To add another, set a new dictionary entry with the key equal to the name of the file that has the data. 

    For example, if the algorithm is being tested with yeast.csv dataset, then the dictionary will look like:

    ds_configs = {
    "emotions":6, 
    "yeast":14, 
    }

    On top of this, to work with that specific dataset, *ds_name* var needs to be set to the dataset. 

    Once the algorithm has this, it will load the dataset. For our experiments we changed the labels column names to label_0 ... label_k

    If you have another label scheme, the code must be modified on this line:
    label_columns = [f"label_{i}" for i in range(0,ds_configs[ds_name])]  # for iris (3) ,for yeast(10) for ecoli(8), satimage(6)

    The parameters are defined on the paper, generally they mean:

    'a_r':always set to 1,    
    'trees_quantity':Amount of trees to use on the Random Forest, should be greater than 1, # normally 20- 130
    "M_groups": Amount of times to test the clustering, # 35 - 120 
    "N_attr": Attributes to select on each clusterization. This parameter is 2**num_of_features-1. So to select 3 features the value should be 2**3-1.
    'leaf_relative_instance_quantity':Percentage of instances to be on a node to continue the split process, 
    'scaler': always set to None,
    'do_ranking_split':always set to True,
    'p': an array containing the weight of each part of TAU metric. It will normalize them, so its possible to set numbers such as 1,2,3....10000. The algorithm will sum them all and divide by the sum to generate a perceptage summing to 1. ,
    'use_complex_model':Always set to True,
    'leaves_max_depth': Maximum depth on every tree,
    'leaves_min_samples_split': Minimum amount of instances needed to split on the leafa model,
    'leaves_min_samples_leaf': Minimum amount of instances needed to keep the process at the supervised leaf model, 
    'leaves_max_features': Always set to 'sqrt',
    'leaves_SS_N_reps': Repetitions needed at leaf to complete the pseudolabels,
    'leaves_SS_M_attributes': Attributes selected on leaf nodes to complete the pseudolabels, it is needed to calculate the distance from unsupervised to supervised instances and set a label. , 
    'distance_function': Always set to 'gower',
    'leaves_RF_estimators':Amount of trees on leaf model,
    'output_tree_sets':Always set to False,
    'm_iterations': Max amount of iterations to try clustering,
    'bagging_pct': Bagging percentage for each tree,
    'depth_limit': Maximum depth for every tree, 
    "output_quality":Always set to False

    The function will take a csv and synthetically create a semisupervised dataset by taking out the labels of the instances selected on "unlabeled_instances" variable. 
    An implementor would need to change this to select specific instances. For example: build a dataset with every supervised instance first. 
    
    The code follows the scikit-learn interface. SSMLKFVForestPredictorMPSC is a BaseEstimator that has a fit, predict and predict_with_proba functions.
    predict_with_proba will output two numpy matrices, one with the specific 1/0 label assignment and another one with the assignment probability. 

    """


    # for linux platforms, could be forkserver, fork, spawn.... forkserver works in mac
    if( sys.platform.find('linux') > -1 ):
        mp.set_start_method('spawn') # tried all in ubuntu... forkserver eventually had a broken pipe error. 
    ds_name = "emotions" # changed specs recently for explain tests... original parameters are overriden only. 

    print("Cpus " , mp.cpu_count() )
    print("Info " , platform.processor() )
    print("Sys " , sys.version_info )
    
    total_jobs = 4
    ds_configs = {
    "emotions":6, # multilabel
    }
    # getFromOpenML will convert automatically the classes found to a mutually exclusive multilabel
    print(training_path)
    dataset = getFromOpenML(ds_name,version="active",ospath=training_path+'/', download=False, save=False)

    label_columns = [f"label_{i}" for i in range(0,ds_configs[ds_name])]  # for iris (3) ,for yeast(10) for ecoli(8), satimage(6)
    # convert dataframe to its minimum size possible
    for label in label_columns:
        dataset[label] = pd.to_numeric( dataset[label] , downcast="unsigned" )

    parameters = {
    'a_r':1,    
    'trees_quantity':(int, 10,20), # normally 20- 130
    "M_groups":(int,20,120), # 35 - 120 
    "N_attr":(int, 2**5-1 , 2**16+1 ),
    'leaf_relative_instance_quantity':(float,0.05,0.17), 
    'scaler': None,
    'do_ranking_split':True,
    'p':[
        (float, 0,1),
        (float, 0,1),
        (float, 0,1),
        (float, 0,1),
        (float, 0,1),
        (float, 0,1),
        (float, 0,1)],
    'use_complex_model':True,
    'leaves_max_depth': (int,5,12),
    'leaves_min_samples_split':(int, 4,10),
    'leaves_min_samples_leaf':(int, 2,9), 
    'leaves_max_features':'sqrt',
    'leaves_SS_N_reps':(int,10,60),
    'leaves_SS_M_attributes':(int, 3,16), # they cannot be too large. tried with 40. 
    'distance_function':'gower',
    'leaves_RF_estimators':(int, 7,20),
    'output_tree_sets':False,
    'm_iterations':(int, 40,200),
    'bagging_pct':(float, 0.60, 0.90),
    'depth_limit': (int,0,10), # 4 and 20 originally, 0-16
    "output_quality":False,
    "min_supervised_per_leaf":2
    }

    final_list = []

    i = 0
    test_size = .30
    k_fold = 10
    unlabeled_ratio = .70

    train_set,test_set = multilabel_train_test_split(dataset, test_size=test_size, random_state=180, stratify=dataset[label_columns]) # .05 for the CLUS test as it was with train and test datasets
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True , inplace=True)
    
    best_label_rank = 0

    for i in range(cycles_to_do):
        params = calculate_parameters(parameters)
        print(params)
        auprc_curve = 0
        label_rank_average_precision = 0
        average_precision = 0
        auc_micro = 0
        auc_macro = 0
        hamming_loss = 0
        accuracy = 0

        instance_columns = get_features(train_set, label_columns)

        # one last time
        # we are not selecting the best one, yet. Just exploring the space with random. 
        labeled_instances, unlabeled_instances =  multilabel_train_test_split(train_set, test_size=unlabeled_ratio, random_state=141, stratify=train_set[label_columns]) # simulate unlabeled instances
        X = train_set[instance_columns]
        y = train_set[label_columns] 

        #print(y)
        labels_distrib = y.mean(axis=0)
        labels_distrib_supervised = y.loc[labeled_instances.index].mean(axis=0)
        predictor = SSMLKVForestPredictor(
                                unlabeledIndex=unlabeled_instances.index,
                                tag=ds_name,
                                hyper_params_dict = params,
                                is_multiclass = False,
                                do_random_attribute_selection = True, # wont be needed for this one
                                njobs=total_jobs # for mediamill case, too much memory
                                )
        predictor.fit(X,y)

        y_true = test_set[label_columns].to_numpy()
        x_test = test_set[instance_columns]

        explain = ""

        if( do_explain ):
            explain = "explain"
            rules_list = []
            print("Getting tree structure")
            predictor.get_tree_structure_df(rules_list)
            rules_df = pd.DataFrame(rules_list)
            rules_df["id"] =  rules_df["tree_id"].astype(str) + "_"  +   rules_df["node_id"].astype(str) 
            #rules_df.to_csv("rules.csv")
            del rules_list

            activations_list = [] 
            predictions, probabilities = predictor.predict_with_proba(x_test, y_true=y_true, activations_list=activations_list, explain_decisions = True, suffix=f"_{i}_") #y_true = y_true
            activation_df = pd.DataFrame(activations_list)
            activation_df["id"] =  activation_df["tree_id"].astype(str) + "_"  +   activation_df["node_id"].astype(str) 
            generate_explainable_datasets(rules_df,activation_df, label_columns )
            del activations_list
            generate_explainability_files(rules_df, labels_distrib,  labels_distrib_supervised, params["trees_quantity"], suffix=f"_{i}_")
            


        else:
            predictions, probabilities = predictor.predict_with_proba(x_test) #y_true = y_true

        results_pred = pd.DataFrame(predictions)
        results_prob = pd.DataFrame(probabilities)
        results_true = pd.DataFrame(y_true)

        params["auprc_curve_average_k_fold"] = auprc_curve/k_fold
        params["label_rank_average_precision_average_k_fold"] = label_rank_average_precision/k_fold
        params["average_precision_average_k_fold"] = average_precision/k_fold
        params["auc_micro_average_k_fold"] = auc_micro/k_fold
        params["auc_macro_average_k_fold"] = auc_macro/k_fold
        params["hamming_loss_average_k_fold"] = hamming_loss/k_fold
        params["accuracy_average_k_fold"] = accuracy/k_fold

        # as only the last model is kept, we are trusting that the performance is pretty close to the average
        o = save_report( model_path + '/', ds_name+f"{explain}_{unlabeled_ratio}_k_{i}_", y_true, predictions, probabilities, do_output=True, parameters=params)
        #print(o)
        final_list.append(o)

        if( explain == "explain"):
            print("testing on training data just supervised")
            y_true_train = train_set[label_columns].loc[labeled_instances.index].to_numpy()
            x_test_train = train_set[instance_columns].loc[labeled_instances.index]

            predictions, probabilities = predictor.predict_with_proba(x_test_train) #y_true = y_true
            o = save_report( model_path + '/', ds_name+f"test_on_train_{explain}_{unlabeled_ratio}_k_{i}_", y_true_train, predictions, probabilities, do_output=False, parameters=params)
        
    df = pd.DataFrame(data=final_list)
    df.to_csv(f'{explain}all_{ds_name}_{unlabeled_ratio}_params_full.csv')
    


if __name__ == '__main__':
    """
    To use explainability try loops_to_do=1 and "explain" parameter as there is a bug on the chart generator. If you are not interested on explainability feel free to ask for any amount of parameters tests via loops_to_do.

    To use the method: python ./test_basic <loops_to_do> <explain>
    Example: python ./test_basic 2 will generate two paramters combinations without explainability files

    Example: python ./test_basic 1 explain will generate one paramter combinations WITH explainability files
    
    
    """
    do_explain = False
    loops_to_do = 1

    if( len(sys.argv)>1):
        loops_to_do = int(sys.argv[1])

    if( len(sys.argv)>2):
        do_explain = (sys.argv[2] == "explain")
        
    train(loops_to_do, do_explain)
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
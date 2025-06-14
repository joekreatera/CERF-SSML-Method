import pandas as pd
import uuid
from .roc_auc_reimplementation import roc_auc as roc_auc_score
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import gower
import numba_gower
import numba_silhouette
from numba.typed import List

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain

from time import time
from scipy.spatial.distance import cdist

numba_signatures = dict()
enable_signatures = False

def multilabel_kfold_split(n_splits=5, shuffle=True, random_state= 180):
    return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from: 
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size,train_size=train_size,
                                random_state=random_state, stratify=None, shuffle=shuffle)
    
    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    
    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=random_state)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


def print_structure(tree_structure, tabs = ""):
    """
    tree should include a left and right dictionary along with columns and instances index. 

    left
    right
    depth
    index
    node_id
    tree_id
    columns
    is_leaf
    supervised
    unsupervised
    joint_columns
    label_inner_distance
    """


    t_id = str(tree_structure["tree_id"]).ljust(3)
    n_id = str(tree_structure["node_id"]).ljust(3)
    depth  = str(tree_structure["depth"]).ljust(3)
    depth = ""
    dist = str( round(tree_structure["label_inner_distance"]*100,2)).ljust(6)
    dist_g = str( round(tree_structure["gower_inner_distance"]*100,2)).ljust(6)
    
    q_a = str(len( tree_structure["index"] )).ljust(4)
    q_s = str(len( tree_structure["supervised"])).ljust(4)
    q_u = str(len( tree_structure["unsupervised"])).ljust(4)
    cols = tree_structure["joint_columns"]
    # remember that either the node is split or not, it cannot have one null child and another filled one
    if(tree_structure["left"] is not None):
        if( tabs == ""):
            print(f"{tabs}" f'{t_id} {n_id}{depth}d:{dist}g:{dist_g}c:{cols}q:({q_a}{q_s} {q_u})')
        else:
            print(f"{tabs}" f'{t_id} {n_id}{depth}d:{dist}g:{dist_g}q:({q_a}{q_s}{q_u})')
        print_structure(tree_structure["left"] , tabs + "(l) ")
        print_structure(tree_structure["right"] ,tabs + "(r) ")
    else:
        print(f"{tabs} |***  " f'{t_id} {n_id}{depth}d:{dist}g:{dist_g}q:({q_a}{q_s}{q_u})')
    #    print(tabs + "(l)")

    
    #else:
    #    print(tabs + "(r)")


def get_metrics( y_true = None, y_predicted = None, y_predicted_proba = None):
    results = dict()
    if(y_predicted_proba is not None):
        score_avg_precision = average_precision_score(y_true, y_predicted_proba)
        results["average_precision"] = score_avg_precision

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y_predicted_proba.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_predicted_proba[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_predicted_proba[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_predicted_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(y_true, y_predicted_proba, average="micro")
        avg_pr_micro = average_precision["micro"]
        results["auprc_curve"] = avg_pr_micro
        
    if(y_predicted_proba is not None):
        label_score_avg_precision = label_ranking_average_precision_score(y_true, y_predicted_proba)
        results["label_rank_average_precision"] = label_score_avg_precision
        
    if(y_predicted is not None):
        score_accuracy = accuracy_score(y_true, y_predicted)
        results["accuracy"] = score_accuracy
        
    if(y_predicted is not None):
        score_hamming = hamming_loss(y_true, y_predicted)
        results["hamming_loss"]= score_hamming
        
    if(y_predicted_proba is not None):
        score_auc_micro = roc_auc_score(y_true, y_predicted_proba, average='micro')
        results["auc_micro"]= score_auc_micro
        
    if(y_predicted_proba is not None):    
        score_auc_macro = roc_auc_score(y_true, y_predicted_proba, average='macro')
        results["auc_macro"]= score_auc_macro

    return results
        
def save_report(root, experiment_name, y_true = None, y_predicted = None, y_predicted_proba = None, do_output = False, parameters = None):
    results = open(root + '/' + experiment_name + f"_{uuid.uuid4()}.txt",'w')
    # results.write(f'Hyperparams selected {gs.best_params_}\n')
    # score_avg_precision = average_precision_score(y_arr, prediction_arr)

    
    if(y_predicted_proba is not None):
        score_avg_precision = average_precision_score(y_true, y_predicted_proba)
        results.write(f'Avg precision :\t {score_avg_precision}\n')
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y_predicted_proba.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_predicted_proba[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_predicted_proba[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_predicted_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(y_true, y_predicted_proba, average="micro")
        avg_pr_micro = average_precision["micro"]
        results.write(f'Micro Avg precision AUCPR :\t {avg_pr_micro}\n')
        """
        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        
        display.plot()
        _ = display.ax_.set_title("Micro-averaged over all classes")
        plt.show()
        """

    if(y_predicted_proba is not None):
        label_score_avg_precision = label_ranking_average_precision_score(y_true, y_predicted_proba)
        results.write(f'Label rank Avg precision :\t {label_score_avg_precision}\n')

    if(y_predicted is not None):
        score_accuracy = accuracy_score(y_true, y_predicted)
        results.write(f'Accuracy:\t{score_accuracy}\n')

    if(y_predicted is not None):
        score_hamming = hamming_loss(y_true, y_predicted)
        results.write(f'Haming:\t{score_hamming}\n')

    if(y_predicted_proba is not None):
        score_auc_micro = roc_auc_score(y_true, y_predicted_proba, average='micro')
        results.write(f'AUC.micro:\t{score_auc_micro}\n')

    if(y_predicted_proba is not None):    
        score_auc_macro = roc_auc_score(y_true, y_predicted_proba, average='macro')
        results.write(f'AUC.macro:\t{score_auc_macro}\n')

    if(y_predicted is not None):
        report = classification_report(y_true, y_predicted)
        results.write(report)

    if(y_predicted is not None):    
        conf = multilabel_confusion_matrix(y_true, y_predicted)

    if(y_predicted is not None):    
        results.write(f'TN\tFN\tTP\tFP\n')
        for i in conf:
            results.write(f'{i[0,0]}\t{i[1,0]}\t{i[1,1]}\t{i[0,1]}\n')
    
    
    if( parameters is not None):
        for i in parameters:
            results.write(f'{i}: {parameters[i]}\n')
        
        
    results.close()
    if(do_output):
        return {
        'score_avg_precision':score_avg_precision,
        'micro_avg_precision_AUPRC':avg_pr_micro,
        'label_score_avg_precision':label_score_avg_precision,
        'score_accuracy':score_accuracy,
        'score_hamming':score_hamming,
        'score_auc_micro':score_auc_micro,
        'score_auc_macro':score_auc_macro,
        'report':report,
        'parameters':parameters
        }

def get_features(train_set, labels):
    instance_columns = train_set.columns.to_list()
    for col in labels:
        #print('col:  ' + col )
        instance_columns.remove(col)
    return instance_columns

def transform_multiclass_to_multilabel(o_dataframe, class_column):
    
    unique_vals = o_dataframe[ class_column ].unique()
    dataframe = o_dataframe.drop(columns=[ class_column ])
    counter = 0
    for i in unique_vals:
        dataframe[f"label_{counter}"] = 0
        dataframe.loc[ o_dataframe[class_column] == i , f"label_{counter}" ] = 1
        counter+=1
    return dataframe


def generate_multiclass_compatibility_matrix(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set
    
    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")
    
    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose()

    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T)
    
    return intersection


def generate_feature_weights(feature_set):
    """
    will output feature weights according to algorithm on 
    A Consolidated Decision Tree-Based Intrusion Detection System for Binary and Multiclass Imbalanced Datasets
    and 
    Infinite Feature Selection:A Graph-based Feature Filtering Approach
    """
    feature_weights = np.zeros(feature_set.shape[1])
    # print(feature_weights.shape)
    alpha =0.5
    normalized_data = (feature_set - feature_set.min())/( feature_set.max() -  feature_set.min())
    std_devs = normalized_data.std(ddof=0)
    feature_counter= 0

    
    A = np.zeros(shape=(feature_weights.shape[0], feature_weights.shape[0]))
    I = np.eye(feature_weights.shape[0])
    for feature_i, feature_column_i in feature_set.items():
        fi_vs_fj = np.zeros(feature_weights.shape[0])
        counter = 0
        
        for feature_j, feature_column_j in feature_set.items():
            
            std_i_j = max( std_devs[feature_i], std_devs[feature_j] )
            
            # print(f'{feature_i} {feature_j}')
            corr_i_j = 1 - np.abs( feature_column_i.corr(feature_column_j, method='spearman')  ) 
            # print(f'comparing {feature_column_i} {feature_column_j}')
            
            fi_vs_fj[counter] = alpha*std_i_j + (1-alpha)*corr_i_j # is not counter but the actual correlation score
            counter += 1
        # yeah i know that this is going to calculate the same thing twice
        # feature_weights[feature_counter] =  fi_vs_fj.mean()
        A[feature_counter, :] = fi_vs_fj
        # check eigen values
        
        feature_counter += 1
    
    # print(A)
    rho_A = np.max( np.linalg.eig(A)[0] )
    # print(rho_A)
    inv_minus_I = np.linalg.inv(I - 0.9/rho_A*A ) - I
    S = np.ones(feature_weights.shape[0])*inv_minus_I
    # print(S) #lets use this one to select the N best features on each tree. 
    # select one at random and then all the most important and significant ones. 
    return S
    """
    print("-------------")
    feature_weights = S.mean(axis = 0)
    print(feature_weights)
    print("**********")
    print( np.argsort(feature_weights) )
    """
    
    
def generate_compatibility_matrix_counting_0s(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set

    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")

    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose(copy=True) # case for counting equal ones
    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T) # instance vs instance matrix

    compatibility_matrix_inv_A = 1 - train_set.loc[labeled_instances.index, label_columns] #inverse, 0 x 1 and viceversa 
    compatibility_matrix_inv_A_T = compatibility_matrix_inv_A.transpose(copy=True) # case for counting equal ones
    intersection_inv = compatibility_matrix_inv_A.dot(compatibility_matrix_inv_A_T) # instance vs instance matrix
    
    equal_labels = (intersection + intersection_inv)/len(label_columns)
    
    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            equal_labels[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            equal_labels.loc[i] = -1 # per row
            
            
    return equal_labels

def generate_cosine_distance_based_compatibility(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set

    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")

    labels_orig = train_set.loc[labeled_instances.index, label_columns]
    #print(labels_orig)
    labels_sum = (train_set.loc[labeled_instances.index, label_columns].sum(axis='columns') )**(0.5)
    labels_sum_t = pd.DataFrame()
    
    for l in label_columns:
        labels_sum_t = labels_sum_t.append( labels_sum.transpose() , ignore_index = True )
    labels_sum = 1/labels_sum_t.transpose()
    labels_sum.columns = label_columns
    #print(labels_sum)
    
    labels_normalized = labels_orig.mul(labels_sum)
    labels_normalized_t = labels_normalized.transpose(copy=True)
    compatibility_matrix = labels_normalized.dot(labels_normalized_t)
    
    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            compatibility_matrix[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            compatibility_matrix.loc[i] = -1 # per row
            
        
    return compatibility_matrix

    
def generate_compatibility_matrix(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set
    
    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")
    
    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose(copy=True)

    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T) # instance vs instance matrix

    # compatibility_A_T is the tranpose of just label columns. Agg counts and sets a row with all the sums per instance of all labels
    union = compatibility_matrix_A_T.agg(['sum']) # sum of ones 
    # transpose to generate a table of instances vs total number of 1's in labels
    union = union.transpose()
    #insert ones to operate after, a matrix multiplication
    union.insert(loc=0, column="ones", value= int(1)) # CA on paper
    # get the values to numpy array to thrash indices
    union_transpose_np_matrix = union.values
    #generate new dataframe with reordered rows, to be able to calculate union before intersection
    union = pd.DataFrame({'sum':union['sum'] , 'ones':union['ones']   }) # CB on paper
    #transpose to be able to multiply both matrices
    union = union.transpose()
    # get numpy 2d matrix
    union_np_matrix = union.values

    # indices to return the compatibility matrix to pandas dataframe
    colIndex = union.columns
    rowIndex = union.columns

    # magic moment obtaining union
    union_before_intersection_np_matrix = union_transpose_np_matrix.dot(union_np_matrix)
    # dataframe going to be final
    union_before_intersection = pd.DataFrame(data=union_before_intersection_np_matrix, index = rowIndex, columns = colIndex)
    # A+B-intersectionofAB
    union_minus_intersection = (union_before_intersection - intersection) + 0.00000001
    # probably add a very small epsilon to avoid 0 on union. Case in which an instance does not have any labels at all.
    # compatibility defined as the intersection/union
    compatibility_matrix = intersection/union_minus_intersection


    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            compatibility_matrix[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            compatibility_matrix.loc[i] = -1 # per row
            
        
    return compatibility_matrix


# requires the X to be a NxN
def silhouette_score(X, labels):
    return numba_silhouette.silhouette_scoreB(X, labels)

def pairwise_distances(X,Y=None,metric='euclidean', min_max_array = None, cat_features = None):
    """
    if one needs vector to matrix, the vector shoould be a matrix of 1 row, passed as the first argument to the function
    In any other case, just with gower distance, Y will not be regarded, and X should be the matrix to take distances
    from. 

    Cat features should be the set of column numbers with categorical data, only available for gower.

    """
    
    if(metric=='euclidean'):
        return euclidean_distances(X,Y=Y,squared=False)
        #if(Y is None): # scipy is not faster
        #    return cdist(X,X, metric='euclidean')
        #return cdist(X,Y, metric='euclidean')
    if(metric=='gower'): # only works for one matrix, or vector to matrix cases
        #print(X)
        # t1 = time()
        #mat =  gower.gower_matrix(X, Y, casting='unsafe', min_max_array=min_max_array)
        # return mat
        
        cats = None
        if( cat_features is None or len(cat_features) == 0):
            cats = List([-1])
        else:
            cats = List(cat_features)
        mn = None
        mx = None
        if( min_max_array is not None):
            mn = min_max_array[0]
            mx = min_max_array[1]
            
        if( X.shape[0] == 1 ):
            try:
                
                mat = numba_gower.gower_distance_vector_to_matrix(X,Y, cat_cols=cats, mn=mn, mx=mx )

                if(enable_signatures):
                    if("gower_distance_vector_to_matrix" not in numba_signatures or len(numba_gower.gower_distance_vector_to_matrix.signatures) < len(numba_signatures["gower_distance_vector_to_matrix"]) ):
                        numba_signatures["gower_distance_vector_to_matrix"] = numba_gower.gower_distance_vector_to_matrix.signatures
                        i = 0
                        for sig in range(0, len(numba_signatures["gower_distance_vector_to_matrix"])):
                            sc = find_instr( numba_gower.gower_distance_vector_to_matrix , keyword='subp', sig=sig)
                            numba_signatures[f"gower_distance_vector_to_matrix_sig_{sig}"] = sc
            except Exception as e:
                print("Signatures! " , numba_gower.gower_distance_vector_to_matrix.signatures)
                print(X, "  " , type(X))
                print(Y, "  " , type(Y))
                print(cats, "  " , type(cats))
                print(min_max_array , "  " , type(min_max_array) )
                raise e    
        else:
            #print("+_______")
            #print(cats)
            #print(X.shape)
            #print(X)
            mat =  numba_gower.gower_distance_matrix(X, cat_cols=cats, mn=mn, mx=mx ) # min_max_array=min_max_array
            #print("***************   "  , numba_signatures)
            #print("--------------- " , numba_gower.gower_distance_matrix.signatures)
            
            if(enable_signatures):
                if("gower_distance_matrix" not in numba_signatures or len(numba_gower.gower_distance_matrix.signatures) < len(numba_signatures["gower_distance_matrix"]) ):
                        numba_signatures["gower_distance_matrix"] = numba_gower.gower_distance_matrix.signatures
                        i = 0
                        for sig in range(0, len(numba_signatures["gower_distance_matrix"])):
                            sc = find_instr( numba_gower.gower_distance_matrix , keyword='subp', sig=sig)
                            numba_signatures[f"gower_distance_matrix_sig_{sig}"] = sc
        # print("Gower distance :", (time()-t1))
        return mat
    if(metric=='cosine'):

        #if(Y is None): # scipy is not faster, but consider using fastdist implementation... uses numba
        #    return cdist(X,X, metric='cosine')
        #return cdist(X,Y, metric='cosine')
    
        return cosine_distances(X,Y=Y)
    
    return None


def generate_explainable_datasets(rules_df, activations_df, labels_columns):
    rules_df.set_index("id", inplace=True)
    grouped = activations_df.groupby("id")
    df_count = grouped.count()["node_id"]

    df_macro_f1 = grouped.sum()
    # print(df_macro_f1.columns)
    results_by_class_f1 =pd.DataFrame()
    results_by_class_p = pd.DataFrame()
    results_by_class_r = pd.DataFrame()

    first = True
    for i in labels_columns:
        a = (2*df_macro_f1[f"{i}_tp"])/(  2*df_macro_f1[f"{i}_tp"] +  df_macro_f1[f"{i}_fp"] + df_macro_f1[f"{i}_fn"]  )
        b = (df_macro_f1[f"{i}_tp"])/(   df_macro_f1[f"{i}_tp"] + df_macro_f1[f"{i}_fp"]  )
        c = (df_macro_f1[f"{i}_tp"])/(    df_macro_f1[f"{i}_tp"] + df_macro_f1[f"{i}_fn"]  )

        if(first):
            first = False
            results_by_class_f1.index = a.index  
            results_by_class_p.index = b.index
            results_by_class_r.index = c.index

        results_by_class_f1[f"{i}"] = a.values
        results_by_class_p[f"{i}"] = b.values
        results_by_class_r[f"{i}"] = c.values

    results_by_class_f1.fillna(0, inplace=True)
    results_by_class_f1["macro_f1_score"] = results_by_class_f1.mean(axis=1)
    results_by_class_f1["activations"] = df_count

    results_by_class_p.fillna(0, inplace=True)
    results_by_class_p["macro_precision_score"] = results_by_class_p.mean(axis=1)
    results_by_class_p["activations"] = df_count

    results_by_class_r.fillna(0, inplace=True)
    results_by_class_r["macro_recall_score"] = results_by_class_r.mean(axis=1)
    results_by_class_r["activations"] = df_count
    
    rules_df["activations"] = df_count
    rules_df["macro_f1"] = results_by_class_f1["macro_f1_score"]
    rules_df["macro_precision"] = results_by_class_p["macro_precision_score"]
    rules_df["macro_recall"] = results_by_class_r["macro_recall_score"]


    df_micro_f1 = df_macro_f1

    df_micro_f1[f"labels_tp"] = df_micro_f1[ [s+"_tp" for s in labels_columns] ].sum(axis=1)
    df_micro_f1[f"labels_fp"] = df_micro_f1[ [s+"_fp" for s in labels_columns] ].sum(axis=1)
    df_micro_f1[f"labels_fn"] = df_micro_f1[ [s+"_fn" for s in labels_columns] ].sum(axis=1)

    df_micro_f1 = df_micro_f1[ [ "labels_tp" , "labels_fp" , "labels_fn"]  ]
    df_micro_f1["micro_f1_score"] = (2*df_micro_f1[f"labels_tp"])/( 2*df_micro_f1[f"labels_tp"] +  df_micro_f1[f"labels_fp"]  +  df_micro_f1[f"labels_fn"]  ) 
    df_micro_f1["micro_precision_score"] = (df_micro_f1[f"labels_tp"])/(  df_micro_f1[f"labels_tp"]  +  df_micro_f1[f"labels_fp"]  )
    df_micro_f1["micro_recall_score"] = (df_micro_f1[f"labels_tp"])/(  df_micro_f1[f"labels_tp"]  +  df_micro_f1[f"labels_fn"]  )

    rules_df["micro_f1"] = df_micro_f1["micro_f1_score"]
    rules_df["micro_precision"] = df_micro_f1["micro_precision_score"]
    rules_df["micro_recall"] = df_micro_f1["micro_recall_score"]
    

def print_numba_signatures():
    pass
    #print("print on this thread _______________ analysis")
    #print(numba_signatures)
    

def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            if count >= limit:
                return count
    return count
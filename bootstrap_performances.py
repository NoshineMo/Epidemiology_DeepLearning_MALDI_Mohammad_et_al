#python
"""
Packages:
	nympy as np
    sklearn.metrics
Functions:
	accuracy
    precision
    specificity
    sensibility
    recall
    f1
    mae
    get_bootstrap_confidence_interval
"""

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error

# Metrics to report
# (to be adapted for non-binary classification)

def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)
def balanced_accuracy(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, y_pred)
def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, pos_label=1)
def specificity(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, pos_label=0)
def vpn(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, pos_label=0)
def sensibility(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, pos_label=1)
def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, pos_label=1)
def f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, pos_label=1)
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
def f1_all(y_true, y_pred):
    return np.mean(metrics.f1_score(y_true, y_pred, average=None))
def recall_all(y_true, y_pred):
    return np.mean(metrics.recall_score(y_true, y_pred,average=None))
def precision_all(y_true, y_pred):
    return np.mean(metrics.precision_score(y_true, y_pred, average=None))
def naive_roc_auc_score(y_true, y_pred):
    #for regression and classification comparisons
    num_same_sign = 0
    num_pairs = 0

    for a in range(len(y_true)):
        for b in range(len(y_true)):
            if y_true[a] > y_true[b]:
                num_pairs += 1
                if y_pred[a] > y_pred[b]:
                      num_same_sign += 1
                elif y_pred[a] == y_pred[b]:
                      num_same_sign += .5

    return num_same_sign / num_pairs


all_metrics = {
    'acc':accuracy,
    'prec':precision,
    'spec':specificity,
    'rec':recall,
    'f1':f1,
    'mae':mae,
    'f1 all':f1_all,
    'recall all':recall_all,
    'precision all':precision_all,
    'naive roc auc':naive_roc_auc_score
}

metrics_for_classif_all = {
    'acc':accuracy,
    'f1 all':f1_all,
    'recall all':recall_all,
    'precision all':precision_all,
    'naive roc auc':naive_roc_auc_score,
    'balanced accuracy': balanced_accuracy
}

metrics_for_regression_all = {
    'mae':mae}

all_metrics_parapsi = {
    'acc':accuracy,
    'prec':precision,
    'spec':specificity,
    'vpn':vpn,
    'rec':recall,
    'f1':f1
}

def get_bootstrap_confidence_interval(y_true, y_pred, metric_functions, 
                                      draw_number=1000, alpha=5.0, return_outputs=False):
    np_y_true = np.asarray(y_true)
    np_y_pred = np.asarray(y_pred)
    if type(metric_functions) == list:
        metric_function_list = metric_functions
    elif type(metric_functions) == dict:
        metric_names, metric_function_list = zip(*[(n, m) for n, m in metric_functions.items()])
    else:
        raise TypeError(metric_functions)
        
    all_outputs = []
    # Perform `draw_number` draw with replacement of the sample
    # and compute the metrics at each time
    for i in range(draw_number):
        # random draw with replacement
        random_indices = np.random.randint(low=0, high=len(y_pred), size=len(y_pred))
        this_y_pred = np_y_pred[random_indices]
        this_y_true = np_y_true[random_indices]
        # get metrics for this random sample
        all_outputs.append([m(this_y_true, this_y_pred) for m in metric_function_list])
    np_outputs = np.array(all_outputs)
    # calculate 95% confidence intervals (1 - alpha)
    lower_p = alpha / 2
    lower = np.maximum(0, np.percentile(all_outputs, lower_p, axis=0)) #change to mae
    upper_p = (100 - alpha) + (alpha / 2)
    upper = np.minimum(1, np.percentile(all_outputs, upper_p, axis=0)) #change to mae
    medians = np.median(np_outputs, axis=0)
    bottom_whisker_pos = np.min(np_outputs, axis=0)
    top_whisker_pos = np.max(np_outputs, axis=0)
    means = np.mean(np_outputs, axis=0)
    
    if type(metric_functions) == list:
        result = [{"main":m(y_true, y_pred), "mean":a, "med":b, "q1":c, "q3":d, "whislo":e, "whishi":f} 
                 for m, a, b, c, d, e, f in zip(metric_function_list, means, medians, lower, upper, bottom_whisker_pos, top_whisker_pos)]
    elif type(metric_functions) == dict:
        result = {n:{"main":m(y_true, y_pred), "mean":a, "med":b, "q1":c, "q3":d, "whislo":e, "whishi":f} 
                 for n, m, a, b, c, d, e, f in zip(metric_names, metric_function_list, means, medians, lower, upper, bottom_whisker_pos, top_whisker_pos)}
    else:
        raise TypeError(metric_functions)
    if return_outputs:
        return result, all_outputs
    else:
        return result


def get_bootstrap_confidence_interval_on_error(y_true, y_pred, metric_functions, draw_number=1000, alpha=5.0, return_outputs=False):
    np_y_true = np.asarray(y_true)
    np_y_pred = np.asarray(y_pred)
    if type(metric_functions) == list:
        metric_function_list = metric_functions
    elif type(metric_functions) == dict:
        metric_names, metric_function_list = zip(*[(n, m) for n, m in metric_functions.items()])
    else:
        raise TypeError(metric_functions)
        
    all_outputs = []
    # Perform `draw_number` draw with replacement of the sample
    # and compute the metrics at each time
    for i in range(draw_number):
        # random draw with replacement
        random_indices = np.random.randint(low=0, high=len(y_pred), size=len(y_pred))
        this_y_pred = np_y_pred[random_indices]
        this_y_true = np_y_true[random_indices]
        # get metrics for this random sample
        all_outputs.append([m(this_y_true, this_y_pred) for m in metric_function_list])
    np_outputs = np.array(all_outputs)
    # calculate 95% confidence intervals (1 - alpha)
    lower_p = alpha / 2
    lower = np.maximum(0, np.percentile(all_outputs, lower_p, axis=0)) #change to mae
    upper_p = (100 - alpha) + (alpha / 2)
    upper = np.percentile(all_outputs, upper_p, axis=0) #change to mae
    medians = np.median(np_outputs, axis=0)
    bottom_whisker_pos = np.min(np_outputs, axis=0)
    top_whisker_pos = np.max(np_outputs, axis=0)
    means = np.mean(np_outputs, axis=0)
    
    if type(metric_functions) == list:
        result = [{"main":m(y_true, y_pred), "mean":a, "med":b, "q1":c, "q3":d, "whislo":e, "whishi":f} 
                for m, a, b, c, d, e, f in zip(metric_function_list, means, medians, lower, upper, bottom_whisker_pos, top_whisker_pos)]
    elif type(metric_functions) == dict:
        result = {n:{"main":m(y_true, y_pred), "mean":a, "med":b, "q1":c, "q3":d, "whislo":e, "whishi":f} 
                for n, m, a, b, c, d, e, f in zip(metric_names, metric_function_list, means, medians, lower, upper, bottom_whisker_pos, top_whisker_pos)}
    else:
        raise TypeError(metric_functions)
    if return_outputs:
        return result, all_outputs
    else:
        return result



#def get_bootstrap_confidence_interval(y_true, y_pred, metric_functions, 
#                                      draw_number=1000, alpha=5.0, return_outputs=False):
#    np_y_true = np.asarray(y_true)
#    np_y_pred = np.asarray(y_pred)
#    if type(metric_functions) == list:
#        metric_function_list = metric_functions
#   elif type(metric_functions) == dict:
#        metric_names, metric_function_list = zip(*[(n, m) for n, m in metric_functions.items()])
#    else:
#        raise TypeError(metric_functions)
#        
#    all_outputs = []
#    # Perform `draw_number` draw with replacement of the sample
#    # and compute the metrics at each time
#    for i in range(draw_number):
#        # random draw with replacement
#        random_indices = np.random.randint(low=0, high=len(y_pred), size=len(y_pred))
#        this_y_pred = np_y_pred[random_indices]
#        this_y_true = np_y_true[random_indices]
#        # get metrics for this random sample
#        all_outputs.append([m(this_y_true, this_y_pred) for m in metric_function_list])
#    np_outputs = np.array(all_outputs)
#    # calculate 95% confidence intervals (1 - alpha)
#    lower_p = alpha / 2
#    lower = np.maximum(0, np.percentile(all_outputs, lower_p, axis=0))
#    upper_p = (100 - alpha) + (alpha / 2)
#    upper = np.minimum(1, np.percentile(all_outputs, upper_p, axis=0))
#    medians = np.median(np_outputs, axis=0)
#    means = np.mean(np_outputs, axis=0)
    
#    if type(metric_functions) == list:
#        result = [{"main":m(y_true, y_pred), "mean":a, "median":b, "lower_p":c, "upper_p":d} 
#                 for m, a, b, c, d in zip(metric_function_list, means, medians, lower, upper)]
#    elif type(metric_functions) == dict:
#        result = {n:{"main":m(y_true, y_pred), "mean":a, "median":b, "lower_p":c, "upper_p":d} 
#                 for n, m, a, b, c, d in zip(metric_names, metric_function_list, means, medians, lower, upper)}
#    else:
#        raise TypeError(metric_functions)
#    if return_outputs:
#        return result, all_outputs
#    else:
#        return result
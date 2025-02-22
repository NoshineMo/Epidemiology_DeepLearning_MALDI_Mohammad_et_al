from libraries_utils import *
from regression_roc_auc import *
from bootstrap_performances import *

#TEST_ACCURACY , TEST_BALANCED_ACCURACY, TEST_RECALL, TEST_PRECISION , TEST_F1_SCORE, TEST_CM, TEST_BOOTSTRAP, TEST_ROC_AUC, TEST_R2_SCORE = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
#TEST_MAE, TEST_ACC_BY_CAT = collections.defaultdict(list), collections.defaultdict(list)

def performances(y_test,y_test_pred, types, collection_feature):
    
    if types=='classification':

        TEST_ACCURACY , TEST_BALANCED_ACCURACY, TEST_RECALL, TEST_PRECISION , TEST_F1_SCORE, TEST_CM, TEST_ROC_AUC = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        TEST_ACC_BY_CAT =  collections.defaultdict(list)

        acc_test = accuracy_score(y_test.ravel(),y_test_pred)
        balanced_acc_test = balanced_accuracy_score(y_test.ravel(),y_test_pred)
        roc_auc_test = naive_roc_auc_score(y_test.ravel(),y_test_pred)
        score_f1_test = f1_score(y_test.ravel(),y_test_pred, average=None)
        matrice_confusion_test = tf.math.confusion_matrix(y_test.ravel(), y_test_pred)

        recall_score_test = recall_score(y_test.ravel(), y_test_pred, average=None)
        precision_score_test = precision_score(y_test.ravel(), y_test_pred, average=None)
        #bootstrap = get_bootstrap_confidence_interval(y_test.ravel(), y_test_pred, {'acc':accuracy})

        cm = confusion_matrix(y_test.ravel(),y_test_pred)
        acc_by_cat = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        acc_by_cat = acc_by_cat.diagonal()

        print ('Test accuracy de {}'.format(acc_test))
        print (' '*len('Test') + 'Balanced accuracy de {}'.format(balanced_acc_test))
        print (' '*len('Test') + ' score F1 de {}'.format(score_f1_test))
        print ('Matrice de Confusion {} '.format(matrice_confusion_test))

        print("Recall Score : ", recall_score_test)
        print("Precision Score : ", precision_score_test)
        print("ROC AUC Score : ", roc_auc_test)

        TEST_ACCURACY[collection_feature].append(acc_test)
        TEST_BALANCED_ACCURACY[collection_feature].append(balanced_acc_test)
        TEST_ROC_AUC[collection_feature].append(roc_auc_test)
        TEST_RECALL[collection_feature].append(recall_score_test)
        TEST_PRECISION[collection_feature].append(precision_score_test)
        TEST_F1_SCORE[collection_feature].append(score_f1_test)
        TEST_CM[collection_feature].append(matrice_confusion_test)
        TEST_ACC_BY_CAT[collection_feature].append(acc_by_cat)
        
        return TEST_ACCURACY, TEST_BALANCED_ACCURACY, TEST_RECALL, TEST_PRECISION,TEST_F1_SCORE, TEST_ACC_BY_CAT, TEST_CM, TEST_ROC_AUC
        
    elif types=='regression':

        TEST_ROC_AUC, TEST_R2_SCORE = collections.defaultdict(list), collections.defaultdict(list)
        TEST_MAE = collections.defaultdict(list)

        
        mae_test = mean_absolute_error(y_test,y_test_pred)
        roc_auc_test = naive_roc_auc_score(y_test,y_test_pred)
        r2 = r2_score(y_test,y_test_pred)
        print ('Test mae de {}'.format(mae_test))
        print("ROC AUC Score : ", roc_auc_test)
        print("R^2 Score : ", r2)
        TEST_MAE[collection_feature].append(mae_test)
        TEST_ROC_AUC[collection_feature].append(roc_auc_test)
        TEST_R2_SCORE[collection_feature].append(r2)
        
        return TEST_MAE, TEST_ROC_AUC, TEST_R2_SCORE
    
    else :
        raise ValueError('Type non reconnu')


def bootstrap_performances(y_test,y_test_pred, types, collection_feature):
    
    if types=='classification':
        TEST_BOOTSTRAP = collections.defaultdict(list)
        
        bootstrap_perf_all_classif = get_bootstrap_confidence_interval(y_test, y_test_pred, metrics_for_classif_all)
        TEST_BOOTSTRAP[collection_feature].append(bootstrap_perf_all_classif)
        
        return TEST_BOOTSTRAP
        
    elif types=='regression':
        TEST_BOOTSTRAP = collections.defaultdict(list)
        
        bootstrap_perf_roc_auc = get_bootstrap_confidence_interval(y_test, y_test_pred, {'naive roc auc':naive_roc_auc_score})
        bootstrap_perf_mae = get_bootstrap_confidence_interval_on_error(y_test, y_test_pred, {'mae':mae})
        
        TEST_BOOTSTRAP[collection_feature].append([bootstrap_perf_roc_auc, bootstrap_perf_mae])
        #TEST_BOOTSTRAP[collection_feature].append(bootstrap_perf_mae)
        
        return TEST_BOOTSTRAP
    
    else :
        raise ValueError('Type non reconnu')
    

def show_table_results(TEST_ACCURACY, TEST_RECALL, TEST_PRECISION,TEST_F1_SCORE, TEST_ACC_BY_CAT, TEST_CM, data_name, position = 0): #edit for regression
    
    table_results= pd.DataFrame()
    table_results['Metrics'] = ['Test accuracy',
                               "F1 score by categories", 'Test Precision/VPP by categories',
                               'Test Recall/Sensibility by categories','Accuracy by categories score']
    table_results = table_results.set_index('Metrics')
    for key in TEST_ACCURACY:
        acc = TEST_ACCURACY[key][position]
        f1_score = np.mean(TEST_F1_SCORE[key],axis=1)[position] # to be reviewed for classif 3 classses or more
        precision = np.mean(TEST_PRECISION[key],axis=1)[position] # to be reviewed for classif 3 classses or more
        recall = np.mean(TEST_RECALL[key],axis=1)[position] # to be reviewed for classif 3 classses or more
        table_results['{}'.format(key)] = ["{:.2f}".format(int(acc*100)/100), "{:.2f}".format(int(f1_score*100)/100),
                                          "{:.2f}".format(int(precision*100)/100), "{:.2f}".format(int(recall*100)/100),
                                          np.round(TEST_ACC_BY_CAT[key][position],2)] # int(x*100)/100
        
    table_results = table_results.style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'),
                                                                                                                         ('font-size', '20px')]}])

    return table_results


def show_table_results_for_cv(TEST_ACCURACY, TEST_RECALL, TEST_PRECISION,TEST_F1_SCORE, TEST_ACC_BY_CAT, TEST_CM, data_name, position = 0): #edit for regression
    # position: clone 0/no clone 1, anopheles: categories 0,1,2 or species: 0, 1 or 2
    table_results= pd.DataFrame()
    table_results['Metrics'] = ['Test accuracy',
                               "F1 score by categories", 'Test Precision/VPP by categories',
                               'Test Recall/Sensibility by categories','Accuracy by categories score']
    table_results = table_results.set_index('Metrics')
    for key in TEST_ACCURACY:
        acc = np.mean(TEST_ACCURACY[key])
        f1_score = np.mean(TEST_F1_SCORE[key],axis=0) # to be reviewed for classif 3 classses or more
        precision = np.mean(TEST_PRECISION[key],axis=0) # to be reviewed for classif 3 classses or more
        recall = np.mean(TEST_RECALL[key],axis=0) # to be reviewed for classif 3 classses or more
        acc_by_cat = np.mean(TEST_ACC_BY_CAT[key],axis=0)
        table_results['{}'.format(key)] = ["{:.2f}".format(int(acc*100)/100), np.round(f1_score,2),
                                          np.round(precision,2), np.round(recall,2),
                                          np.round(acc_by_cat,2)] # int(x*100)/100
        
    table_results = table_results.style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'),
                                                                                                                         ('font-size', '20px')]}])

    return table_results


def show_table_results_classif(TEST_ACCURACY, TEST_BALANCED_ACCURACY, TEST_RECALL, TEST_PRECISION,TEST_F1_SCORE, TEST_ACC_BY_CAT, TEST_CM, TEST_ROC_AUC,TEST_BOOTSTRAP, data_name, position = 0):
    
    table_results = pd.DataFrame()
    table_results['Metrics'] = ['Test accuracy', " ", 'Test balanced accuracy', " ", "Mean F1 score", " ", "F1 score by categories", "Mean precision", " ",
                                    "Precision by categories", 'Mean recall', " ",'Recall by categories','Accuracy by categories', 'Roc AUC score', " "]
    table_results = table_results.set_index('Metrics')
    for key in TEST_ACCURACY:
        acc = TEST_ACCURACY[key][position]
        balanced_acc = TEST_BALANCED_ACCURACY[key][position]
        f1_score = np.mean(TEST_F1_SCORE[key],axis=1)[position]
        precision = np.mean(TEST_PRECISION[key],axis=1)[position]
        recall = np.mean(TEST_RECALL[key],axis=1)[position]
        roc_auc = TEST_ROC_AUC[key][position]

        table_results['{}'.format(key)] = ["{:.2f}".format(int(acc*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['acc']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['acc']['q3'],2)),
                                               "{:.2f}".format(int(balanced_acc*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['balanced accuracy']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['balanced accuracy']['q3'],2)),
                                               "{:.2f}".format(int(f1_score*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['f1 all']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['f1 all']['q3'],2)),
                                               np.round(TEST_F1_SCORE[key][position],2),
                                               "{:.2f}".format(int(precision*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['precision all']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['precision all']['q3'],2)),
                                               np.round(TEST_PRECISION[key][position],2),
                                               "{:.2f}".format(int(recall*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['recall all']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['recall all']['q3'],2)),
                                               np.round(TEST_RECALL[key][position],2),
                                               np.round(np.mean(TEST_ACC_BY_CAT[key], axis = 0),2),
                                               "{:.2f}".format(int(roc_auc*100)/100), 
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['naive roc auc']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['naive roc auc']['q3'],2))]

    if data_name is not None:
        table_results = table_results .style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'), ('font-size', '20px')]}])

    return table_results


def show_table_results_regression(TEST_MAE, TEST_ROC_AUC, TEST_R2_SCORE, TEST_BOOTSTRAP, data_name, position = 0):
    
    table_results = pd.DataFrame()
    table_results['Metrics'] = ['Test MAE', " ", 'Roc AUC score', " ", "R2 score"]
    table_results = table_results.set_index('Metrics')
    for key in TEST_MAE:
        mae = TEST_MAE[key][position]
        roc_auc = TEST_ROC_AUC[key][position]
        r2 = TEST_R2_SCORE[key][position]

        table_results['{}'.format(key)] = ["{:.2f}".format(int(mae*100)/100),
                                            '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position][1]['mae']['q1'],2), np.round(TEST_BOOTSTRAP[key][position][1]['mae']['q3'],2)),
                                            "{:.2f}".format(int(roc_auc*100)/100), 
                                            '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position][0]['naive roc auc']['q1'],2), np.round(TEST_BOOTSTRAP[key][position][0]['naive roc auc']['q3'],2)),
                                          "{:.2f}".format(int(r2*100)/100)]
        
    if data_name is not None:
        table_results = table_results.style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'), ('font-size', '20px')]}])

    return table_results



# For the ML part

def performances_ml(y_test,y_test_pred, types, collection_feature):
    if types=='classification':
        TEST_ACCURACY , TEST_BALANCED_ACCURACY = collections.defaultdict(list), collections.defaultdict(list)
        acc_test = accuracy_score(y_test.ravel(),y_test_pred)
        balanced_acc_test = balanced_accuracy_score(y_test.ravel(),y_test_pred)
        print ('Test accuracy de {}'.format(acc_test))
        print (' '*len('Test') + 'Balanced accuracy de {}'.format(balanced_acc_test))
        TEST_ACCURACY[collection_feature].append(acc_test)
        TEST_BALANCED_ACCURACY[collection_feature].append(balanced_acc_test)
        return TEST_ACCURACY, TEST_BALANCED_ACCURACY
        
    elif types=='regression':
        TEST_MAE = collections.defaultdict(list)
        mae_test = mean_absolute_error(y_test,y_test_pred)
        return TEST_MAE
    
    else :
        raise ValueError('Type non reconnu')
        
        


def bootstrap_performances_ml(y_test,y_test_pred, types, collection_feature):
    if types=='classification':
        TEST_BOOTSTRAP = collections.defaultdict(list)
        bootstrap_perf_all_classif = get_bootstrap_confidence_interval(y_test, y_test_pred, metrics_for_classif_all)
        TEST_BOOTSTRAP[collection_feature].append(bootstrap_perf_all_classif)
        return TEST_BOOTSTRAP
        
    elif types=='regression':
        TEST_BOOTSTRAP = collections.defaultdict(list)
        bootstrap_perf_roc_auc = get_bootstrap_confidence_interval(y_test, y_test_pred, {'naive roc auc':naive_roc_auc_score})
        bootstrap_perf_mae = get_bootstrap_confidence_interval_on_error(y_test, y_test_pred, {'mae':mae})
        TEST_BOOTSTRAP[collection_feature].append([bootstrap_perf_roc_auc, bootstrap_perf_mae])
        #TEST_BOOTSTRAP[collection_feature].append(bootstrap_perf_mae)
        return TEST_BOOTSTRAP
    
    else :
        raise ValueError('Type non reconnu')
    

def show_table_results_classif_ml(TEST_ACCURACY, TEST_BALANCED_ACCURACY,TEST_BOOTSTRAP, data_name, position = 0):
    
    table_results = pd.DataFrame()
    table_results['Metrics'] = ['Test accuracy', " ", 'Test balanced accuracy', " "]
    table_results = table_results.set_index('Metrics')
    for key in TEST_ACCURACY:
        acc = TEST_ACCURACY[key][position]
        balanced_acc = TEST_BALANCED_ACCURACY[key][position]

        table_results['{}'.format(key)] = ["{:.2f}".format(int(acc*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['acc']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['acc']['q3'],2)),
                                               "{:.2f}".format(int(balanced_acc*100)/100),
                                               '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position]['balanced accuracy']['q1'],2), np.round(TEST_BOOTSTRAP[key][position]['balanced accuracy']['q3'],2))]

    table_results = table_results .style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'), ('font-size', '20px')]}])

    return table_results


def show_table_results_regression_ml(TEST_MAE, TEST_BOOTSTRAP, data_name, position = 0):
    
    table_results = pd.DataFrame()
    table_results['Metrics'] = ['Test MAE', " "]
    table_results = table_results.set_index('Metrics')
    for key in TEST_MAE:
        mae = TEST_MAE[key][position]
        roc_auc = TEST_ROC_AUC[key][position]
        r2 = TEST_R2_SCORE[key][position]

        table_results['{}'.format(key)] = ["{:.2f}".format(int(mae*100)/100),
                                            '[{0},{1}]'.format(np.round(TEST_BOOTSTRAP[key][position][1]['mae']['q1'],2), np.round(TEST_BOOTSTRAP[key][position][1]['mae']['q3'],2))]

    table_results = table_results .style.set_caption("{} results".format(data_name)).set_table_styles([{'selector': 'caption','props': [('color', 'black'), ('font-size', '20px')]}])

    return table_results


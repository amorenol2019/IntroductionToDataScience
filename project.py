#------ADABOOST + c4.5
import time
import pandas as pd
import numpy as np
from rotation_forest import RotationForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def get_values_trained():
    hrv = pd.read_excel('DATA/AnalizaHRVRhythmDataset.xlsx')
    hrv.replace(('yes', 'no'), (1, 0), inplace = True)
    X = hrv.drop(["signal_type"], axis = 1).values
    Y = hrv["signal_type"].values
    train_X, test_X, train_Y, test_Y = train_test_split(X,Y)

    return train_X, test_X, train_Y, test_Y

def switcher_clases(i):
    switch_class = {
        0: ABI,
        1: AFIB,
        2: ATR,
        3: NSR,
        4: PAC,
        5: PVC,
        6: PACED,
        7: VBI,
        8: VTR,
    }
    return switch_class.get(i)

def sens_spec(confusion):
    total_sensitivity = 0
    total_specificity = 0
    for i in range(9):
        tp_i = confusion[i][1][1] #true positive for each class
        fn_i = confusion[i][1][0] # false negative for each class
        tn_i = confusion[i][0][0] # true negative for each class
        fp_i = confusion[i][0][1] # false positive for each class
        sensitivity_i = tp_i / ( tp_i + fn_i)  #--------Calculate sensitivity of each class
        total_sensitivity += switcher_clases(i)*sensitivity_i
        specificity_i = tn_i / ( tn_i + fp_i)  #--------Calculate specificity of each class
        total_specificity += switcher_clases(i)*specificity_i

    return total_sensitivity, total_specificity

def confusion_matrix(test_Y, pred_Y):
    #-------MULTICASE CONFUSION MATRIX
    return multilabel_confusion_matrix(test_Y, pred_Y)

def calculation_var(confusion, first):
    #-------Calculation of SENSITIVITY AND SPECIFICITY
    sensitivity = sens_spec(confusion)[0]/TOTAL
    specificity = sens_spec(confusion)[1]/TOTAL

    #-------Calculation of time executed
    final = time.time()
    time_executed = final - first

    print("Sensitivity: ", sensitivity, "\nSpecificity: ", specificity)
    print("\nTime executed: ", time_executed)

def encode_x_y(train_Y, test_Y):
    #Encode to an equal type (needed int/float) before treating the data
    label = LabelEncoder()
    train_Y = label.fit_transform(train_Y)

    #for column_name in range(230):
    #    train_X[:,column_name] = label.fit_transform(train_X[:,column_name].astype(str))
    test_Y = label.fit_transform(test_Y)
    #for column_name in range(230):
    #    test_X[:,column_name] = label.fit_transform(test_X[:,column_name].astype(str))

    #-------CONVERT FROM ENCODE INFORMATION TO ORIGINAL ONE
    keys = label.classes_
    values = label.transform(label.classes_)
    dictionary = dict(zip(keys, values))
    #print(dictionary)

    return train_Y, test_Y

def Adaboost():
    first = time.time()
    train_X, test_X, train_Y, test_Y = get_values_trained()
    train_Y, test_Y = encode_x_y(train_Y, test_Y)
    adaboost = AdaBoostClassifier(n_estimators=40, learning_rate = 0.4)
    #-------FIT
    adaboost.fit(train_X,train_Y)
    pred_Y = adaboost.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME

def RandomForest():
    first = time.time()
    train_X, test_X, train_Y, test_Y = get_values_trained()

    rf = RandomForestClassifier(n_estimators=100, max_depth = 20)
    rf.fit(train_X, train_Y)
    pred_Y = rf.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME

def RotationForest():
    first = time.time()
    train_X, test_X, train_Y, test_Y = get_values_trained()
    train_Y, test_Y = encode_x_y(train_Y, test_Y)

    rtf = RotationForestClassifier(n_estimators = 30, n_features_per_subset = 8)
    rtf.fit(train_X, train_Y)
    pred_Y = rtf.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME

#We want to represent in a plot the sensitivity of AB, MB, RF, RTF, SVM LINEAR,
#SVM SQUARED, SVM RADIAL and in another the specificity
#-----CONSTANTS. AMOUNT OF EACH CLASS
TOTAL = 8843
PAC = 1065
PACED = 318
NSR= 4121
PVC = 1466
VBI = 375
AFIB = 749
VTR = 299
ATR = 178
ABI = 272


Adaboost()
RandomForest()
RotationForest()

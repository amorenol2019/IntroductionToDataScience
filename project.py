import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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

    #print("Sensitivity: ", sensitivity, "\nSpecificity: ", specificity)
    #print("\nTime executed: ", time_executed)
    return sensitivity, specificity

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

    adaboost = AdaBoostClassifier(n_estimators=40, learning_rate = 0.4)
    adaboost.fit(train_X,train_Y)
    pred_Y = adaboost.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)

    return sensitivity, specificity, accuracy

def RandomForest():
    first = time.time()

    rf = RandomForestClassifier(n_estimators=100, max_depth = 20)
    rf.fit(train_X, train_Y)
    pred_Y = rf.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)

    return sensitivity, specificity, accuracy

def RotationForest():
    first = time.time()

    rtf = RotationForestClassifier(n_estimators = 30, n_features_per_subset = 8)
    rtf.fit(train_X, train_Y)
    pred_Y = rtf.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)

    return sensitivity, specificity, accuracy

def SVM_Linear():
    first = time.time()

    linear = LinearSVC()
    linear.fit(train_X, train_Y)
    pred_Y = linear.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)

    return sensitivity, specificity, accuracy

def SVM_Square():
    first = time.time()

    sqr = SVC(kernel = 'poly', C = 0.03)
    sqr.fit(train_X, train_Y)
    pred_Y = sqr.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)

    return sensitivity, specificity, accuracy

def SVM_Radial():
    first = time.time()

    rad = SVC(C = 128, gamma = 0.01)
    rad.fit(train_X, train_Y)
    pred_Y = rad.predict(test_X)

    confusion = confusion_matrix(test_Y, pred_Y)
    sensitivity, specificity = calculation_var(confusion, first) #PRINT SENSITIVITY, SPECIFICITY, TIME
    accuracy = (test_Y == pred_Y).sum() / len(test_Y)
    return sensitivity, specificity, accuracy

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

#------Read data from excel, train it
train_X, test_X, train_Y, test_Y = get_values_trained()
train_Y, test_Y = encode_x_y(train_Y, test_Y)

#----List of sensitivities and specificities for each method
#----------X = methods, Y = sensitivities/specificities
methods = ['Adaboost', 'RF', 'RTF', 'SVM Linear', 'SVM Radial']
sensitivities = [None]
specificities = [None]
error = [None]
sens_ADA, spec_ADA, acc_ADA = Adaboost()
sensitivities[0] = sens_ADA*100
specificities[0] = spec_ADA*100
error[0] = 1 - acc_ADA*100
sens_RF,spec_RF, acc_RF = RandomForest()
sensitivities.append(sens_RF*100)
specificities.append(spec_RF*100)
error.append(1 - acc_RF*100)
sens_RTF, spec_RTF, acc_RTF = RotationForest()
sensitivities.append(sens_RTF*100)
specificities.append(spec_RTF*100)
error.append(1 - acc_RTF*100)
sens_linear, spec_linear, acc_linear = SVM_Linear()
sensitivities.append(sens_linear*100)
specificities.append(spec_linear*100)
error.append(1 - acc_linear*100)
sens_rad, spec_rad, acc_rad = SVM_Radial()
sensitivities.append(sens_rad*100)
specificities.append(spec_rad*100)
error.append(1 - acc_rad*100)

#----TAKES A LONG TIME TO EXECUTE THE FUNCTION, DON'T KNOW THE PROBLEM
#sens_sqr, spec_sqr = SVM_Square()
#sensitivities.append(sens_sqr*100)
#specificities.append(spec_sqr*100)

#------PLOT OF SENSITIVITY
plt.bar(methods, sensitivities)
#plt.errorbar(methods, sensitivities, xerr=error)
plt.ylabel('Sensitivity %')
plt.title('HRV')
plt.show()

#------PLOT OF SPECIFICITY
plt.bar(methods, specificities)
#plt.errorbar(methods, specificities, xerr=error) 
plt.ylabel('Specificity %')
plt.title('HRV')
plt.show()

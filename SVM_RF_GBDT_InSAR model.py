import pandas as pd
from sklearn import metrics
import itertools
import pickle
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif']=['Times New Roman']
np.random.seed(10)
n_estimator = 100

def dataPrepare(filePath,flag=0):
    data = pd.read_csv(filePath)
    colName = ['x1', 'x2', 'x3', 'x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']
    X = data[colName]
    y = data['value']
    GeoID = data['FID']
    if flag==0:
        return X, y, GeoID
    else :
        return X, GeoID

def pre_class(y_probability):
    pred_class = []
    for i in y_probability:
        if i > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)
    return pred_class

# save results
def saveResults(GeoID, y_pred_proba, result_file):
    results = np.vstack((GeoID,y_pred_proba))
    results = np.transpose(results)
    header_string = 'GeoID, y_pred_proba'
    np.savetxt(result_file, results, header = header_string, fmt = '%d,%0.5f',delimiter = ',')
    print('Saving file Done!')

sampleData = r'InSAR_train.csv'
X, y, _= dataPrepare(sampleData,flag=0)
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X, y, test_size=0.3)

model1 = svm.SVC (C=1.0,kernel='rbf',probability=True)
model2 = RandomForestClassifier(max_depth=3, n_estimators=n_estimator, class_weight = 'balanced')#
model3 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=n_estimator)
model1.fit(train_data_x,train_data_y)
model2.fit(train_data_x,train_data_y)
model3.fit(train_data_x,train_data_y)

#SVM
accuracy1 = model1.score(test_data_x,test_data_y)
y_probability1 = model1.predict_proba(test_data_x)[:,1]
test_auc1 = metrics.roc_auc_score(test_data_y,y_probability1)
kappa1 = metrics.cohen_kappa_score(test_data_y, pre_class(y_probability1))
mcc1 = metrics.matthews_corrcoef(test_data_y, pre_class(y_probability1))
f1_1 = metrics.f1_score(test_data_y, pre_class(y_probability1))
print ('accuracy1 = %f' %accuracy1)
print ('AUC1 = %f' %test_auc1)
print('kappa1 = %f' %kappa1)
print('mcc1 = %f' %mcc1)
print('f1_1 = %f' %f1_1)
#RF
accuracy2 = model2.score(test_data_x,test_data_y)
y_probability2 = model2.predict_proba(test_data_x)[:,1]
test_auc2 = metrics.roc_auc_score(test_data_y,y_probability2)
kappa2 = metrics.cohen_kappa_score(test_data_y, pre_class(y_probability2))
mcc2 = metrics.matthews_corrcoef(test_data_y, pre_class(y_probability2))
f1_2 = metrics.f1_score(test_data_y, pre_class(y_probability2))
print ('accuracy2 = %f' %accuracy2)
print ('AUC2 = %f' %test_auc2)
print('kappa2 = %f' %kappa2)
print('mcc2 = %f' %mcc2)
print('f1_2 = %f' %f1_2)
#GBDT
accuracy3 = model3.score(test_data_x,test_data_y)
y_probability3 = model3.predict_proba(test_data_x)[:,1]
test_auc3 = metrics.roc_auc_score(test_data_y,y_probability3)
kappa3 = metrics.cohen_kappa_score(test_data_y, pre_class(y_probability3))
mcc3 = metrics.matthews_corrcoef(test_data_y, pre_class(y_probability3))
f1_3 = metrics.f1_score(test_data_y, pre_class(y_probability3))
print ('accuracy3 = %f' %accuracy3)
print ('AUC3 = %f' %test_auc3)
print('kappa3 = %f' %kappa3)
print('mcc3 = %f' %mcc3)
print('f1_3 = %f' %f1_3)

fpr1, tpr1, thresholds1 = metrics.roc_curve(test_data_y, y_probability1, pos_label=1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(test_data_y, y_probability2, pos_label=1)
fpr3, tpr3, thresholds3 = metrics.roc_curve(test_data_y, y_probability3, pos_label=1)

plt.figure()
plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
plt.axis('square')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.plot(fpr1, tpr1, linestyle='-', label='InSAR+SVM(AUC=%0.3f)' %test_auc1, color='b')
plt.plot(fpr2, tpr2, linestyle='-', label='InSAR+RF(AUC=%0.3f)' %test_auc2, color='g')
plt.plot(fpr3, tpr3, linestyle='-', label='InSAR+GBDT(AUC=%0.3f)' %test_auc3, color='r')
plt.xlabel('False positive rate',fontsize=12)
plt.ylabel('True positive rate',fontsize=12)
plt.text(0.8, 0.4,'(a)',fontsize=14)
plt.legend(loc='lower right',fontsize=10)
plt.show()

#model result
allData = r'Alldata2.csv'
X, GeoID= dataPrepare(allData,flag=1)
y_pred1 = model1.predict(X)
y_pred_proba1 = model1.predict_proba(X)[:,1]
result_file1 = './result.txt'
saveResults(GeoID, y_pred_proba1, result_file1)

y_pred_proba2 = model2.predict_proba(X)[:,1]
result_file2 = './InSAR_RF_result.txt'
saveResults(GeoID, y_pred_proba2, result_file2)

y_pred_proba3 = model3.predict_proba(X)[:,1]
result_file3 = './InSAR_GBDT_result.txt'
saveResults(GeoID, y_pred_proba3, result_file3)
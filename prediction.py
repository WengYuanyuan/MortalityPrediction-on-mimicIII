
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


l_AUC = []
l_accu = []
l_rec = []
l_f1 = []
l_tn, l_fp, l_fn, l_tp = [], [], [], [] 

for i in range(3):   
    # data = pd.read_csv('/Users/quanquan/Documents/Project/mimic-child-mortality/features.txt', header = None)
    data = pd.read_csv('features.txt', header = None)
    
    
    data = data.values
    for i in range(418):
        if type(data[0,i]) == str and i!= 2 and i!=3:
            for j in range(8200):
                data[j,i] = float(data[j,i].replace("[","").replace("(","").replace("]","").replace(")",""))
    for j in range(8200):
        data[j,2] = 0 if data[j,2] == "u'F'" else 1
        data[j,3] = 0 if data[j,3] == ' False' else 1
    
    
    data = data[:,1:] #delete the first column
    
    dead_patients = data[np.where( data[:,2] == 1)]
    
    alive_patients = data[np.where( data[:,2] == 0)]
    np.random.shuffle(alive_patients)
    alive_patients = alive_patients[:73] #randomly select same number of patients as that of dead data.
    
    bal_data = np.concatenate((dead_patients, alive_patients), axis=0) # dataset contains balanced data.
    np.random.shuffle(bal_data)
    
    feature = np.delete(bal_data, 2, 1)
    label = bal_data[:, 2] 
    
    train_rows = int(math.floor(0.8* bal_data.shape[0]))
    # test_rows = bal_data.shape[0] - train_rows
    
    trainX = feature[:train_rows]
    trainY = label[:train_rows]
    trainY = trainY.astype('int')
    testX = feature[train_rows:]
    testY = label[train_rows:] 
    testY = testY.astype('int')
    
    ratio = testY[np.where( testY[:] == 1)].size/float(testY.size)
    print 'dead patients accounts for: ', ratio
    
    
    
    ########## SVM @#############
    clf = svm.SVC(kernel='linear', C=1).fit(trainX, trainY)  
    predY = clf.predict(testX)
    
    # AUC 
    fpr, tpr, thresholds = metrics.roc_curve(testY, predY, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    print "AUC: ", AUC  
    l_AUC.append(AUC)
    
    # accuracy
    accu_score = accuracy_score(testY, predY)
    print "accuracy: ", accu_score  
    l_accu.append(accu_score)
    
    # Recall score
    rec_score = recall_score(testY, predY, average='macro') 
    print "recall score: ", rec_score  
    l_rec.append(rec_score)
    
    # F1 score
    f1 = f1_score(testY, predY, average='macro') 
    print "F1 score: ", f1 
    l_f1.append(f1)
    
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(testY, predY).ravel()
    print "tn, fp, fn, tp: ", tn, fp, fn, tp 
    l_tn.append(tn)
    l_fp.append(fp)
    l_fn.append(fn)
    l_tp.append(tp)

    
print np.mean(l_AUC), np.mean(l_accu), np.mean(l_rec), np.mean(l_f1), np.mean(l_tn), np.mean(l_fp),np.mean(l_fn), np.mean(l_tp)
        
            
                

                    
                                                            

l_AUC = []
l_accu = []
l_rec = []
l_f1 = []
l_tn, l_fp, l_fn, l_tp = [], [], [], []                         
                                
for i in range(3):   
    # data = pd.read_csv('/Users/quanquan/Documents/Project/mimic-child-mortality/features.txt', header = None)
    data = pd.read_csv('features.txt', header = None)
    
    
    data = data.values
    for i in range(418):
        if type(data[0,i]) == str and i!= 2 and i!=3:
            for j in range(8200):
                data[j,i] = float(data[j,i].replace("[","").replace("(","").replace("]","").replace(")",""))
    for j in range(8200):
        data[j,2] = 0 if data[j,2] == "u'F'" else 1
        data[j,3] = 0 if data[j,3] == ' False' else 1
    
    
    data = data[:,1:] #delete the first column
    
    dead_patients = data[np.where( data[:,2] == 1)]
    
    alive_patients = data[np.where( data[:,2] == 0)]
    np.random.shuffle(alive_patients)
    alive_patients = alive_patients[:73] #randomly select same number of patients as that of dead data.
    
    bal_data = np.concatenate((dead_patients, alive_patients), axis=0) # dataset contains balanced data.
    np.random.shuffle(bal_data)
    
    feature = np.delete(bal_data, 2, 1)
    label = bal_data[:, 2] 
    
    train_rows = int(math.floor(0.8* bal_data.shape[0]))
    # test_rows = bal_data.shape[0] - train_rows
    
    trainX = feature[:train_rows]
    trainY = label[:train_rows]
    trainY = trainY.astype('int')
    testX = feature[train_rows:]
    testY = label[train_rows:] 
    testY = testY.astype('int')
    
    ratio = testY[np.where( testY[:] == 1)].size/float(testY.size)
    print 'dead patients accounts for: ', ratio
    
    
    ########## Logistic Regression @#############
    clf = svm.SVC(kernel='linear', C=1).fit(trainX, trainY)  
    logreg = linear_model.LogisticRegression()
    logreg.fit(trainX, trainY)
    predY_LR = logreg.predict(testX)
    
    # AUC 
    fpr, tpr, thresholds = metrics.roc_curve(testY, predY_LR, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    print "AUC: ", AUC  
    l_AUC.append(AUC)

    # accuracy
    accu_score = accuracy_score(testY, predY_LR)
    print "accuracy: ", accu_score  
    l_accu.append(accu_score)

    # Recall score
    testY = testY.astype('int')
    rec_score = recall_score(testY, predY_LR, average='weighted')
    print "recall score: ", rec_score
    l_rec.append(rec_score)

    
    # F1 score
    f1 = f1_score(testY, predY_LR, average='macro') 
    print "F1 score: ", f1 
    l_f1.append(f1)
    
    
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(testY, predY_LR).ravel()
    print "tn, fp, fn, tp: ", tn, fp, fn, tp 
    l_tn.append(tn)
    l_fp.append(fp)
    l_fn.append(fn)
    l_tp.append(tp)    

print np.mean(l_AUC), np.mean(l_accu), np.mean(l_rec), np.mean(l_f1), np.mean(l_tn), np.mean(l_fp),np.mean(l_fn), np.mean(l_tp)

   
      
         
            
                  

l_AUC = []
l_accu = []
l_rec = []
l_f1 = []
l_tn, l_fp, l_fn, l_tp = [], [], [], []  
for i in range(3):   
    # data = pd.read_csv('/Users/quanquan/Documents/Project/mimic-child-mortality/features.txt', header = None)
    data = pd.read_csv('features.txt', header = None)
    
    
    data = data.values
    for i in range(418):
        if type(data[0,i]) == str and i!= 2 and i!=3:
            for j in range(8200):
                data[j,i] = float(data[j,i].replace("[","").replace("(","").replace("]","").replace(")",""))
    for j in range(8200):
        data[j,2] = 0 if data[j,2] == "u'F'" else 1
        data[j,3] = 0 if data[j,3] == ' False' else 1
    
    
    data = data[:,1:] #delete the first column
    
    dead_patients = data[np.where( data[:,2] == 1)]
    
    alive_patients = data[np.where( data[:,2] == 0)]
    np.random.shuffle(alive_patients)
    alive_patients = alive_patients[:73] #randomly select same number of patients as that of dead data.
    
    bal_data = np.concatenate((dead_patients, alive_patients), axis=0) # dataset contains balanced data.
    np.random.shuffle(bal_data)
    
    feature = np.delete(bal_data, 2, 1)
    label = bal_data[:, 2] 
    
    train_rows = int(math.floor(0.8* bal_data.shape[0]))
    # test_rows = bal_data.shape[0] - train_rows
    
    trainX = feature[:train_rows]
    trainY = label[:train_rows]
    trainY = trainY.astype('int')
    testX = feature[train_rows:]
    testY = label[train_rows:] 
    testY = testY.astype('int')
    
    ratio = testY[np.where( testY[:] == 1)].size/float(testY.size)
    print 'dead patients accounts for: ', ratio
       
    ########## Random Forest @#############
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(trainX, trainY)
    predY_RF = clf.predict(testX)
    
    # AUC 
    fpr, tpr, thresholds = metrics.roc_curve(testY, predY_RF, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    print "AUC: ", AUC  
    l_AUC.append(AUC)
   
    # accuracy
    accu_score = accuracy_score(testY, predY_RF)
    print "accuracy: ", accu_score  
    l_accu.append(accu_score)

    # Recall score
    rec_score = recall_score(testY, predY_RF, average='macro') 
    print "recall score: ", rec_score  
    l_rec.append(rec_score)
    
    
    # F1 score
    f1 = f1_score(testY, predY_RF, average='macro') 
    print "F1 score: ", f1 
    l_f1.append(f1)
  
    
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(testY, predY_RF).ravel()
    print "tn, fp, fn, tp: ", tn, fp, fn, tp 
    l_tn.append(tn)
    l_fp.append(fp)
    l_fn.append(fn)
    l_tp.append(tp) 
    
print np.mean(l_AUC), np.mean(l_accu), np.mean(l_rec), np.mean(l_f1), np.mean(l_tn), np.mean(l_fp),np.mean(l_fn), np.mean(l_tp)
    

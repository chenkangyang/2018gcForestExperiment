
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest


# #### Layering: divide the data into N layers, make sure every layer has the same distribution of 0-1

# #### evaluate function

# In[2]:


def evaluate(true, pred):
    # compute accuracy, precision and recall
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(0, len(pred)):
        if pred[i] == true[i] and true[i] == 1:
            TP += 1
        elif pred[i] == true[i] and true[i] == 0:
            TN += 1
        elif pred[i] != true[i] and true[i] == 0:
            FP += 1
        elif pred[i] != true[i] and true[i] == 1:
            FN += 1
    print('TP=',TP,'FP=',FP,'TN=',TN,'FN=',FN)
#     print("F1", f1_score(true, pred))
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    

    F1 = 2*precision*recall / (precision + recall)
    print("precision", precision,"\nrecall", recall,"\naccuracy", accuracy)
    print('F1=',F1)
    return F1, accuracy, precision, recall


# #### Batch
# 
# combine serveral datas‘ features together

# In[3]:


def Batch(X, y, size):
    batch_size = size

    X_trim = X
    y_trim = y

    if len(X) % batch_size != 0:
        extra_num = len(X) % batch_size
        X_trim = np.delete(X, range(len(X) - extra_num, len(X)), axis = 0)
        y_trim = np.delete(y, range(len(y) - extra_num, len(y)), axis = 0)

    X_batch = np.split(X_trim, len(X_trim)/batch_size)
    y_batch = np.split(y_trim, len(y_trim)/batch_size)

    num_batch = 0

    for each_batch in X_batch:
        X_batch[num_batch] = np.reshape(X_batch[num_batch], (9*batch_size))
        y_batch[num_batch] = y_batch[num_batch][-1]
        num_batch += 1

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


# ### clean the data before somte
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[4]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])


# #### gc_config

# In[5]:


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 10
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


# ### HyperParameter

# In[6]:


batch_size = 3
kf = 100 # N-layers
test_size = 0.33
random_seed = 42


# # GcForest
# 
# ## test gc

# #### test GcForest on valid datasets

# # load 2017 Test datasets

# In[7]:


lines = open("../data/water/txt/2017waterDataTesting.txt").readlines()
num_lines = len(lines) - 1

X_test = np.ones((num_lines, 9))
y_test = np.ones((num_lines, 1))
flag = 0

lines = np.delete(lines, 0, axis = 0)
i = 0

for line in lines:
    data_line = line.split()
    feature = data_line[3:12]
    for k in range(9):
        if feature[k] == 'NA':
            flag = 1
            break
    if flag == 1:
        flag = 0
        continue    # jump out of the loop
    X_test[i] = feature    
    if data_line[12] == 'FALSE':
        y_test[i] = 0
    elif data_line[12] == 'TRUE':
        y_test[i] = 1
    i += 1


X_test = clean_pipeline.fit_transform(X_test) 


# #### 1. test gcForest on 2017 Test datasets

# In[8]:


with open("../pkl/2017_test.pkl", "rb") as f:
    gc = pickle.load(f)
    y_test_pred = gc.predict(X_test)
    print("============= 2017 datasets' results on test =============")
    evaluate(y_test, y_test_pred)


# #### 2. test GcForest on 2017 batched Test datasets

# In[9]:


X_test_batch, y_test_batch = Batch(X_test, y_test, batch_size)
    
with open("../pkl/2017_test_batch.pkl", "rb") as f:
    gc = pickle.load(f)
    y_test_pred = gc.predict(X_test_batch)
    print("============= 2017 datasets' results on %d batched test =============" %(batch_size))
    evaluate(y_test_batch, y_test_pred)


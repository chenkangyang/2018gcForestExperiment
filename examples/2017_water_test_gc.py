
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


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

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    
    print('TP=',TP,'FP=',FP,'TN=',TN,'FN=',FN)
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


# ### HyperParameter

# In[4]:


batch_size = 3
test_size = 0.33
random_seed = 42


# ### clean the data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[5]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])


# # GcForest
# 
# ## test gc

# # load 2017 Test datasets

# In[6]:


test = pd.read_csv("../data/water/csv/test2017.csv")

X_test = test.values[:, 0:-1]
y_test = test.values[:, -1]

X_test = clean_pipeline.fit_transform(X_test)


# #### 1. test gcForest on 2018 Test datasets

# In[7]:


with open("../pkl/2018_gc.pkl", "rb") as f:
    gc = pickle.load(f)
    y_test_pred = gc.predict(X_test)
    print("============= 2017 datasets' results on test =============")
    evaluate(y_test, y_test_pred)



# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


# #### somte sampling

# In[2]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        # random_lst = list(np.random.randint(0, 1000, 4))
        sm = SMOTE(random_state=random_seed)
    elif is_random == False:
        sm = SMOTE(random_state=0)

    # sm = SMOTE(random_state=random_lst[2])
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


# #### evaluate function

# In[3]:


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

# In[4]:


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


# #### gc_config

# In[5]:


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = random_seed
    ca_config["max_layers"] = 10
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, 
                                    "type": "XGBClassifier", 
                                    "n_estimators": 1000, 
                                    "learning_rate": 0.1,
                                    "gamma":0,
                                    "subsample":0.8,
                                    "colsample_bytree":0.8,
                                    "objective":'binary:logistic',
                                    "scale_pos_weight":1,
                                    "seed":random_seed,
                                    "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, 
#                                     "type": "XGBClassifier", 
#                                     "n_estimators": 1000, 
#                                     "max_depth": 4, 
#                                     "learning_rate": 0.1,
#                                     "gamma":0,
#                                     "subsample":0.8,
#                                     "colsample_bytree":0.8,
#                                     "objective":'binary:logistic',
#                                     "scale_pos_weight":1,
#                                     "seed":random_seed,
#                                     "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


# ### HyperParameter

# In[6]:


batch_size = 3
test_size = 0.33
random_seed = 42
cv = 5


# # load train

# In[15]:


data_all = pd.read_csv("../data/water/csv/train2018.csv")

X_train = data_all.values[:, 0:-1]
y_train = data_all.values[:, -1]


# #### train_valid_split

# In[16]:


# print("============ train_valid_split ============")
# X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=test_size, stratify=y, random_state=random_seed)
# print("train: %d, valid: %d" %(X_train.shape[0], X_valid.shape[0]))


# ### clean the data before somte
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[17]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
# X_valid = clean_pipeline.fit_transform(X_valid)


# #### Do somte sampling on the train data to solve data imblance problem

# In[18]:


# X_train_oversampled, y_train_oversampled = Smoter(X_train, y_train, is_random=True)
# print("============ SMOTE ============")
# print("train: %d, contains %.4f of 0 , after SMOTE: train: %d contains %.4f of 1" %(X_train.shape[0], (y_train == 0).sum()/y_train.shape[0], X_train_oversampled.shape[0], (y_train_oversampled == 0).sum()/y_train_oversampled.shape[0]))


# In[19]:


# X_train_oversampled_batch, y_train_oversampled_batch = Batch(X_train_oversampled, y_train_oversampled, batch_size)
# X_train_batch, y_train_batch = Batch(X_train, y_train, batch_size)
# X_valid_batch, y_valid_batch = Batch(X_valid, y_valid, batch_size)


# # GcForest
# 
# ## test gc

# # load 2018 Test datasets

# In[20]:


test = pd.read_csv("../data/water/csv/test2018.csv")

X_test = test.values[:, 0:-1]
y_test = test.values[:, -1]

X_test = clean_pipeline.fit_transform(X_test)


# In[21]:


# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify = y, random_state = random_seed)

# X_train_oversampled, y_train_oversampled = Smoter(X_train, y_train, is_random=True)
config = get_toy_config()
gc = GCForest(config)

gc.fit_transform(X_train, y_train, X_test, y_test)
# y_valid_pred = gc.predict(X_valid)


# In[22]:


# dump
with open("../pkl/2018_gc.pkl", "wb") as f:
    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    
# # load
# with open("../pkl/2018_gc.pkl", "rb") as f:
#     gc = pickle.load(f)


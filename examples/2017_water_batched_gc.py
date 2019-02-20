
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


# #### Layering: divide the data into N layers, make sure every layer has the same distribution of 0-1

# In[2]:


def Layering(df, N):
    new_data=df.iloc[:,0:]

    data_maj = new_data[new_data['EVENT']==0]
    data_min = new_data[new_data['EVENT']==1]
    n_maj=data_maj.iloc[:,0].size
    n_min=data_min.iloc[:,0].size
    M1=n_maj%N
    M2=n_min%N
    stepD=int(n_maj/10)
    stepS=int(n_min/10)

    maj_data = []
    for i in range(N):
        maj_data.append(data_maj.iloc[i*stepD:(i+1)*stepD])
    for i in range(M1):
        maj_data[i]=maj_data[i].append(data_maj.iloc[stepD*N+i:stepD*N+i+1])


    min_data = []
    for i in range(N):
        min_data.append(data_min.iloc[i*stepS:(i+1)*stepS])
    for i in range(M2):
        min_data[i]=min_data[i].append(data_min.iloc[stepS*N+i:stepS*N+i+1])

    Last_Data = pd.DataFrame()
    for i in range(N):
        Last_Data=Last_Data.append(maj_data[i].append(min_data[i]))
    return Last_Data


# #### somte sampling

# In[3]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        random_lst = list(np.random.randint(0, 1000, 4))
    elif is_random == False:
        random_lst = [0] * 4

    print("rs:", random_lst)
    sm = SMOTE(random_state=random_lst[2], kind = 0.24)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


# #### evaluate function

# In[4]:


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

# In[5]:


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

# In[6]:


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

# In[7]:


batch_size = 3
kf = 100
valid_size = 0.33
random_seed = 42


# # load 2017 Test datasets

# In[8]:


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


# # load 2017 train

# In[9]:


df = pd.read_table('../data/water/txt/2018waterDataTraining.txt',delim_whitespace=True)


# In[10]:


df = df.reset_index()
Time = np.zeros(df.shape[0]).astype("str")
for i in range(len(df)):
    Time[i] = df['index'][i]+" "+ df['Time'][i]
df['Time'] = Time
df = df.drop(['index'], axis=1)


# ## feature engineering on train data

# In[11]:


drop_columns = ['Time']
continuous_features = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
cat_features =[]


# In[12]:


all_df_dummies = pd.get_dummies(df, columns=cat_features)


# In[13]:


all_df_dummies.drop(drop_columns, 1, inplace=True)
# delte NA datas
all_df_dummies = all_df_dummies.dropna(axis=0)


# In[14]:


X = all_df_dummies.drop(['EVENT'], axis=1) # Series
y = all_df_dummies['EVENT'].apply(lambda x: 0 if x == False else 1) # Series


# In[15]:


data_all = pd.concat([X,y], axis=1)


# In[16]:


data_all.head()


# ### layer sampling on train

# In[17]:


print("============ layer sampling ============")
data_layer = Layering(data_all, kf)
array = data_layer.values
X = array[:, 0:-1] # ndarray
y = array[:, -1] # ndarray


# ### train_valid_split

# In[18]:


print("============ train_valid_split ============")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, 
                                       stratify = y, random_state = random_seed)
print("train: %d, valid: %d" %(X_train.shape[0], X_valid.shape[0]))


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[19]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_valid = clean_pipeline.fit_transform(X_valid)
X_test = clean_pipeline.fit_transform(X_test)


# #### Do somte sampling on the train data to solve data imblance problem

# In[20]:


X_train_oversampled, y_train_oversampled = X_train, y_train
# X_train_oversampled, y_train_oversampled = Smoter(X_train, y_train, is_random=True)
# print("============ SMOTE =Smoter===========")
# print("train: %d, contains %.4f of 0 , after SMOTE: train: %d contains %.4f of 1" %(X_train.shape[0], (y_train == 0).sum()/y_train.shape[0], X_train_oversampled.shape[0], (y_train_oversampled == 0).sum()/y_train_oversampled.shape[0]))


# In[21]:


X_train_oversampled_batch, y_train_oversampled_batch = Batch(X_train_oversampled, y_train_oversampled, batch_size)
X_train_batch, y_train_batch = Batch(X_train, y_train, batch_size)
X_valid_batch, y_valid_batch = Batch(X_valid, y_valid, batch_size)
X_test_batch, y_test_batch = Batch(X_test, y_test, batch_size)


# # GcForest
# 
# ## train gc

# #### 1.train GcForest on oversampled datasets

# In[22]:


config = get_toy_config()
gc = GCForest(config)

X_train_enc = gc.fit_transform(X_train_oversampled, y_train_oversampled)


# dump
with open("../pkl/2017_test.pkl", "wb") as f:
    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
# load
with open("../pkl/2017_test.pkl", "rb") as f:
    gc = pickle.load(f)


# #### test GcForest on valid datasets

# In[23]:


y_valid_pred = gc.predict(X_valid)
y_valid_nonezero = np.count_nonzero(y_valid)
y_valid_pred_nonezero = np.count_nonzero(y_valid_pred)

print("y_valid, 1 contains: ", y_valid_nonezero/len(y_valid))
print("y_valid_pred, 1 contains: ", y_valid_pred_nonezero/len(y_valid_pred))

print("============= 2017 datasets' results on valid =============")
gc_f1, gc_accraucy, gc_precision, gc_recall = evaluate(y_valid, y_valid_pred)


# #### test gcForest on 2017 Test datasets

# In[25]:


y_test_pred = gc.predict(X_test)

y_test_pred_nonezero = np.count_nonzero(y_test_pred)
print("y_test_pred: {:d}, 1 contains: {:6f}".format(len(y_test_pred), y_test_pred_nonezero/len(y_test_pred)))


print("============= 2017 datasets' results on test =============")
gc_f1, gc_accraucy, gc_precision, gc_recall = evaluate(y_test, y_test_pred)


# #### 2. train GcForest on batched datasets

# In[26]:


X_train_batch_enc = gc.fit_transform(X_train_batch, y_train_batch)

# dump
with open("../pkl/2017_test_batch.pkl", "wb") as f:
    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
# load
with open("../pkl/2017_test_batch.pkl", "rb") as f:
    gc = pickle.load(f)


# #### test GcForest on batched valid datasets

# In[27]:


y_valid_batch_pred = gc.predict(X_valid_batch)    
y_valid_batch_nonezero = np.count_nonzero(y_valid_batch)
y_valid_batch_pred_nonezero = np.count_nonzero(y_valid_batch_pred)

print("y_valid_batch, 1 contains: ", y_valid_batch_nonezero/len(y_valid_batch))
print("y_valid_batch_pred, 1 contains: ", y_valid_batch_pred_nonezero/len(y_valid_batch_pred))



print("============= 2017 datasets' results on %d batched valid =============" %(batch_size))
gc_f1, gc_accraucy, gc_precision, gc_recall = evaluate(y_valid_batch, y_valid_batch_pred)


# #### test gcForest on 2017 batched Test datasets

# In[28]:


y_test_batch_pred = gc.predict(X_test_batch)

y_test_batch_pred_nonezero = np.count_nonzero(y_test_batch_pred)
print("y_test_pred: {:d}, 1 contains: {:6f}".format(len(y_test_batch_pred), y_test_pred_nonezero/len(y_test_batch_pred)))

print("============= 2017 datasets' results on %d batched test =============" %(batch_size))
gc_f1, gc_accraucy, gc_precision, gc_recall = evaluate(y_test_batch, y_test_batch_pred)


# In[ ]:





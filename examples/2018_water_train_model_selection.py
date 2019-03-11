
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# #### somte sampling

# In[2]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        sm = SMOTE(random_state=random_seed)
    elif is_random == False:
        sm = SMOTE(random_state=0)
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


# ### HyperParameter

# In[4]:


batch_size = 3
test_size = 0.33
random_seed = 42
cv = 5


# # Load 2018 Train Set

# In[5]:


data_all = pd.read_csv("../data/water/csv/train2018.csv")

X = data_all.values[:, 0:-1]
y = data_all.values[:, -1]


# ### clean the data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[14]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                               ('std_scaler', preprocessing.StandardScaler()),])


# # Load 2018 Test Set

# In[15]:


test = pd.read_csv("../data/water/csv/test2018.csv")

X_test = test.values[:, 0:-1]
y_test = test.values[:, -1]

X_test = clean_pipeline.fit_transform(X_test)


# In[16]:


models = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    LinearSVC(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
]


scoring = ['accuracy', 'precision', 'recall', 'f1']

entries = []
columns = []

for model in models:
    values = []
    columns = []
    model_name = model.__class__.__name__
    scores = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True, n_jobs=-1)
    columns.append("model_name")
    for key in scores.keys():
        columns.append(key)

    values.append(model_name)
    for value in scores.values():
        values.append(value.mean())
    
    entries.append(values)
models_df = pd.DataFrame(entries, columns=columns)


# In[17]:


print("results on train and valid")
print(models_df)


# In[18]:


models_df


# In[19]:


models = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    LinearSVC(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
]

entries = []
for model in models:
    model_name = model.__class__.__name__
    acc_valid = []
    precision_valid = []
    recall_valid = []
    f1_valid = []
    acc_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    skf = StratifiedKFold(n_splits=cv, random_state=random_seed)
    skf.get_n_splits(X, y)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        X_train = clean_pipeline.fit_transform(X_train)
        X_valid = clean_pipeline.fit_transform(X_valid)
        X_train_oversampled, y_train_oversampled = Smoter(X_train, y_train, is_random=True)
        # print("============ SMOTE ============")
        #cprint("train: %d, contains %.4f of 0 , after SMOTE: train: %d contains %.4f of 1" %(X_train.shape[0], (y_train == 0).sum()/y_train.shape[0], X_train_oversampled.shape[0], (y_train_oversampled == 0).sum()/y_train_oversampled.shape[0]))
        model.fit(X_train_oversampled, y_train_oversampled)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)
        acc_valid.append(accuracy_score(y_valid, y_valid_pred))
        precision_valid.append(precision_score(y_valid, y_valid_pred))
        recall_valid.append(recall_score(y_valid, y_valid_pred))
        f1_valid.append(f1_score(y_valid, y_valid_pred))
        acc_test.append(accuracy_score(y_test, y_test_pred))
        precision_test.append(precision_score(y_test, y_test_pred))
        recall_test.append(recall_score(y_test, y_test_pred))
        f1_test.append(f1_score(y_test, y_test_pred))
    entries.append((model_name, np.mean(acc_valid), np.mean(precision_valid), np.mean(recall_valid), np.mean(f1_valid), np.mean(acc_test), np.mean(precision_test), np.mean(recall_test), np.mean(f1_test)))
models_df = pd.DataFrame(entries, columns=['model_name', 'valid_accuracy', 'valid_precision', 'valid_recall', 'valid_f1 score', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1 score'])


# In[20]:


print("After SMOTE, results on valid and test")


# In[21]:


models_df


# ### train set

# In[25]:


# matplotlib.use('Agg')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

cols = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
plt.figure(figsize=(10,6)) 
plt.subplot(3,3,1)
plt.plot(X_train[:,0], label = 'Tp')
plt.ylabel('Tp')
plt.legend(frameon=False)

plt.subplot(3,3,2)
plt.plot(X_train[:,1], label = 'Cl')
plt.ylabel('Cl')
plt.legend(frameon=False)

plt.subplot(3,3,3)
plt.plot(X_train[:,2], label = 'pH')
plt.ylabel('pH')
plt.legend(frameon=False)

plt.subplot(3,3,4)
plt.plot(X_train[:,3], label = 'Redox')
plt.ylabel('Redox')
plt.legend(frameon=False)

plt.subplot(3,3,5)
plt.plot(X_train[:,4], label = 'Leit')
plt.ylabel('Leit')
plt.legend(frameon=False)

plt.subplot(3,3,6)
plt.plot(X_train[:,5], label = 'Trueb')
plt.ylabel('Trueb')
plt.legend(frameon=False)

plt.subplot(3,3,7)
plt.plot(X_train[:,6], label = 'Cl_2')
plt.ylabel('Cl_2')
plt.legend(frameon=False)

plt.subplot(3,3,8)
plt.plot(X_train[:,7], label = 'Fm')
plt.ylabel('Fm')
plt.legend(frameon=False)

plt.subplot(3,3,9)
plt.plot(X_train[:,8], label = 'Fm_2')
plt.ylabel('Fm_2')
plt.legend(frameon=False)


# plt.savefig('../paper/img/before_z_score.eps',format='eps')


# ### test set

# In[26]:


plt.figure(figsize=(10,6)) 
plt.subplot(3,3,1)
plt.plot(X_test[:,0], label = 'Tp')
plt.ylabel('Tp')
plt.legend(frameon=False)

plt.subplot(3,3,2)
plt.plot(X_test[:,1], label = 'Cl')
plt.ylabel('Cl')
plt.legend(frameon=False)

plt.subplot(3,3,3)
plt.plot(X_test[:,2], label = 'pH')
plt.ylabel('pH')
plt.legend(frameon=False)

plt.subplot(3,3,4)
plt.plot(X_test[:,3], label = 'Redox')
plt.ylabel('Redox')
plt.legend(frameon=False)

plt.subplot(3,3,5)
plt.plot(X_test[:,4], label = 'Leit')
plt.ylabel('Leit')
plt.legend(frameon=False)

plt.subplot(3,3,6)
plt.plot(X_test[:,5], label = 'Trueb')
plt.ylabel('Trueb')
plt.legend(frameon=False)

plt.subplot(3,3,7)
plt.plot(X_test[:,6], label = 'Cl_2')
plt.ylabel('Cl_2')
plt.legend(frameon=False)

plt.subplot(3,3,8)
plt.plot(X_test[:,7], label = 'Fm')
plt.ylabel('Fm')
plt.legend(frameon=False)

plt.subplot(3,3,9)
plt.plot(X_test[:,8], label = 'Fm_2')
plt.ylabel('Fm_2')
plt.legend(frameon=False)


# plt.savefig('../paper/img/before_z_score.eps',format='eps')


# ### valid set

# In[27]:


plt.figure(figsize=(10,6)) 
plt.subplot(3,3,1)
plt.plot(X_valid[:,0], label = 'Tp')
plt.ylabel('Tp')
plt.legend(frameon=False)

plt.subplot(3,3,2)
plt.plot(X_valid[:,1], label = 'Cl')
plt.ylabel('Cl')
plt.legend(frameon=False)

plt.subplot(3,3,3)
plt.plot(X_valid[:,2], label = 'pH')
plt.ylabel('pH')
plt.legend(frameon=False)

plt.subplot(3,3,4)
plt.plot(X_valid[:,3], label = 'Redox')
plt.ylabel('Redox')
plt.legend(frameon=False)

plt.subplot(3,3,5)
plt.plot(X_valid[:,4], label = 'Leit')
plt.ylabel('Leit')
plt.legend(frameon=False)

plt.subplot(3,3,6)
plt.plot(X_valid[:,5], label = 'Trueb')
plt.ylabel('Trueb')
plt.legend(frameon=False)

plt.subplot(3,3,7)
plt.plot(X_valid[:,6], label = 'Cl_2')
plt.ylabel('Cl_2')
plt.legend(frameon=False)

plt.subplot(3,3,8)
plt.plot(X_valid[:,7], label = 'Fm')
plt.ylabel('Fm')
plt.legend(frameon=False)

plt.subplot(3,3,9)
plt.plot(X_valid[:,8], label = 'Fm_2')
plt.ylabel('Fm_2')
plt.legend(frameon=False)


# plt.savefig('../paper/img/before_z_score.eps',format='eps')


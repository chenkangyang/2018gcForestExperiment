
# coding: utf-8

# In[34]:


import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# #### somte sampling

# In[35]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        sm = SMOTE(random_state=random_seed)
    elif is_random == False:
        sm = SMOTE(random_state=0)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


# #### evaluate function

# In[36]:


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

# In[37]:


batch_size = 3
test_size = 0.33
random_seed = 42
cv = 5


# # Load 2018 Train Set

# In[38]:


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


print("样本不平衡 %f" %(np.sum(y==1)/len(y)))


# In[57]:


# model = XGBClassifier(learning_rate=0.1,
#                       n_estimators=100,         # 树的个数--1000棵树建立xgboost
#                       max_depth=6,               # 树的深度
#                       min_child_weight = 1,      # 叶子节点最小权重
#                       gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#                       subsample=0.8,             # 随机选择80%样本建立决策树
#                       colsample_btree=0.8,       # 随机选择80%特征建立决策树
#                       objective='binary:logistic', # 指定损失函数
#                       scale_pos_weight=90,        # 解决样本个数不平衡的问题
#                       random_state=random_seed            # 随机数
#                       )

model = XGBClassifier(
learning_rate =0.1,
n_estimators=1000,
max_depth=5,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
scale_pos_weight=1,
seed=random_seed)

xgb_acc_valid = []
xgb_precision_valid = []
xgb_recall_valid = []
xgb_f1_valid = []
entries = []

skf = StratifiedKFold(n_splits=cv, random_state=random_seed)
skf.get_n_splits(X, y)
for train_index, valid_index in skf.split(X, y):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    X_train = clean_pipeline.fit_transform(X_train)
    X_valid = clean_pipeline.fit_transform(X_valid)
    # X_train_smote, y_train_smote = Smoter(X_train, y_train, is_random = True)
    model.fit(X_train,
              y_train,
              eval_set = [(X_valid, y_valid)],
              early_stopping_rounds = 50,
              )
    y_valid_pred = model.predict(X_valid)
    xgb_acc_valid.append(accuracy_score(y_valid, y_valid_pred))
    xgb_precision_valid.append(precision_score(y_valid, y_valid_pred))
    xgb_recall_valid.append(recall_score(y_valid, y_valid_pred))
    xgb_f1_valid.append(f1_score(y_valid, y_valid_pred))
entries.append((np.mean(xgb_acc_valid), np.mean(xgb_precision_valid), np.mean(xgb_recall_valid), np.mean(xgb_f1_valid)))
xgb_df = pd.DataFrame(entries, columns=['valid_accuracy', 'valid_precision', 'valid_recall', 'valid_f1'])


# In[56]:


xgb_df


# In[30]:


param_test1 = {
    'max_depth':np.arange(2,10,2),
    'min_child_weight':np.arange(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                  n_estimators=1000, 
                                                  gamma=0, 
                                                  subsample=0.8, 
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', 
                                                  scale_pos_weight=1, 
                                                  seed=random_seed), 
                                                 param_grid = param_test1, scoring="f1", n_jobs=-1, cv=cv)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify = y, random_state = random_seed)
gsearch1.fit(X_train, y_train)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[32]:


best_clf = gsearch1.best_estimator_
best_clf.fit(X_train, y_train)

y_valid_pred = best_clf.predict(X_valid)

print("Valid f1: %f" %(f1_score(y_valid, y_valid_pred)))


# In[39]:


print(gsearch1.best_estimator_)


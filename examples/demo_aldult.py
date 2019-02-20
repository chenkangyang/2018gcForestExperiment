
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import argparse

print('Loading data...')
names = ['age', 'workclass', 'fnlwgt', 'education', 'educationnum', 'maritalstatus', 'occupation', 'relationship', 'race',
        'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'nativecountry', 'label']
train_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                      header=None, names=names)
test_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                      header=None, names=names, skiprows=[0])
all_df = pd.concat([train_df, test_df])

all_df.head()

# all_df.to_csv('../data/adult/all_df.csv', encoding='utf-8', index=False)
# train_df.to_csv('../data/adult/train.csv', encoding='utf-8', index=False)
# test_df.to_csv('../data/adult/test.csv', encoding='utf-8', index=False)


print('Data loaded!')
print(all_df.shape)
# # 特征工程
# It looks like we have 14 columns to help us predict our classification. We will drop fnlwgt and education and then convert our categorical features to dummy variables. We will also convert our label to 0 and 1 where 1 means the person made more than $50k

drop_columns = ['fnlwgt', 'education']
continuous_features = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
cat_features =['educationnum', 'workclass', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'nativecountry']

all_df_dummies = pd.get_dummies(all_df, columns=cat_features)

all_df_dummies.drop(drop_columns, 1, inplace=True)

y = all_df_dummies['label'].apply(lambda x: 0 if '<' in x else 1)
X = all_df_dummies.drop(['label'], axis=1)

y.value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('train:', X_train.shape)
print('test:', X_test.shape)

# ## 数据清洗

# 中位数填充特征值后，将数据标准化
clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(strategy="median")),
                          ('std_scaler', preprocessing.StandardScaler()),])

X_train_clean = clean_pipeline.fit_transform(X_train)
X_test_clean = clean_pipeline.transform(X_test)


# 评估函数

def evaluate(true, pred):
    f1 = metrics.f1_score(true, pred)
    roc_auc = metrics.roc_auc_score(true, pred)
    accuracy = metrics.accuracy_score(true, pred)
    print("F1: {0}\nROC_AUC: {1}\nACCURACY: {2}".format(f1, roc_auc, accuracy))
    return f1, roc_auc, accuracy


# ## Logistic Regression
# 
# The first model up is a simple logistic regression with the default hyperparameters.

clf = LogisticRegression()
clf.fit(X_train, y_train)
lr_predictions = clf.predict(X_test)
print("Logistic regression with the default hyperparameters")
lr_f1, lr_roc_auc, lr_acc = evaluate(y_test, lr_predictions)


# ## GcForest
# 
# The second model up is a gcforest with our hyperparameters.

# 
# If you wish to use Cascade Layer only, the legal data type for X_train, X_test can be:
# 
#     2-D numpy array of shape (n_sampels, n_features).
#     3-D or 4-D numpy array are also acceptable. For example, passing X_train of shape (60000, 28, 28) or (60000,3,28,28) will be automatically be reshape into (60000, 784)/(60000,2352).
#

import sys 
sys.path.append("..") 
from gcforest.gcforest import GCForest

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 10
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
#     ca_config["estimators"].append(
#             {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
#              "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

config = get_toy_config()
gc = GCForest(config)

# If the model you use cost too much memory for you.
# You can use these methods to force gcforest not keeping model in memory
# gc.set_keep_model_in_mem(False), default is TRUE.

X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)
print("gcForest with our hyperparameters")
gc_f1, gc_roc_auc, gc_acc = evaluate(y_test, y_pred)
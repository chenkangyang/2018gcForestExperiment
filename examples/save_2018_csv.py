
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# #### Layering: divide the data into N layers, make sure every layer has the same distribution of 0-1

# ## load 2017 train

# In[2]:


df = pd.read_table('../data/water/txt/2018waterDataTraining.txt',delim_whitespace=True)


# In[3]:


df = df.reset_index()
Time = np.zeros(df.shape[0]).astype("str")
for i in range(len(df)):
    Time[i] = df['index'][i]+" "+ df['Time'][i]
df['Time'] = Time
df = df.drop(['index'], axis=1)


# ## feature engineering
# 
# 
# It looks like we have 14 columns to help us predict our classification. We will drop fnlwgt and education and then convert our categorical features to dummy variables. We will also convert our label to 0 and 1 where 1 means the person made more than $50k
# 
# 

# In[4]:


drop_columns = ['Time']
continuous_features = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
cat_features =[]


# In[5]:


all_df_dummies = pd.get_dummies(df, columns=cat_features)


# In[6]:


all_df_dummies.drop(drop_columns, 1, inplace=True)
# delte NA datas
all_df_dummies = all_df_dummies.dropna(axis=0)


# In[7]:


X_train = all_df_dummies.drop(['EVENT'], axis=1) # Series
y_train = all_df_dummies['EVENT'].apply(lambda x: 0 if x == False else 1) # Series


# In[8]:


train = pd.concat([X_train,y_train], axis=1)


# In[9]:


train.head()


# In[10]:


train.to_csv('../data/water/csv/train2018.csv', encoding='utf-8', index=False)


# #### layer sampling

# In[11]:


# print("============ layer sampling ============")
# train_layer = Layering(train, kf)
# array = train_layer.values
# X_train = array[:, 0:-1] # ndarray
# y_train = array[:, -1] # ndarray


# Train/Validation Split

# In[12]:


# X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, y_train, test_size=valid_size,
#                                                 stratify = y_train, random_state = random_seed)
# # stratify： 按正负样本原始比例random_seed分配给train 和 valid


# #### Do somte sampling on the train data to solve data imblance problem

# In[13]:


# X_train_oversampled, y_train_oversampled = Smoter(X_tr, y_tr, is_random=True)
# print("============ SMOTE ============")
# print("train: %d, contains %.4f of 0 , after SMOTE: train: %d contains %.4f of 1" %(X_train.shape[0], (y_train == 0).sum()/y_train.shape[0], X_train_oversampled.shape[0], (y_train_oversampled == 0).sum()/y_train_oversampled.shape[0]))


# ### normalize the train and valid
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[14]:


# clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
#                            ('std_scaler', preprocessing.StandardScaler()),])
# X_train_oversampled = clean_pipeline.fit_transform(X_train_oversampled)
# X_vld = clean_pipeline.fit_transform(X_vld)


# ### transfer y into probability vector

# In[15]:


# y_train_oversampled_pro = np.zeros([y_train_oversampled.shape[0], 2])
# for i in range(len(y_train_oversampled)):
#     if y_train_oversampled[i] == 1:
#         y_train_oversampled_pro[i] = np.array([0, 1])
#     else:
#         y_train_oversampled_pro[i] = np.array([1, 0])
# y_train_oversampled = y_train_oversampled_pro    

# y_vld_pro = np.zeros([y_vld.shape[0], 2])
# for i in range(len(y_vld)):
#     if y_vld[i] == 1:
#         y_vld_pro[i] = np.array([0, 1])
#     else:
#         y_vld_pro[i] = np.array([1, 0])
# y_vld = y_vld_pro


# ## load 2017 test

# In[16]:


lines = open("../data/water/txt/2018waterDataTesting.txt").readlines()
num_lines = len(lines) - 1

X_test = np.ones((num_lines, 9))
y_test = np.ones((num_lines, 1))
flag = 0

lines = np.delete(lines, 0, axis = 0)
i = 0

for line in lines:
    data_line = line.split()
    feature = data_line[2:11]
    for k in range(9):
        if feature[k] == 'NA':
            flag = 1
            break
    if flag == 1:
        flag = 0
        continue    # jump out of the loop
    X_test[i] = feature    
    if data_line[11] == 'FALSE':
        y_test[i] = 0
    elif data_line[11] == 'TRUE':
        y_test[i] = 1
    i += 1


# X_test = clean_pipeline.fit_transform(X_test) 

test = np.concatenate([X_test, y_test], axis=1)

# y_test_pro = np.zeros([y_test.shape[0], 2])
# for i in range(len(y_test)):
#     if y_test[i] == 1:
#         y_test_pro[i] = np.array([0, 1])
#     else:
#         y_test_pro[i] = np.array([1, 0])
# y_test = y_test_pro


# In[17]:


test = pd.DataFrame(test, columns =['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2', 'EVENT'])
test.head()


# In[18]:


test.to_csv('../data/water/csv/test2018.csv', encoding='utf-8', index=False)


# In[ ]:





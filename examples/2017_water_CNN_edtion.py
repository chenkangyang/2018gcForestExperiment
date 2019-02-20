
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# get_ipython().run_line_magic('matplotlib', 'inline')


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

    sm = SMOTE(random_state=random_lst[2])
    X_smote, y_smote = sm.fit_sample(X, y)
    y_smote = y_smote[:,np.newaxis]
    return X_smote, y_smote

def OverSampler(X, y, is_random=False):
    if is_random == True:
        random_lst = list(np.random.randint(0, 1000, 4))
    elif is_random == False:
        random_lst = [0] * 4

    ros = RandomOverSampler(random_state=random_lst[0])
    X_oversampled, y_oversampled = ros.fit_sample(X, y)

    return X_oversampled, y_oversampled

# #### evaluate function

# In[4]:


def evaluate(v_xs, v_ys, sess):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob:1}) # y_pre 是一个 n_samples*2 的概率向量
    
    predicted = np.argmax(y_pre, 1)
    actual = np.argmax(v_ys, 1)
    
    # print("predicted", predicted)
    # print("actual", actual)
    
    # Count true positives, true negatives, false positives and false negatives.
    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)

    # print('TP=',tp,'FP=',fp,'TN=',tn,'FN=',fn)
    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    # print('Precision = ', precision)
    # print('Recall = ', recall)
    # print('F1 Score = ', f1_score)
    # print('Accuracy = ', accuracy)
    return precision, recall, f1_score, accuracy


# #### Batch

# In[5]:


def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]


# In[6]:


def weight_variable(shape,name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)


# In[7]:


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)


# In[8]:


def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=stride_num, padding='SAME')


# ### HyperParameter

# In[9]:


kp = 0.15
stride_num = 1
pool_patch_size = 2
kf = 100
lr = 0.01
max_iterations = 3000
hidden_cell_num = 8
random_seed = 42
valid_size = 0.33


# ## load 2017 train

# In[10]:


df = pd.read_table('../data/water/txt/2017waterDataTraining.txt',delim_whitespace=True)


# In[11]:


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

# In[12]:


drop_columns = ['Time']
continuous_features = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
cat_features =[]


# In[13]:


all_df_dummies = pd.get_dummies(df, columns=cat_features)


# In[14]:


all_df_dummies.drop(drop_columns, 1, inplace=True)
# delte NA datas
all_df_dummies = all_df_dummies.dropna(axis=0)


# In[15]:


X_train = all_df_dummies.drop(['EVENT'], axis=1) # Series
y_train = all_df_dummies['EVENT'].apply(lambda x: 0 if x == False else 1) # Series


# In[16]:


train = pd.concat([X_train,y_train], axis=1)


# In[17]:


train.head()


# #### layer sampling

# In[18]:


print("============ layer sampling ============")
train_layer = Layering(train, kf)
array = train_layer.values
X_train = array[:, 0:-1] # ndarray
y_train = array[:, -1] # ndarray


# Train/Validation Split

# In[19]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, y_train, test_size=valid_size,
                                                stratify = y_train, random_state = random_seed)
# stratify： 按正负样本原始比例random_seed分配给train 和 valid


# #### Do somte sampling on the train data to solve data imblance problem

# In[20]:


# X_train_oversampled, y_train_oversampled = Smoter(X_tr, y_tr, is_random=True)
X_train_oversampled, y_train_oversampled = OverSampler(X_tr, y_tr, is_random=True)

print("============ SMOTE ============")
print("train: %d, contains %.4f of 0 , after SMOTE: train: %d contains %.4f of 1" %(X_train.shape[0], (y_train == 0).sum()/y_train.shape[0], X_train_oversampled.shape[0], (y_train_oversampled == 0).sum()/y_train_oversampled.shape[0]))


# ### normalize the train and valid
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[21]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train_oversampled = clean_pipeline.fit_transform(X_train_oversampled)
X_vld = clean_pipeline.fit_transform(X_vld)


# ### transfer y into probability vector

# In[22]:


y_train_oversampled_pro = np.zeros([y_train_oversampled.shape[0], 2])
for i in range(len(y_train_oversampled)):
    if y_train_oversampled[i] == 1:
        y_train_oversampled_pro[i] = np.array([0, 1])
    else:
        y_train_oversampled_pro[i] = np.array([1, 0])
y_train_oversampled = y_train_oversampled_pro    

y_vld_pro = np.zeros([y_vld.shape[0], 2])
for i in range(len(y_vld)):
    if y_vld[i] == 1:
        y_vld_pro[i] = np.array([0, 1])
    else:
        y_vld_pro[i] = np.array([1, 0])
y_vld = y_vld_pro


y_vld_nonezero = np.count_nonzero(np.argmax(y_vld, 1))
print("y_vld: 1 contains: ", y_vld_nonezero, "/",len(y_vld))


# ## load 2017 test

# In[23]:


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


X_test = clean_pipeline.transform(X_test) 

y_test_pro = np.zeros([y_test.shape[0], 2])
for i in range(len(y_test)):
    if y_test[i] == 1:
        y_test_pro[i] = np.array([0, 1])
    else:
        y_test_pro[i] = np.array([1, 0])
y_test = y_test_pro


# In[24]:


y_test_nonezero = np.count_nonzero(np.argmax(y_test, 1))
print("y_test: 1 contains: ", y_test_nonezero, "/",len(y_test))


# ## 构造网络
# 两个（卷积+最大池化），两个全联接层

# In[26]:


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 9])
ys = tf.placeholder(tf.float32, [None, 2])
learning_rate = tf.placeholder(tf.float32)
X_input = tf.reshape(xs,[-1,9,1]) # [n_samples, 9 ,1]    -1 具体是多少由导入数据决定（多少组数据） 
    
# def my_cnn():
    
#     ## conv1 layer ##
#     W_conv1 = weight_variable([3,1,6],name="W_conv1") # patch: 3, in size 1(通道数), out size 6（feature_map数量，一个卷积核生成一个feature_map）
#     b_conv1 = bias_variable([6],name="b_conv1")
#     h_conv1 = tf.nn.relu(conv1d(X_input, W_conv1) + b_conv1) # output size 1*9*6
#     print("h_conv1", h_conv1.shape)
#     ## poo1 layer ##
#     max_pool_1 = tf.layers.max_pooling1d(inputs=h_conv1, pool_size=pool_patch_size, strides=stride_num, padding='same')
#     print("max_pool_1", max_pool_1.shape)

#     ## conv2 layer ##
#     W_conv2 = weight_variable([3,6,12],name="W_conv2") # patch: 3, in size 6，out size 12
#     b_conv2 = bias_variable([12],name="b_conv2")
#     h_conv2 = tf.nn.relu(conv1d(max_pool_1, W_conv2) + b_conv2) # output size 1*9*12
#     print("h_conv2", h_conv2.shape)
#     ## poo2 layer ##
#     max_pool_2 = tf.layers.max_pooling1d(inputs=h_conv2, pool_size=pool_patch_size, strides=stride_num, padding='same')
#     print("max_pool_2", max_pool_2.shape)

#     ## func1 layer ##
#     W_fc1 = weight_variable([9*12,hidden_cell_num],name="W_fc1")
#     b_fc1 = bias_variable([hidden_cell_num],name="b_fc1")

#     max_pool_2_flat = tf.reshape(max_pool_2, [-1,9*12])
#     h_fc1 = tf.nn.relu(tf.matmul(max_pool_2_flat, W_fc1)+b_fc1)
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#     print("h_fc1_drop", h_fc1_drop.shape)

#     ## func2 layer ##
#     W_fc2 = weight_variable([hidden_cell_num,2],name="W_fc2")
#     b_fc2 = bias_variable([1],name="b_fc2")
#     prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#     print("prediction", prediction.shape)

#     var_dict = {'W_conv1': W_conv1, 
#                 'b_conv1': b_conv1, 
#                 'W_conv2': W_conv2, 
#                 'b_conv2': b_conv2, 
#                 'W_fc1': W_fc1, 
#                 'b_fc1': b_fc1,
#                 'W_fc2': W_fc2,
#                 'b_fc2': b_fc2}
#     return prediction, var_dict


def my_cnn():
    ## func1 layer ##
    W_fc1 = weight_variable([9,2],name="W_fc1")
    b_fc1 = bias_variable([2],name="b_fc1")

    X_input_flat = tf.reshape(X_input, [-1,9])
    h_fc1 = tf.nn.softmax(tf.matmul(X_input_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    prediction = h_fc1_drop
    
    # h_fc1 = tf.nn.relu(tf.matmul(X_input_flat, W_fc1)+b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # print("h_fc1_drop", h_fc1_drop.shape)

    # ## func2 layer ##
    # W_fc2 = weight_variable([8,4],name="W_fc2")
    # b_fc2 = bias_variable([4],name="b_fc2")
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # print("h_fc2_drop", h_fc2_drop.shape)


    # W_fc3 = weight_variable([4,2],name="W_fc3")
    # b_fc3 = bias_variable([2],name="b_fc3")
    # prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    # print("prediction", prediction.shape)


    # var_dict = {'W_fc1': W_fc1, 
    #             'b_fc1': b_fc1,
    #             'W_fc2': W_fc2,
    #             'b_fc2': b_fc2,
    #             'W_fc3': W_fc3,
                # 'b_fc3': b_fc3}

    var_dict = {'W_fc1': W_fc1, 
                'b_fc1': b_fc1}
    return prediction, var_dict



# ## loss

# In[27]:


prediction, var_dict = my_cnn()

# the error between prediction and real data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ## train

# In[ ]:


vld_acc = []
vld_f1 = []
vld_loss = []

train_acc = []
train_f1 = []
train_loss = []

X_train_oversampled = np.array(X_train_oversampled, dtype=np.float32)
y_train_oversampled = np.array(y_train_oversampled, dtype=np.float32)
X_vld = np.array(X_vld, dtype=np.float32)
y_vld = np.array(y_vld, dtype=np.float32)

saver = tf.train.Saver(var_dict)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
            
    true_iteration = 1
    for i in range(max_iterations):
        feed = {xs : X_train_oversampled, ys : y_train_oversampled, keep_prob : kp, learning_rate : lr}
        
        # train
        # print("++++++++++++++++++ train ++++++++++++++++++")
        loss, _ = sess.run([cost, train_op],feed_dict=feed)
        precision, recall, f1_score, accuracy = evaluate(X_train_oversampled, y_train_oversampled, sess)
        
        train_loss.append(loss)
        train_f1.append(f1_score)
        train_acc.append(accuracy)
        
        # vld cost
        loss_v = sess.run(cost, feed_dict={xs: X_vld, ys: y_vld, keep_prob: 1})
        vld_loss.append(loss_v)
        # vld evaluationX_testX_testX_testX_test
        # print("++++++++++++++++++ valid ++++++++++++++++++")
        vld_precision, vld_recall, vld_f1_score, vld_accuracy = evaluate(X_vld, y_vld, sess)
        vld_f1.append(vld_f1_score)
        vld_acc.append(vld_accuracy)

        if vld_f1_score >= 0.98:
            break
        print("Iteration: {}/{}\n".format(true_iteration, max_iterations),
              "Train loss: {:6f}".format(loss),
              "Train acc: {:.6f}".format(accuracy),
              "Train f1: {:.6f}\n".format(f1_score),
              "Valid loss: {:6f}".format(loss_v),
              "Valid acc: {:.6f}".format(vld_accuracy),
              "Valid f1: {:.6f}".format(vld_f1_score))
        true_iteration += 1
            
    save_path = saver.save(sess,"cnn_2017/2017_save_net.ckpt")
    print("Save to path:", save_path)


# In[ ]:


# Plot training and valid loss
t = np.arange(true_iteration - 1)


plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t, np.array(vld_loss), 'b-')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'valid'], loc='upper right')
plt.show()



# Plot learning rate
# t = np.arange(0.001, 0.1, 0.001)
# plt.figure(figsize = (6,6))
# plt.plot(t, np.array(vld_loss), 'b-')
# plt.xlabel("Learning_rate")
# plt.ylabel("Loss")
# plt.legend(['valid'], loc='upper right')
# plt.show()

# In[ ]:


# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t, vld_acc, 'b-')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


# In[ ]:


# Plot F1
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_f1), 'r-', t, vld_f1, 'b-')
plt.xlabel("iteration")
plt.ylabel("F1")
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


 ### smote采样的数据集上	5折

| model_name | valid_accuracy | valid_precision | valid_recall | valid_f1 score | test_accuracy | test_precision | test_recall | test_f1 score |
| --------   | --------   | --------   | --------   | --------   | --------   | --------   | --------   | -----:  |
| LogisticRegression | 0.850024| 0.066804 |	0.549230 |	0.116250 |	0.983420 | 1.000000 |	0.913686 |	0.954896|
| LinearDiscriminantAnalysis | 0.857033 | 0.035046 | 0.307132 | 0.062557 | 0.982927 | 0.997651 | 0.913290 | 0.953603 |
| LinearSVC | 0.866736 | 0.102196 | 0.547494 | 0.160790 | 0.983413 | 1.000000 | 0.913648 | 0.954876 |
| DecisionTreeClassifier | 0.833360 | 0.045214 | 0.332460 |  0.076179 | 0.798114 | 0.404598 | 0.556530 | 0.453227 |
| ExtraTreeClassifier | 0.912959 | 0.082642 | 0.225254|  0.107365 | 0.789328 | 0.330405 | 0.371480 | 0.349241 |
| GaussianNB | 0.981129 | 0.444949 | 0.864983| 0.565912 | 0.983881| 1.000000| 0.916088| 0.956207 |
| KNeighborsClassifier| 0.987792| 0.572797| 0.554556| 0.517920| 0.807890| 0.000000| 0.000000| 0.000000|
| RandomForestClassifier| 0.983656| 0.338093| 0.147712| 0.189545| 0.795492| 0.405304| 0.002566| 0.004641|
| ExtraTreesClassifier| 0.988161| 0.722452| 0.205105| 0.299578| 0.841193| 0.390247| 0.182849| 0.186805|

### 不采样数据集上  5折
| model_name| fit_time| score_time | valid_accuracy| train_accuracy| valid_precision | train_precision| valid_recall| train_recall| valid_f1| train_f1 |
| --------   | --------   | --------   | --------   | --------   | --------   | --------   | --------   | --------   | --------   | -----:  |
| LogisticRegression| 1.558848| 0.044470| 0.989771| 0.991110| 0.919745| 0.996803| 0.215597| 0.287360| 0.334206| 0.441984|
| LinearDiscriminantAnalysis| 0.203514| 0.078648| 0.990738| 0.991978| 0.870842| 0.977820| 0.300513| 0.364276| 0.427147| 0.529237|
| LinearSVC| 30.584592| 0.038038| 0.987764| 0.988430| 0.800000| 0.989005| 0.017968| 0.072706| 0.034503| 0.131825|
| DecisionTreeClassifier| 0.730955| 0.039737| 0.842449| 0.999984| 0.377514| 1.000000| 0.543472| 0.998696| 0.345645| 0.999347|
| ExtraTreeClassifier| 0.114650| 0.044397| 0.940456| 0.999984| 0.313748| 1.000000| 0.618998| 0.998696| 0.333215| 0.999347|
| GaussianNB| 0.068438| 0.049321| 0.935612| 0.989538| 0.466529| 0.580089| 0.616292| 0.605139| 0.422856| 0.589307|
| KNeighborsClassifier| 0.223291| 1.740487| 0.962056| 0.995679| 0.163934| 0.968936| 0.228768| 0.674237| 0.169068| 0.793933|
| RandomForestClassifier| 1.683555| 0.127813| 0.937070| 0.999921| 0.484783| 0.999564| 0.587459| 0.994061| 0.428754| 0.996802|
| ExtraTreesClassifier| 0.686059| 0.144104| 0.956699| 0.999984| 0.467763| 1.000000| 0.384654| 0.998696| 0.316707| 0.999347|



不采样数据集：train > valid
SMOTE采样数据集：test > valid

不采样valid（Bayes） < SMOTE采样valid（Bayes），但显然其他模型的效果基本都下降了

所以是哪里出了问题？

Applying Tree Ensemble to Detect Anomalies in Real-World 中

训练集8份，测试集2份，5折交叉验证基于分层采样。调参数基于暴力遍历搜索（网格搜索）Parameter Tuning utilizes exhaustive search over the parameter grid，and each set of selected parameters is evaluated with 5-fold cross valida- tion. 
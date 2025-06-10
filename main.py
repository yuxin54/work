import pandas as pd 
#导入数据 
data_train = pd.read_csv('D:/机器学习课设/tianchi-game---code-master/01-金融风控-贷款违约预测/train.csv')
data_test_a = pd.read_csv(r'D:\机器学习课设\tianchi-game---code-master\01-金融风控-贷款违约预测\testA.csv')

#查看数据整体
data_train

#查看所有数据类型
data_train.info()

data_test_a
a = list(data_train.columns.values)
b = list(data_test_a.columns.values)

#获取两个列表中的交集
a_intersection_b = list(set(a).intersection(set(b)))
a_intersection_b
#获取两个列表中的并集
a_union_b = list(set(a).union(set(b)))
a_union_b

#获取两个列表中的差集a中有，但b中没有
a_difference_b = list(set(a).difference(set(b)))
a_difference_b

#查看标签，在isDefault中，1表示违约的，0表示没有违约的
data_train[['id','isDefault']]

#统计下0和1的个数
data_train.isDefault.value_counts()

#先去掉id这个无用特征
data_train = data_train.iloc[:,1:]
data_test_a = data_test_a.iloc[:,1:]
data_train

#先引用一个autoML的一个机器学习包pycaret，进行简单的数据分析建模
from pycaret.classification import *
#70%训练集，30%测试集
data_train_clf = setup(data_train,target='isDefault',train_size=0.9,n_jobs=6)
catboost = create_model('catboost', cross_validation=False)
ada_model = create_model('ada', cross_validation=False)

#画模型
plot_model(catboost,save=True)

#评估模型
evaluate_model(catboost)

#预测模型
catboost_test_a = predict_model(catboost,data=data_test_a)
catboost_test_a
catboost_test_a_submit = catboost_test_a[['prediction_label','prediction_score']]
catboost_test_a_submit
data_test_a1 = pd.read_csv(r'D:\机器学习课设\tianchi-game---code-master\01-金融风控-贷款违约预测\testA.csv')
data_test_a1
data_test_a1['id']
catboost_test_a_submit.insert(0,'id',data_test_a1['id'])
catboost_test_a_submit
catboost_test_a_submit.loc[catboost_test_a_submit.prediction_label==0,'prediction_score'] = 1-catboost_test_a_submit.prediction_score
catboost_test_a_submit

#更改列名称
catboost_test_a_submit.rename(columns={'prediction_score':'isDefault'},inplace=True)
catboost_test_a_submit
catboost_test_a_submit_end = catboost_test_a_submit[['id','isDefault']]
catboost_test_a_submit_end
catboost_test_a_submit_end.to_csv('catboost_submit1.csv',index=False) 

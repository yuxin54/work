import pandas as pd 
"""
导入数据 
"""
data_train = pd.read_csv('F:/tianchi-game---code-master/01-金融风控-贷款违约预测/train.csv')
data_test_a = pd.read_csv('F:/tianchi-game---code-master/01-金融风控-贷款违约预测/testA.csv')
data_train
"""
查看所有数据类型
""" 
data_train.info()
data_test_a
a = list(data_train.columns.values)
b = list(data_test_a.columns.values)
"""
获取两个列表中的交集
"""
a_intersection_b = list(set(a).intersection(set(b)))
a_intersection_b
"""
获取两个列表中的并集
"""
a_union_b = list(set(a).union(set(b)))
a_union_b
"""
获取两个列表中的差集
a中有，但b中没有
"""
a_difference_b = list(set(a).difference(set(b)))
a_difference_b
"""
查看标签，在isDefault中，1表示违约的，0表示没有违约的
"""
data_train[['id','isDefault']]
"""
统计下0和1的个数
"""
data_train.isDefault.value_counts()
"""
先去掉id这个无用特征
"""
data_train = data_train.iloc[:,1:]
data_test_a = data_test_a.iloc[:,1:]
data_train
"""
先引用一个autoML的一个机器学习包pycaret，进行简单的数据分析建模
"""

from pycaret.classification import *
#from pycaret.classification import add_model
"""
70%训练集，30%测试集
"""
data_train_clf = setup(data_train,target='isDefault',train_size=0.7,n_jobs=6)  # 全局限制并行任务数)
catboost = create_model('catboost', cross_validation=False)

# 设置并行后端
#n_jobs = 1  # 设置为1以减少并行任务的数量
# 使用Parallel和delayed进行调优
tunedcatboost = tune_model(catboost, optimize='AUC')

import matplotlib.pyplot as plt
import matplotlib

# 设置图形格式为SVG
matplotlib.use('Agg')  # 如果在无界面的环境中运行，需要设置后端
plt.switch_backend('Agg')  # 确保使用非交互式后端
plt.rcParams['svg.fonttype'] = 'none'  # 使SVG中的文本为普通文本而非路径

plot_model(catboost,save=True)
evaluate_model(catboost)
interpret_model(catboost)
catboost_test_a = predict_model(catboost,data=data_test_a)
catboost_test_a_submit = catboost_test_a[['Label','Score']]
catboost_test_a_submit
data_test_a1 = pd.read_csv('F:/tianchi-game---code-master/01-金融风控-贷款违约预测/testA.csv')
data_test_a1
data_test_a1['id']
catboost_test_a_submit.insert(0,'id',data_test_a1['id'])
catboost_test_a_submit
catboost_test_a_submit.loc[catboost_test_a_submit.Label==0,'Score'] = 1-catboost_test_a_submit.Score
catboost_test_a_submit
"""
更改列名称 
"""
catboost_test_a_submit.rename(columns={'Score':'isDefault'},inplace=True)
catboost_test_a_submit
catboost_test_a_submit_end = catboost_test_a_submit[['id','isDefault']]
catboost_test_a_submit_end
catboost_test_a_submit_end.to_csv('catboost_submit.csv',index=False) 
# -----------------线性回归----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 内嵌画图，直接在输出画图
%matplotlib inline
data = pd.read_csv("data/实验报告数据/data_regression.csv")
# # 对特征进行归一化
data = (data-data.min())/(data.max()-data.min())
data.head()

# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6,7,8]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

# 绘制训练集散点图
sns.lmplot(x = 'Feature_5',y = 'Lable',data = data)
plt.show()

# 导入sklearn的线性回归模型/
from sklearn.linear_model import LinearRegression
line = LinearRegression()
# 训练模型
lin = line.fit(x_train,y_train)


# 获取预测值
Y_pre = lin.predict(x_test)
# 计算MSE值
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,Y_pre)
print("MSE值为:%s" % MSE)

# --------------------KNN--------------
import pandas as pd
import numpy as np
# 导入模型,训练KNN模型
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale
# 导入数据
data = pd.read_csv("data/实验报告数据/data_regression.csv")
# 标准化
data=(data - data.mean()) / (data.std())
# data = pd.DataFrame(minmax_scale(data))
data.head()

# 分割数据,分为训练集,测试集
# test_size:测试集占比
from sklearn.preprocessing import minmax_scale  #归一化库
from sklearn.model_selection import train_test_split

X = data.iloc[:,[1,2,3,4,5,6,7,8]]

Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# # 遍历k,寻找最优k值
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# # 中文和负号的正常显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# K = np.arange(1,np.ceil(np.log2(data.shape[0]))).astype('int').tolist()
# score_list =[]
# for i in K:
#     score = cross_val_score(KNeighborsRegressor(i,weights='distance'),x_train,y_train,cv=10,scoring='neg_mean_squared_error').mean()
#     score_list.append(-1*score)
# plt.figure(figsize=[8,4])

# plt.plot(K,score_list)
# # 最佳K值
# arg_min = np.array(score_list).argmin()
# plt.text(K[arg_min], score_list[arg_min] + 300, '最佳k值为%s' %int(K[arg_min]))
# plt.show()

# 网格搜索寻找最优参数K
from sklearn.model_selection import GridSearchCV
n_neighbors = range(1,10)
tips = {"n_neighbors":n_neighbors}
gr = GridSearchCV(KNeighborsRegressor(),tips).fit(x_train,y_train)
print("最优参数：")
gr.best_params_

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(x_train,y_train)
Y_pre = knn.predict(x_test)

# 计算MSE值
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,Y_pre)
print("MSE值为:%s" % MSE)

# -------------------随机森林----------------
import pandas as pd
import numpy as np
# 导入第三方库
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data/实验报告数据/data_regression.csv")
# 特征标准化
data=(data - data.mean()) / (data.std())
# 分割数据,分为训练集,测试集
# test_size:测试集占比
from sklearn.model_selection import train_test_split

X = data.iloc[:,[1,2,3,4,5,6,7,8]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
# x_test.tail()

# 网格搜索
from sklearn.model_selection import GridSearchCV
n_estimators = range(10,100)
tips = {"n_estimators":n_estimators}
gr = GridSearchCV(RandomForestRegressor(),tips).fit(x_train,y_train)
print("最优参数：")
gr.best_params_


ran = RandomForestRegressor(n_estimators=57,criterion="mse")
ran.fit(x_train,y_train)
Y_pre = ran.predict(x_test)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,Y_pre)
print(MSE)

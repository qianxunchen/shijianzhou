''' 
使用sklearn库解决分类问题的五个经典算法，基础用法，没有对数据进行处理
'''

#-----------------------逻辑回归----------------

import pandas as pd
import numpy as np

data = pd.read_csv("data/实验报告数据/data_classification.csv")
data.head()

# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

#拟合
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)
Y_pre = log.predict(x_test)

#计算准确率
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,Y_pre)
print("准确率:%s" % score)


#-------------------KNN--------------------
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("data/实验报告数据/data_classification.csv")

# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#网格搜索
tips = [
    {
        "weights":['uniform'],
        "n_neighbors":range(2,10),
        "p":range(1,6)
    },
    {
        "weights":['distance'],
        "n_neighbors":range(2,10),
        "p":range(1,6)
    }
]

# 网格搜索
from sklearn.model_selection import GridSearchCV
gr = GridSearchCV(KNeighborsClassifier(),tips).fit(x_train,y_train)
print("最优参数：")
gr.best_params_

# 拟合
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4,p=4,weights="distance")
knn.fit(x_train,y_train)
Y_pre = knn.predict(x_test)

# 准确率
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,Y_pre)
print("准确率:%s" % score)

# -----------------贝叶斯------------------
import pandas as pd
import numpy as np
data = pd.read_csv("data/实验报告数据/data_classification.csv")


# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.35,random_state=0)

from sklearn.naive_bayes import GaussianNB
gn = GaussianNB()
gn.fit(x_train,y_train)
Y_pre = gn.predict(x_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,Y_pre)
print("准确率:%s" % score)


# --------------------决策树--------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("data/实验报告数据/data_classification.csv")

# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

# 拟合
treeclass = DecisionTreeClassifier()
treeclass.fit(x_train,y_train)
Y_pre = treeclass.predict(x_test)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,Y_pre)
print("准确率:%s" % score)

# -------------------XGBoost-----------
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
data = pd.read_csv("data/实验报告数据/data_classification.csv")

# 分割数据,分为训练集,测试集
from sklearn.model_selection import train_test_split
X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.35,random_state=0)

tips =  {
        "max_depth":range(3,10),
        "n_estimators":range(10,100)
    }

# 网格搜索
from sklearn.model_selection import GridSearchCV
gr = GridSearchCV(XGBClassifier(),tips).fit(x_train,y_train)
print("最优参数：")#输出最优参数
gr.best_params_

xgb = XGBClassifier(max_depth=3,n_estimators=65)
xgb.fit(x_train,y_train)
xg_pre = xgb.predict(x_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,xg_pre)
print("准确率:%s" % score)



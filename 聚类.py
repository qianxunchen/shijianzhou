# ----------------K-Means（基于划分）-----------
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
data = pd.read_csv("data/实验报告数据/data_clustering.csv")
data.head()
# data = data.iloc[:,[1,2,3,4,5,6,7]]
data = data.drop("Unnamed: 0",axis=1)
data

# 计算轮廓系数,寻找最优解K
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(8,6))

sc=[]
for i in range(2,10):
    cluster_=KMeans(n_clusters=i,random_state=0).fit(data)
    y_pre=cluster_.labels_
    sc_=silhouette_score(data,y_pre)
    sc.append(sc_)
plt.plot(range(2,10),sc,color='red',linewidth=2.0,marker='o')
plt.ylabel("轮廓系数")
plt.xlabel("聚簇数量")
plt.show()

from sklearn.cluster import KMeans
# 最优解 k = 2
km = KMeans(n_clusters=2)
km.fit(data)
data['cluster'] = km.labels_
# 各类频数统计
data.cluster.value_counts()


# 轮廓系数
sil = silhouette_score(data,data['cluster']).round(3)
print("轮廓系数为:%s" % sil)
# >>轮廓系数为:0.532

# 对数据进行降维在聚类
data = data.drop("cluster",axis=1)
# 使用PCA对数据将维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
# 构建DataFrame
data = pd.DataFrame(data,columns=["1","2"])

# 计算轮廓系数,寻找最优解K
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(8,6))

sc=[]
for i in range(2,10):
    cluster_=KMeans(n_clusters=i,random_state=0).fit(data)
    y_pre=cluster_.labels_
    sc_=silhouette_score(data,y_pre)
    sc.append(sc_)
plt.plot(range(2,10),sc,color='red',linewidth=2.0,marker='o')
plt.ylabel("轮廓系数")
plt.xlabel("聚簇数量")
plt.show()

from sklearn.cluster import KMeans
# 最优解 k = 3
km = KMeans(n_clusters=3)
km.fit(data)
data['cluster'] = km.labels_

# 降维后的轮廓系数
sil = silhouette_score(data,data['cluster'])
print("轮廓系数为:%s" % sil)

# 绘制聚类后的散点图
# import seaborn as sns
# # 三个簇的簇中心
# centers = km.cluster_centers_
# # 绘制聚类效果的散点图
# # sns.lmplot(x = '1', y = '2', hue = 'cluster', markers = ['^','s','o'], 
# #            data = data, fit_reg = False, scatter_kws = {'alpha':0.8}, legend_out = False)
# # plt.scatter(centers[:,0], centers[:,1], marker = '*', color = 'black', s = 130)
plt.scatter(data['1'], data['2'], c = data['cluster'],marker='*', s = 130)
plt.xlabel('1')
plt.ylabel('2')
plt.show()



# ----------------DBSCAN（基于密度）-----------
import pandas as pd
import numpy as np
data = pd.read_csv("data/实验报告数据/data_clustering.csv")
# 去除无关数据
data = data.iloc[:,[1,2,3,4,5,6,7]]
data.head()

# 寻找最优参数，轮廓系数法
from sklearn.metrics import silhouette_score
import pandas as pd
sc=[]
ep = []
min_sa = []
for eps in np.arange(2,5,0.1):
    
    for i in range(5,20):
        cluster_=DBSCAN(eps=eps,min_samples=i).fit(data)
        y_pre=cluster_.labels_
        sc_=silhouette_score(data,y_pre)
        sc.append(sc_)
        ep.append(eps)
        min_sa.append(i)

df = pd.DataFrame({"轮廓系数":sc,"eps":ep,"min_sa":min_sa})
df.sort_values('轮廓系数',ascending=False)

from sklearn.cluster import DBSCAN
# eps:扫描半径，min_samples:一个类最小包含的点数
dbs = DBSCAN(eps=4.9,min_samples=19)
dbs.fit(data)
data['labels'] = dbs.labels_
# 统计各类的数据量
data['labels'].value_counts()

from sklearn.metrics import silhouette_samples, silhouette_score
sil = silhouette_score(data,data['labels'])
print("轮廓系数为:%s" % sil)
# >>轮廓系数为:0.4750946156935249


# 降维
data = data.drop("labels",axis=1)
# 使用PCA对数据将维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
data = pd.DataFrame(data,columns=["1","2"])
data

# 计算轮廓系数,寻找最优解K
from sklearn.metrics import silhouette_score
import pandas as pd

sc=[]
ep = []
min_sa = []
for eps in np.arange(1,3,0.1):
    
    for i in range(10,80):
        cluster_=DBSCAN(eps=eps,min_samples=i).fit(data)
        y_pre=cluster_.labels_
        sc_=silhouette_score(data,y_pre)
        sc.append(sc_)
        ep.append(eps)
        min_sa.append(i)

df = pd.DataFrame({"轮廓系数":sc,"eps":ep,"min_sa":min_sa})
df.sort_values('轮廓系数',ascending=False)

from sklearn.cluster import DBSCAN
# eps:扫描半径，min_samples:一个类最小包含的点数
dbs = DBSCAN(eps=2.6,min_samples=43)
dbs.fit(data)
data['labels'] = dbs.labels_
# 统计各类的数据量
data['labels'].value_counts()

# 降维后的轮廓系数
from sklearn.metrics import silhouette_samples, silhouette_score
sil = silhouette_score(data,data['labels'])
print("轮廓系数为:%s" % sil)

# # 绘制聚类效果的散点图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(8,6))
plt.scatter(data['1'], data['2'], c = data['labels'], marker = '*', s = 130)
plt.show()

# ------------------AgglomerativeClustering(基于层次)---------------

from  sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/实验报告数据/data_clustering.csv")
data = data.iloc[:,[1,2,3,4,5,6,7]]
data.head()

# 计算轮廓系数,寻找最优簇数量
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(7,5))

sc=[]
for i in range(2,10):
    cluster_=AgglomerativeClustering(n_clusters=i).fit(data)
    y_pre=cluster_.labels_
    sc_=silhouette_score(data,y_pre)
    sc.append(sc_)
plt.plot(range(2,10),sc,color='red',linewidth=2.0,marker='o')
plt.ylabel("轮廓系数")
plt.xlabel("聚簇数量")
plt.show()

agg = AgglomerativeClustering(n_clusters=3)
agg.fit(data)
data['labels'] = agg.labels_
data['labels'].value_counts()

# 轮廓系数
from sklearn.metrics import silhouette_samples, silhouette_score
sil = silhouette_score(data,data['labels'])
print("轮廓系数为:%s" % sil)
# >>轮廓系数为:0.48662306849458603

# 降维
data = data.drop("labels",axis=1)
# 使用PCA对数据将维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
data = pd.DataFrame(data,columns=["1","2"])
data
# 计算轮廓系数,寻找最优簇数量
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(7,5))

sc=[]
for i in range(2,10):
    cluster_=AgglomerativeClustering(n_clusters=i,linkage='average',affinity='cosine').fit(data)
    y_pre=cluster_.labels_
    sc_=silhouette_score(data,y_pre)
    sc.append(sc_)
plt.plot(range(2,10),sc,color='red',linewidth=2.0,marker='o')
plt.ylabel("轮廓系数")
plt.xlabel("聚簇数量")
plt.show()

agg = AgglomerativeClustering(n_clusters=3,linkage='average',affinity='cosine')
agg.fit(data)
data['labels'] = agg.labels_
data['labels'].value_counts()

# 轮廓系数
from sklearn.metrics import silhouette_samples, silhouette_score
sil = silhouette_score(data,data['labels'])
print("轮廓系数为:%s" % sil)
# >>轮廓系数为:0.6974887374456571

# # 绘制聚类效果的散点图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(8,6))
plt.scatter(data['1'], data['2'], c = data['labels'], marker = '*', s = 130)
plt.show()


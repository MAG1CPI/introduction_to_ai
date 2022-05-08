import os
import joblib
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score,silhouette_score

warnings.filterwarnings('ignore')


class KMeans():
    '''
    Parameters
    ----------
    n_clusters 聚类数

    n_init 计算次数，运行指定次数后返回最好的一次结果

    max_iter 单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    '''

    def __init__(self, n_clusters:'int'=8, n_init:'int'=10, max_iter:'int'=300)->'None':
        '''
        :param n_clusters: 聚类数
        :param n_init: 计算次数，运行指定次数后返回最好的一次结果
        :param max_iter: 单次运行中最大的迭代次数，超过当前迭代次数即停止运行
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, x:'pd.DataFrame')->'KMeans':
        '''
        用fit方法对数据进行聚类

        :param x: 输入数据
        :best_centers: 簇中心点坐标，数据类型: ndarray
        :best_labels: 聚类标签，数据类型: ndarray
        :return self:
        '''
        data=x.values
        min_cost = 1e9
        for i in range(self.n_init):
            step = self.kmeans(data)
            if min_cost > step[-1]:
                best_centers, best_labels, min_cost = step

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

    def kmeans(self, samples:'np.ndarray')->'tuple[np.ndarray, np.ndarray, float]':
        '''
        kmeans算法

        :param x: 数据集
        :return centers: 聚类中心集
                labels: 标签集
                cost: 总代价
        '''
        # 初始化簇中心
        centers = self.init_center(samples)
        # 迭代聚类
        m,n=samples.shape
        # 初始化
        cost_old = 1e9
        cost_new = 0
        labels = np.zeros(m,int)
        cluster_list= []
        for i in range(self.n_clusters):
            cluster_list.append([])
        # 迭代
        for iter in range(self.max_iter):
            cost_new = 0
            labels = np.zeros(m,int)
            for i in range(self.n_clusters):
                cluster_list[i] = []
            # 根据到各簇中心距离聚类
            for i in range(m):
                # sample到各中心距离
                distances = self.distance(samples[i,:], centers)
                # sample标签
                sample_label = distances.argmin()
                # 标签列表更新
                labels[i]=sample_label
                # 在相应类别中添加x
                cluster_list[sample_label].append(samples[i,:])
                # 累加代价
                cost_new += distances[sample_label]
            # 更新各簇中心
            for i in range(self.n_clusters):
                centers[i] = np.mean(cluster_list[i])
            # 判断是否提前结束迭代
            if 0.0 <= (cost_old - cost_new) <1e-5:
                break
            cost_old = cost_new

        return centers, labels, cost_new

    def init_center(self, sample:'np.ndarray')->'np.ndarray':
        '''
        初始化簇类中心

        :param sample: 数据点集
        :return centers: 聚类中心集
        '''
        centers = np.zeros((self.n_clusters, sample.shape[1]))
        for i in range(self.n_clusters):
            num=np.random.choice(list(range(len(sample))))
            centers[i,:]=sample[num,:]
        return centers

    def distance(self, point:'np.ndarray', centers:'np.ndarray')->'np.ndarray':
        '''
        计算样本点到各簇中心的距离

        :param point: 样本点
        :param centers: 簇中心集
        :return distances: 样本点到各簇中心的距离
        '''
        distances=np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            distances[i]=np.sum((point - centers[i]) ** 2) ** 0.5
        return distances

def preprocess_data(df:'pd.DataFrame')->'pd.DataFrame':
    '''
    数据处理及特征工程

    :param df: 读取原始csv数据，有timestamp、cpc、cpm共3列
    :return data: 处理后的数据, 返回 pca 降维后的特征
    '''
    scaler = StandardScaler()
    # 通过 n_components 指定需要降低到的维度
    n_components = 5
    pca = PCA(n_components=n_components)
    # 数据标准化
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df[['cpc','cpm']] = scaler.fit_transform(df[['cpc','cpm']])
    # 引入其他关系
    # 时间的其他表示
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    # 非线性关系
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    df['cpc / cpm'] = df['cpc'] / df['cpm']

    # 选取要用的列
    columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm', 'hours', 'daylight']
    data = df[columns]
    # 归一化
    data = scaler.fit_transform(data)
    # 降维
    data = pca.fit_transform(data)
    data = pd.DataFrame(data, columns=['Dimesion' + str(i + 1) for i in range(n_components)])
  
    return data

def get_distance(data:'pd.DataFrame', kmeans:'KMeans', n_features:'int')->'pd.Series':
    '''
    计算样本点与其对应的聚类中心的距离

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return distance: 每个点距离自己簇中心的距离
    '''
    distance = []
    # 遍历所有数据
    for i in range(0,len(data)):
        # 样本点
        point = np.array(data.iloc[i,:n_features])
        # 相应聚类的中心点
        center = kmeans.cluster_centers_[kmeans.labels_[i],:n_features]
        # L2范数
        distance.append(np.linalg.norm(point - center))

    distance = pd.Series(distance)
    return distance

def get_anomaly(data:'pd.DataFrame', kmeans:'KMeans', ratio:'float') ->'pd.DataFrame':
    '''
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间
    :return data: 添加 distance, is_anomaly列，is_anomaly列数据是根据阈值距离大小判断每个点是否是异常值，值为False/True
    '''
    # 异常点数
    num_anomaly = int(len(data) * ratio)
    # 不改变原数据
    new_data = deepcopy(data)
    # 添加distance列
    new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
    # 获得阈值
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    # 分类，添加is_anomaly列
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)

    return new_data

def predict(preprocess_data:'pd.DataFrame') ->'tuple[pd.DataFrame, pd.DataFrame, KMeans, float]':
    '''
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断

    :param preprocess_data: preprocess_data 函数返回值，即 pca 降维后的数据
    :return is_anomaly: get_anomaly函数的返回值，各个属性为(Dimesion1,Dimension2,......数量取决于具体的pca)，distance, is_anomaly
            preprocess_data: 即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio: 异常点的比例，ratio <= 0.03
    '''
    # 异常值所占比率
    ratio = 0.03
    # 加载模型 
    kmeans = joblib.load('./results/model.pkl')
    # 获取异常点数据信息
    is_anomaly = get_anomaly(preprocess_data, kmeans, ratio)
    
    return is_anomaly, preprocess_data, kmeans, ratio

if __name__=='__main__':
    # 读文件
    file_dir = './data'
    csv_files = os.listdir(file_dir)
    # 获取数据
    df = pd.DataFrame()
    feature = ['cpc', 'cpm']
    df_features = []
    for col in feature:
        infix = col + '.csv'
        path = os.path.join(file_dir, infix)
        df_feature = pd.read_csv(path)
        # 将两个特征存储起来用于后续连接
        df_features.append(df_feature)
    # 两张表按时间连接
    df = pd.merge(left=df_features[0], right=df_features[1])

    # 预处理
    data=preprocess_data(df)
    # 聚类
    kmeans = KMeans(n_clusters=3, n_init=50, max_iter=100)
    kmeans.fit(data)
    # 评分
    score1 = calinski_harabasz_score(data,kmeans.labels_)
    score2 = silhouette_score(data,kmeans.labels_)
    print('calinski_harabasz_score:', score1)
    print('silhouette_score:', score2)
    # 保存模型
    joblib.dump(kmeans, './results/model.pkl')

    # 预测
    new_data, preprocess_data, kmeans, ratio=predict(data)

    # 可视化分析
    # 'timestamp'&'cpc'
    a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpc']] 
    plt.figure(figsize=(20,6))
    plt.plot(df['timestamp'], df['cpc'], color='blue')
    # 聚类后 cpc 的异常点
    plt.scatter(a['timestamp'],a['cpc'], color='red')
    plt.show()
    # 'timestamp'&'cpm'
    a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpm']] 
    plt.figure(figsize=(20,6))
    plt.plot(df['timestamp'], df['cpm'], color='blue')
    # 聚类后 cpm 的异常点
    plt.scatter(a['timestamp'],a['cpm'], color='red')
    plt.show()
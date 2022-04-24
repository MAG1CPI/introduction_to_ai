# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    # 向量化方法
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler                    # 归一化方法
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB        # 模型
from sklearn import metrics
import joblib



def read_stopwords(stopwords_path: 'str') -> 'list':
    '''
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表
    '''
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = file.read()
    stopwords = stopwords.splitlines()
    return stopwords


def train_model(X: 'np.ndarray', y: 'np.ndarray', stopwords: 'list', model_path: 'str' = '.\\results', model_name: 'str' = 'result') -> None:
    '''
    训练模型, 然后将模型保存
    :param X: 数据集特征值
    :param y: 数据集标签值
    :param model_path: 保存的模型路径(使用'.'表示当前目录)
    :param model_name: 保存的模型名称
    '''
    # 构建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)
    
    # 搭建 pipeline
    pipeline_list = [
        ('cv', CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2), max_df=0.6, stop_words=stopwords)),
        ('MaxAbsScaler', MaxAbsScaler()),
        ('classifier', MultinomialNB())
    ]
    process = Pipeline(pipeline_list)

    # 对训练集的数据进行训练
    process.fit(X_train, y_train)
    # 对测试集的数据集进行预测
    y_pred = process.predict(X_test)
    # 在测试集上评估结果
    print('在测试集上的混淆矩阵: ')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('在测试集上的分类结果报告: ')
    print(metrics.classification_report(y_test, y_pred))
    print('在测试集上的 f1-score: ', metrics.f1_score(y_test, y_pred))
    print('在测试集上的准确率：', metrics.accuracy_score(y_test, y_pred))

    # 在所有的样本上训练一次，充分利用已有的数据，提高模型的泛化能力
    process.fit(X, y)

    # 保存 pipeline
    path = model_path+'\\'+model_name+'.model'
    joblib.dump(process, path)


if __name__ == '__main__':
    # 读取数据，获得数据集特征值与标签值
    data_path = '.\\datasets\\sms_pub.csv'
    sms = pd.read_csv(data_path, encoding='utf-8')
    # sms_pos = sms[(sms['label'] == 1)]
    # sms_neg = sms[(sms['label'] == 0)].sample(frac=1.0)[: 2*len(sms_pos)]
    # sms = pd.concat([sms_pos, sms_pos, sms_neg], axis=0).sample(frac=1.0)
    X = np.array(sms.msg_new)
    y = np.array(sms.label)

    # 读取停用词
    stopwords_path = '.\\datasets\\scu_stopwords.txt'
    stopwords = read_stopwords(stopwords_path)

    # 训练模型
    train_model(X, y, stopwords, model_name='pipeline')

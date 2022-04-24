# -*- coding: utf-8 -*-

import joblib

# 读取 pipeline
pipeline_path = '.\\results\\pipeline.model'
pipeline = joblib.load(pipeline_path)

def predict(message):
    """
    预测短信短信的类别和每个类别的概率
    :param message: 经过jieba分词的短信
    :return label: 整数类型，短信的类别，0(正常)/1(恶意)
    :return proba: 列表类型，短信属于每个类别的概率，[属于0的概率, 属于1的概率]
    """
    label = pipeline.predict([message])[0]
    proba = list(pipeline.predict_proba([message])[0])
    
    return label, proba

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    # 测试用例
    testdata_path='datasets\\sms_eval.csv'
    data_eval = pd.read_csv(testdata_path, encoding='utf-8')
    y_eval = np.array(data_eval['label'])
    X_eval = np.array(data_eval['msg_new'])

    total = y_eval.shape[0]
    count = 0
    for x, y in zip(X_eval, y_eval):
        y_pred, _ = predict(x)
        if y_pred == y:
            count += 1
    print('分数：{} / {}'.format(count, total))
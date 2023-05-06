import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def get_raw_data():
    global df_code2name
    # 读取所有文件
    df_risk = pd.read_excel(r"ALL_FDFDRISKINDCSI.xlsx", dtype={'基金代码_FdCd': 'object'})
    df_holder = pd.read_excel(r"ALL_FDHOLDERSTRU.xlsx", dtype={'基金代码_FdCd': 'object'})
    df_change = pd.read_excel(r"ALL_FDSHRCHG.xlsx", dtype={'基金代码_FdCd': 'object'})
    df_reward = pd.read_excel(r"ALL_REWARD.xlsx", dtype={'基金代码_FdCd': 'object'})
    df_code2name = pd.read_excel(r"code2name.xls", dtype={'基金代码_FdCd': 'object'})
    # 对文件进行合并生成源数据表
    # 合并源数据表
    d1 = df_risk.merge(df_holder, how='inner', on='基金代码_FdCd')
    d2 = d1.merge(df_change, how='inner', on='基金代码_FdCd')
    raw = d2.merge(df_reward, how='inner', on='基金代码_FdCd')
    raw = raw.drop_duplicates(subset=raw.columns[0], keep='first')  # 删除重复值
    temp = raw.dropna()  # 删除NaN值
    temp.to_excel('raw_data.xlsx', index=False)  # 保存合并后的源数据表


get_raw_data()


def isolation_forest_process():  # 使用孤立森林算法处理源数据，删除异常值
    global df
    df = pd.read_excel(r'raw_data.xlsx', dtype={'基金代码_FdCd': 'object'})
    # 切分数据
    df_even = df[df.index % 2 == 0]  # 偶数行
    df_odd = df[df.index % 2 == 1]  # 奇数行
    print(df_odd)
    print(df_even)
    # 生成训练数据
    rng = np.random.RandomState(89)  # 指定随机种子
    X_train = df_even.drop(columns=['基金代码_FdCd'])  # 删除基金代码列，防止基金代码对孤立森林算法产生影响
    X_test = df_odd.drop(columns=['基金代码_FdCd'])  # 删除基金代码列，防止基金代码对孤立森林算法产生影响
    # 拟合模型
    clf = IsolationForest(random_state=rng, contamination=0.04)  # 设定异常值比例为0.04
    clf.fit(X_train)  # 使用训练集进行模型训练
    # 预测
    y_predict_train = clf.predict(X_train)
    y_predict_test = clf.predict(X_test)
    # 合并预测结果
    predict_res = []
    if len(y_predict_test) == len(y_predict_train):  # 训练集与测试集一样大的时候直接合并
        for i in range(len(y_predict_test)):
            predict_res.append(y_predict_train[i])
            predict_res.append(y_predict_test[i])
    else:
        for i in range(len(y_predict_test)):  # 训练集与测试集不一样大的时候将训练集多出来的一个数据加入列表
            predict_res.append(y_predict_train[i])
            predict_res.append(y_predict_test[i])
        predict_res.append(y_predict_train[len(y_predict_test)])
    # 根据预测结果进行异常值舍弃
    for idx in range(len(predict_res)):
        if predict_res[idx] == -1:
            df.drop(index=idx, inplace=True)
    # 保存处理完异常值的表格
    df.to_excel('isolation_processed_data.xlsx', index=False, sheet_name='Sheet1')


isolation_forest_process()


def entropy_weight(data):
    data = np.array(data)  # Convert data to numpy array
    # 归一化
    p = data / data.sum(axis=0)

    # 计算熵值
    e = np.nansum(-p * np.log(p) / np.log(len(data)), axis=0)

    # 计算权系数
    return (1 - e) / (1 - e).sum()


# Function to perform TOPSIS analysis on given data
def topsis(data, weight=None):
    # 归一化
    data = data / np.sqrt((data ** 2).sum())

    # 最优最劣方案
    z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

    # Calculate distances
    weight = entropy_weight(data) if weight is None else np.array(weight)
    result = data.copy()
    result['正理想解'] = np.sqrt(((data - z.loc['正理想解']) ** 2 * weight).sum(axis=1))
    result['负理想解'] = np.sqrt(((data - z.loc['负理想解']) ** 2 * weight).sum(axis=1))

    # 综合得分指数
    result['综合得分指数'] = result['负理想解'] / (result['负理想解'] + result['正理想解'])
    result['排序'] = result.rank(ascending=False)['综合得分指数']

    return result, z, weight


# Function to calculate weight from given data
def get_weight(data):
    result = []
    for index in range(len(data)):
        result.append(data[index] / sum(data))
    return result


def get_all_kind_data():
    # List of consumer types
    people_kind = ['稳健型', '激进型', '短期型', '专业型',
                   '散户型', '拮据型', '富裕型', '长期型']

    # Weight matrix for different types of consumers
    weight_in = [[2, 1, 1, 1, 1], [1, 2, 2, 1, 1], [1, 2, 1, 1, 1], [1, 1, 1, 2, 1],
                 [2, 2, 2, 1, 2], [2, 2, 1, 2, 2], [1, 1, 2, 1, 1], [1, 1, 2, 1, 1]]

    # Calculate weights for each type of consumer
    weight = []

    for data in weight_in:
        weight.append(get_weight(data))

    # Generate predictions for each type of consumer
    for index in range(8):
        # Read data from the file
        print(f'Generating prediction for {people_kind[index]} consumer')
        data_in = df
        # Select relevant columns based on the consumer type
        if index < 6:
            _ = data_in[
                ['周均日beta值', '一周回报率(%)_RRInSinWk', '六个月回报率(%)_RRInSixMon', '从业人员持有率',
                 '基金份额变化(份)_ShrChg']]
            max_val = _['周均日beta值'].max()
            _['周均日beta值'] = max_val - _['周均日beta值']
        else:
            _ = data_in[
                ['半年均beta值', '一周回报率(%)_RRInSinWk', '六个月回报率(%)_RRInSixMon', '从业人员持有率',
                 '基金份额变化(份)_ShrChg']]
            max_val = _['半年均beta值'].max()
            _['半年均beta值'] = max_val - _['半年均beta值']
        # Perform TOPSIS analysis on the selected data using the calculated weight
        out, idiom, weight_list = topsis(_, weight[index])
        # Combine the results with the code of each fund
        sheet = pd.DataFrame(out)
        code = pd.DataFrame(data_in[['基金代码_FdCd']])
        res = pd.concat([code, sheet], axis=1)
        # Save the predictions to a file
        final = df_code2name.merge(res, how='inner')
        final.to_excel(f'for {people_kind[index]}.xlsx', index=False)


def get_main():
    data_in = df
    _ = data_in[
        ['半年均beta值', '一周回报率(%)_RRInSinWk', '六个月回报率(%)_RRInSixMon', '从业人员持有率',
         '基金份额变化(份)_ShrChg']]
    max_val = _['半年均beta值'].max()
    _['半年均beta值'] = max_val - _['半年均beta值']
    out, idiom, weight_list = topsis(_, )
    sheet = pd.DataFrame(out)
    code = pd.DataFrame(data_in[['基金代码_FdCd']])
    res = pd.concat([code, sheet], axis=1)
    final = df_code2name.merge(res, how='inner')
    # Save the predictions to a file
    final.to_excel(f'answer_for_all.xlsx', index=False)


get_main()

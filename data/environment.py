# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 05:41:28 2018

@author: Administrator
"""

# 导入必要的Python库
import numpy as np  # 数值计算库，用于数组操作和数学运算
import pandas as pd  # 数据分析库，用于处理表格数据和时间序列
from math import log  # 数学函数，用于计算对数（虽然导入但未使用）
from datetime import datetime  # 日期时间处理（虽然导入但未使用）
import time  # 时间相关操作（虽然导入但未使用）

# 模块作用：定义一个极小的常数，用于数值稳定性，防止除以零等数学错误
eps = 10e-8  # 极小正数，用于避免数值计算中的除零错误或对数计算错误


# 模块作用：辅助函数，为股票代码补零到6位格式（主要用于中国市场）
def fill_zeros(x):
    """
    将股票代码补零到6位长度，符合中国股市代码格式

    Args:
        x: 原始股票代码字符串，如'1'或'600001'

    Returns:
        补零后的6位股票代码字符串，如'000001'或'600001'
    """
    return '0' * (6 - len(x)) + x  # 在字符串前补零直到长度为6位


# 模块作用：强化学习环境类，模拟金融市场并提供状态、奖励、状态转移功能
# 论文对应：第3.1节"在模拟器中训练智能体"和第4节"实验设置"
class Environment:
    """
    投资组合管理强化学习环境

    负责：
    1. 加载和处理金融数据
    2. 生成状态表示（市场状态张量）
    3. 计算交易奖励（考虑交易成本）
    4. 管理环境状态转移
    """

    # 模块作用：环境类的构造函数，初始化整个强化学习环境
    def __init__(self, start_date, end_date, codes, features, window_length, market):
        """
        初始化环境，加载数据并构建状态空间

        Args:
            start_date: 开始日期字符串，如'2015-01-01'
            end_date: 结束日期字符串，如'2017-1-1'
            codes: 股票代码列表，"codes":["AAPL","ADBE","BABA","SNE","V"],
            features: 特征列表，如 ['close','high','low','open']
            window_length: 时间窗口长度整数，如 1
            market: 市场名称字符串，如'US'或'China'
        """

        # preprocess parameters
        self.cost = 0.0025  # 交易成本率，每次交易收取0.25%的费用

        # read all data
        # 读取市场数据文件，将第一列作为日期索引，自动解析日期格式
        data = pd.read_csv(r'./data/' + market + '.csv', index_col=0, parse_dates=True, dtype=object)
        data["code"] = data["code"].astype(str)  # 确保股票代码列为字符串类型

        # 如果是中国市场，对股票代码进行补零处理
        if market == 'China':
            data["code"] = data["code"].apply(fill_zeros)

        # 只保留指定的股票代码
        data = data.loc[data["code"].isin(codes)]
        # 将特征列转换为浮点数类型，便于数值计算
        data[features] = data[features].astype(float)

        # 生成有效时间范围
        # # 找到数据集中第一个大于指定开始日期的日期
        # start_date = [date for date in data.index if date > pd.to_datetime(start_date)][0]
        # # 找到数据集中最后一个小于指定结束日期的日期
        # end_date = [date for date in data.index if date < pd.to_datetime(end_date)][-1]
        # # 使用日期对象直接切片（推荐）
        # data = data[start_date:end_date]
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data[mask]



        # 初始化环境参数
        self.M = len(codes) + 1  # 资产数量 + 1（现金资产），M=股票数+现金【6】
        self.N = len(features)  # 特征数量，N=使用的特征个数【4】
        self.L = window_length  # 时间窗口长度，L=历史数据点数【1】

        # 为每一个资产生成独立的数据集
        asset_dict = dict()  # 创建空字典，键为股票代码，值为该股票的DataFrame
        datee = data.index.unique()  # 获取所有唯一的交易日期
        self.date_len = len(datee)  # 总日期数量，用于后续循环

        # 遍历每个股票代码，进行数据预处理
        for asset in codes:
            # 获取单个资产的数据，重新索引到所有日期（会产生缺失值）
            # data[data["code"]==asset] - 筛选出该股票的所有数据
            # .reindex(datee) - 重新索引到所有日期，缺失日期会产生NaN【有的日期没有这个股票】
            # .sort_index() - 按日期排序
            asset_data = data[data["code"] == asset].reindex(datee).sort_index()
            # 用前向填充法填充收盘价缺失值：用前一天的值填充当天的缺失值
            asset_data['close'] = asset_data['close'].fillna(method='pad')

            # 获取结束日的收盘价作为基准价格，进行价格归一化
            base_price = asset_data['close'].iloc[-1]  # 使用最后一个有效日期
            asset_dict[str(asset)] = asset_data  # 将处理后的数据存入字典
            # 将所有价格相对于基准价格进行归一化，使所有价格在1附近
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            # 对各个特征进行类似的归一化处理
            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low'] = asset_dict[str(asset)]['low'] / base_price

            if 'open' in features:
                asset_dict[str(asset)]['open'] = asset_dict[str(asset)]['open'] / base_price

            # # 处理基本面指标特征（市盈率）【中国市场】
            # if 'PE' in features:
            #     asset_data['PE'] = asset_data['PE'].fillna(method='pad')  # 填充市盈率缺失值
            #     base_PE = asset_data.ix[end_date, 'PE']  # 获取基准日市盈率
            #     asset_dict[str(asset)]['PE'] = asset_dict[str(asset)]['PE'] / base_PE  # 归一化
            #
            # # 处理基本面指标特征（市净率）
            # if 'PB' in features:
            #     asset_data['PB'] = asset_data['PB'].fillna(method='pad')  # 填充市净率缺失值
            #     base_PB = asset_data.ix[end_date, 'PB']  # 获取基准日市净率
            #     asset_dict[str(asset)]['PB'] = asset_dict[str(asset)]['PB'] / base_PB  # 归一化
            #
            # # 处理交易量相关特征
            # if 'TR' in features:
            #     asset_data['TR'] = asset_data['TR'].fillna(method='pad')  # 填充交易量缺失值
            #     base_TR = asset_data.ix[end_date, 'TR']  # 获取基准日交易量
            #     asset_dict[str(asset)]['TR'] = asset_dict[str(asset)]['TR'] / base_TR  # 归一化
            #
            # if 'TV1' in features:
            #     base_TV1 = asset_data.ix[end_date, 'TV1']  # 获取基准日TV1指标
            #     asset_dict[str(asset)]['TV1'] = asset_dict[str(asset)]['TV1'] / base_TV1  # 归一化
            #
            # if 'TV2' in features:
            #     base_TV2 = asset_data.ix[end_date, 'TV2']  # 获取基准日TV2指标
            #     asset_dict[str(asset)]['TV2'] = asset_dict[str(asset)]['TV2'] / base_TV2  # 归一化

            # 数据清理：先用后向填充，再用前向填充
            # fillna(method='bfill') - 用后面的值填充前面的缺失值
            # fillna(method='ffill') - 用前面的值填充后面的缺失值
            asset_data = asset_data.fillna(method='bfill', axis=1)
            asset_data = asset_data.fillna(method='ffill', axis=1)

            # 删除代码列，只保留特征数据
            asset_data = asset_data.drop(columns=['code'])
            asset_dict[str(asset)] = asset_data  # 更新字典中的数据集

        # 开始生成状态张量
        # 论文对应：第4节中描述的状态空间构建
        self.states = []  # 存储所有状态张量的列表
        y0 = np.array([[1], [1], [1], [1], [1], [1]])
        self.price_history = [y0]  # 存储价格历史用于计算回报的列表

        print("*-------------Now Begin To Generate Tensor---------------*")  # 提示开始生成张量

        t = self.L   # 从第L+1个时间点开始（因为需要L个历史数据点作为初始状态）【1】
        # 循环生成每个时间点的状态张量
        while t <= self.date_len: # 【1~504】
            # 初始化各个特征矩阵，第一行是现金资产（值始终为1）
            V_close = np.ones(self.L)  # 收盘价特征矩阵，现金资产始终为1

            # 根据选择的特征初始化对应的矩阵
            V_high = np.ones(self.L) if 'high' in features else None  # 最高价矩阵
            V_open = np.ones(self.L) if 'open' in features else None  # 开盘价矩阵
            V_low = np.ones(self.L) if 'low' in features else None  # 最低价矩阵
            # V_TV1 = np.ones(self.L) if 'TV1' in features else None  # TV1指标矩阵
            # V_TV2 = np.ones(self.L) if 'TV2' in features else None  # TV2指标矩阵
            # V_DA = np.ones(self.L) if 'DA' in features else None  # DA指标矩阵
            # V_TR = np.ones(self.L) if 'TR' in features else None  # 交易量矩阵
            # V_PE = np.ones(self.L) if 'PE' in features else None  # 市盈率矩阵
            # V_PB = np.ones(self.L) if 'PB' in features else None  # 市净率矩阵

            # 初始化价格变化向量（用于计算回报）
            y = np.ones(1)  # 现金资产的价格变化始终为1

            # 为每个资产填充特征数据
            for asset in codes:
                asset_data = asset_dict[str(asset)]  # 获取该资产的数据

                # 堆叠各个特征的历史数据
                # np.vstack() - 垂直堆叠数组
                # asset_data.iloc[t - self.L - 1:t - 1] - 获取从t-L-1到t-1的历史数据【窗口为1也可以直接取值】
                V_close = np.vstack((V_close, asset_data.iloc[t - self.L :t]['close'].values))

                # 类似地堆叠其他特征
                if 'high' in features:
                    V_high = np.vstack((V_high, asset_data.iloc[t - self.L :t]['high'].values))
                if 'low' in features:
                    V_low = np.vstack((V_low, asset_data.iloc[t - self.L :t ]['low'].values))
                if 'open' in features:
                    V_open = np.vstack((V_open, asset_data.iloc[t - self.L :t ]['open'].values))
                # if 'TV1' in features:
                #     V_TV1 = np.vstack((V_TV1, asset_data.iloc[t - self.L - 1:t - 1]['TV1'].values))
                # if 'TV2' in features:
                #     V_TV2 = np.vstack((V_TV2, asset_data.iloc[t - self.L - 1:t - 1]['TV2'].values))
                # if 'DA' in features:
                #     V_DA = np.vstack((V_DA, asset_data.iloc[t - self.L - 1:t - 1]['DA'].values))
                # if 'TR' in features:
                #     V_TR = np.vstack((V_TR, asset_data.iloc[t - self.L - 1:t - 1]['TR'].values))
                # if 'PE' in features:
                #     V_PE = np.vstack((V_PE, asset_data.iloc[t - self.L - 1:t - 1]['PE'].values))
                # if 'PB' in features:
                #     V_PB = np.vstack((V_PB, asset_data.iloc[t - self.L - 1:t - 1]['PB'].values))

                # 计算价格变化率：今日收盘价 / 昨日收盘价
                if t != 1:
                    y = np.vstack((y, asset_data.iloc[t-1]['close'] / asset_data.iloc[t - 2]['close']))

            # 构建状态张量：将各个特征堆叠在第三个维度
            state = V_close  # 以收盘价作为基础特征

            # 堆叠其他特征到状态张量中
            if 'high' and 'low' and 'open' in features:
                # np.stack() - 沿着新轴堆叠数组，axis=2表示在第三个维度堆叠
                state = np.stack((state, V_high, V_low, V_open), axis=2)

            # # 逐个堆叠其他特征
            # if 'TV1' in features and V_TV1 is not None:
            #     state = np.stack((state, V_TV1), axis=2)
            # if 'TV2' in features and V_TV2 is not None:
            #     state = np.stack((state, V_TV2), axis=2)
            # if 'DA' in features and V_DA is not None:
            #     state = np.stack((state, V_DA), axis=2)
            # if 'TR' in features and V_TR is not None:
            #     state = np.stack((state, V_TR), axis=2)
            # if 'PE' in features and V_PE is not None:
            #     state = np.stack((state, V_PE), axis=2)
            # if 'PB' in features and V_PB is not None:
            #     state = np.stack((state, V_PB), axis=2)

            # 重塑状态张量形状：[1, 资产数, 时间长度, 特征数]
            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)  # 保存状态到列表
            if t != self.L:
                self.price_history.append(y)  # 保存价格变化到列表
            t = t + 1  # 移动到下一个时间点
        # # 再最后冗余加L个，让最后一个状态也能利用上
        self.states.append(self.states[-1])
        # for i in range(self.L):
        #     self.states.append(self.states[-1])
        #     self.price_history.append(self.price_history[-1])
        self.reset()  # 初始化环境状态，准备开始训练
        # print(self.states)
        # print("——————————————————————————————")
        # print(self.price_history)
        # print("——————————————————————————————")

    # 模块作用：获取环境的第一个观测状态【多余，step中涵盖了】
    def first_ob(self):
        """
        获取第一个观测值（状态）
        在训练开始时调用，提供初始的市场状态

        Returns:
            第一个状态张量，形状为[1, M, L, N]
        """
        return self.states[0]

    def test_first_step(self ):
        info = {
            'reward': 1,  # 初始奖励为1
            'continue': 1,  # 继续交易
            'next state': self.states[self.t + 1],  # 第2个真实状态
            'weight vector': np.array([[1] + [0 for i in range(self.M - 1)]]),  # 100%现金
            # 'price': np.array([1,1,1,1,1]),  # 初始价格
            'price': np.squeeze(self.price_history[self.t]),
            'risk': 0  # 初始风险为0
        }
        return info


    # 模块作用：执行一步环境交互，计算奖励和状态转移
    def step(self, w1, w2):
        """
        执行一步环境交互，模拟一天的投资交易
        论文对应：第3.1节中描述的状态转移和奖励计算

        Args:
            w1: 当前投资组合权重 [1, M]，执行动作前的资产分配
            w2: 下一个投资组合权重 [1, M]，DRL代理建议的新资产分配

        Returns:
            info: 包含奖励、下一个状态等信息的字典，用于DRL学习
        """
        if self.FLAG:  # 正常交易模式（非初始状态）
            not_terminal = 1  # 是否终止标志，1表示继续，0表示回合结束
            price = self.price_history[self.t]  # 当前价格向量，包含所有资产的价格变化率

            # 计算交易成本：成本率 × 权重变化绝对值之和
            # 只计算股票部分的交易成本（排除现金）
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            # 计算风险：资产波动率的加权和
            # 计算前一个状态的标准差作为风险度量
            std = self.states[self.t - 1][0].std(axis=(1, 2)).reshape(-1)
            # 计算新权重的风险暴露
            w2_std = (w2[0] * std).sum()

            # 添加风险惩罚项（gamma控制风险厌恶程度）
            gamma = 0.00  # 风险惩罚系数，为0表示不考虑风险惩罚
            risk = gamma * w2_std  # 风险惩罚值

            # 计算原始回报：新权重下的收益 - 交易成本
            # np.dot(w2, price) - 新权重下的投资组合收益
            r = (np.dot(w2, price)[0] - mu)[0]

            # 计算最终奖励：取对数回报（更稳定的优化目标）
            # reward = np.log(r + eps)  # 加eps防止对数计算错误
            # reward = np.log(max(r + eps, eps))  # 确保参数大于0
            reward = r

            # 权重更新：根据价格变化调整权重（市值加权调整）
            w2 = w2 / (np.dot(w2, price) + eps)


            # 检查是否到达终止状态
            if self.t == len(self.states) - 2:
                not_terminal = 0  # 设置为终止状态


            # 准备返回信息
            price = np.squeeze(price)  # 压缩价格向量维度，从2D变为1D
            info = {
                'reward': reward,  # 当前步的奖励值
                'continue': not_terminal,  # 是否继续标志
                'next state': self.states[self.t+1],  # 下一个状态
                'weight vector': w2,  # 调整后的权重
                'price': price,  # 当前价格向量
                'risk': risk  # 风险值
            }

            self.t += 1  # 时间步前进，移动到下一个交易日
            if self.t == len(self.states) - 1:
                self.reset()  # 重置环境

            return info
        else:
            # 初始状态：全部持有现金（第一次调用step时）
            info = {
                'reward': 1,  # 初始奖励为1
                'continue': 1,  # 继续交易
                'next state': self.states[self.t+1],  # 第2个真实状态
                'weight vector': np.array([[1] + [0 for i in range(self.M - 1)]]),  # 100%现金
                # 'price': np.array([1,1,1,1,1]),  # 初始价格
                'price': np.squeeze(self.price_history[self.t]) ,
                'risk': 0  # 初始风险为0
            }
            self.FLAG = True  # 切换到正常交易模式
            self.t += 1
            return info

    # 模块作用：重置环境状态，开始新的训练回合
    def reset(self):
        """
        重置环境状态
        论文对应：开始新的训练回合或测试回合

        将环境重置到初始状态，用于开始新的训练回合
        """
        self.t = 0   # 重置时间指针到第一个有效状态位置
        self.FLAG = False  # 重置标志位，表示处于初始状态
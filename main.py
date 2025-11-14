# -*- coding: utf-8 -*-
# 编码声明：指定文件使用UTF-8编码，确保中文字符正常显示

import sys

# Python语法：sys是系统相关的参数和函数模块
# 作用：将上级目录添加到Python路径的开头，这样Python就能找到上级目录中的模块
# sys.path.insert(0, '../')

from argparse import ArgumentParser
import json
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import csv
import os

# 全局变量定义
# eps：一个极小的正数，用于数值稳定性（防止除以零等数学错误）
eps = 10e-8
# epochs：训练的总回合数，初始为0
epochs = 0
# M：资产数量（股票数+现金），初始为0
M = 0

# 创建必要的目录
# os.makedirs：递归创建目录，exist_ok=True表示如果目录已存在也不报错
# 作用：确保保存神经网络权重的目录存在
os.makedirs('./saved_network/PPO', exist_ok=True)


# 模块作用：股票交易员类，负责记录交易结果、管理投资组合状态
# 论文对应：第4节"实验"中记录交易结果和性能评估
class StockTrader():
    """
    股票交易员类，负责：
    1. 记录投资组合的财富变化
    2. 计算和记录奖励
    3. 保存交易历史用于分析
    4. 处理动作噪声（探索）
    """
    # 初始化股票交易员
    def __init__(self):
        """
        初始化股票交易员
        调用reset方法设置初始状态
        """
        self.reset()

    # 模块作用：重置交易员状态，开始新的交易回合
    def reset(self):
        """
        重置交易员到初始状态

        初始化所有记录变量，为新的训练回合做准备
        """
        # 初始财富：10,000元
        self.wealth = 1e4
        # 累计奖励
        self.total_reward = 1
        # 平均最大Q值（用于监控学习状态）
        self.ep_ave_max_q = 0
        # 累计损失
        self.loss = 0
        # Actor网络损失
        self.actor_loss = 0

        # 历史记录列表
        self.wealth_history = [self.wealth]  # 财富历史
        self.r_history = []  # 回报历史
        self.w_history = []  # 权重历史
        self.p_history = []  # 价格历史

        # Ornstein-Uhlenbeck噪声，用于动作探索
        # 论文对应：在连续动作空间中添加相关性噪声进行探索
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    # 模块作用：更新交易摘要信息，记录每一步的交易结果
    def update_summary(self, loss, r, q_value, actor_loss, w, p):
        """
        更新交易摘要信息

        Args:
            loss: 当前步的损失值
            r: 当前步的回报（对数回报）
            q_value: Q值估计
            actor_loss: Actor网络损失
            w: 投资组合权重
            p: 资产价格
        """
        # 累加损失
        self.loss += loss
        # 累加Actor损失
        self.actor_loss += actor_loss
        # 累加回报
        self.total_reward *= r
        # print("r",r)
        # print("total_reward:",self.total_reward)
        # 累加Q值
        self.ep_ave_max_q += q_value
        # 记录当前回报
        self.r_history.append(r)
        # 更新财富：使用指数回报计算新财富
        # 论文对应：财富增长的计算公式
        # self.wealth = self.wealth * math.exp(r)
        self.wealth = self.wealth * r
        # print(self.wealth)

        # 记录财富历史
        self.wealth_history.append(self.wealth)

        # 权重记录 - 明确四舍五入到2位小数
        # 使用round函数进行四舍五入
        self.w_history.extend([','.join([
            f"{round(w0, 2):.2f}"  # 先四舍五入再格式化
            for w0 in w.tolist()[0]
        ])])

        # 价格记录 - 明确四舍五入到3位小数
        self.p_history.extend([','.join([
            f"{round(p0, 3):.3f}"  # 先四舍五入再格式化
            for p0 in p.tolist()
        ])])

    # 模块作用：将交易结果写入CSV文件
    def write(self, epoch):
        """
        将交易历史写入CSV文件

        Args:
            epoch: 当前训练回合数，用于文件名
        """
        # 将列表转换为pandas Series对象
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)

        # 合并所有历史数据
        history = pd.concat([wealth_history, p_history, w_history, r_history], axis=1)
        # 设置列名
        history.columns = ['Wealth', 'Price', 'Yesterday-Weight', 'Return']
        # 写入CSV文件，文件名包含回合数和总回报
        if epoch==-1:
            history.to_csv(f'{RESULT_FOLDER}/test-return-{self.total_reward * 100:.2f}%.csv')
        else:
            history.to_csv(f'{RESULT_FOLDER}/train/epoch{epoch}-return-{self.total_reward * 100:.2f}%.csv')

    # 模块作用：打印训练结果并保存模型
    def print_result(self, epoch, agent):
        """
        打印回合结果并保存模型

        Args:
            epoch: 当前回合数
            agent: DRL代理对象
        """
        # 将累计对数回报转换为百分比回报
        self.total_reward = self.total_reward * 100
        # 打印结果
        print('*-----Episode: {:d}, Reward:{:.3f}%,  ep_ave_max_q:{:.2f}, actor_loss:{:2f}, critic_loss:{:2f}-----*'.format(epoch,
                                                                                                         self.total_reward,
                                                                                                         self.ep_ave_max_q,
                                                                                                         self.actor_loss,
                                                                                                         self.loss))
        # 写入TensorBoard摘要
        agent.write_summary(self.loss, self.total_reward, self.ep_ave_max_q, self.actor_loss, epoch)
        # 保存模型
        agent.save_model(epoch)

    # 模块作用：绘制财富变化曲线
    def plot_result(self):
        """
        绘制财富历史曲线
        """
        # 将财富历史转换为pandas Series并绘图
        pd.Series(self.wealth_history).plot()
        # 显示图形
        plt.show()

    # 模块作用：处理动作，添加噪声进行探索
    # 论文对应：在训练过程中通过噪声进行探索
    def action_processor(self, a, ratio):
        """
        处理动作，添加探索噪声

        Args:
            a: 原始动作（投资组合权重）
            ratio: 噪声比例，随训练进行逐渐减小

        Returns:
            处理后的动作
        """
        # 添加噪声并裁剪到[0,1]范围
        # print(ratio)
        a = np.clip(a + self.noise() * ratio, 0, 1)
        # 重新归一化，确保权重和为1
        a = a / (a.sum() + eps)
        return a


# 模块作用：解析环境返回的信息字典
def parse_info(info):
    """
    解析环境返回的信息字典

    Args:
        info: 环境返回的信息字典

    Returns:
        分解后的各个信息组件
    """
    return info['reward'], info['continue'], info['next state'], info['weight vector'], info['price'], info['risk']


# 模块作用：保存状态-动作对到CSV文件，用于后续的可解释性分析
# 论文对应：第3.3节"保存状态-动作对用于事后解释"
def save_state_actions(filename, states, actions):
    """
    保存状态和动作到CSV文件

    Args:
        filename: 输出文件名
        states: 状态列表
        actions: 动作列表
    """
    # 清理状态字符串：移除换行符和多余空格
    states = [str(state).replace('\n', '').replace('   ', ' ').replace('  ', ' ').replace('. ', '.0') for state in
              states]
    # 清理动作字符串
    actions = [str(action).replace('\n', '').replace('  ', ' ') for action in actions]

    # 写入CSV文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['State', 'Action'])
        indexs = list(range(len(states)))  # 增加索引列
        # 逐行写入状态-动作对
        for index, state, action in zip(indexs ,states, actions):
            writer.writerow([index, state, action])


# 模块作用：遍历一个完整的交易回合，执行训练或测试
# 论文对应：第4节中描述的完整交易流程
def traversal(stocktrader, agent, env, epoch, noise_flag, framework, method, trainable):
    """
    遍历一个完整的交易回合

    Args:
        stocktrader: 股票交易员实例
        agent: DRL代理实例
        env: 环境实例
        epoch: 当前回合数
        noise_flag: 是否添加噪声的标志
        framework: 框架类型（DDPG/PPO）
        method: 训练方法
        trainable: 是否可训练
    """
    # 初始化记录列表
    states = []  # 状态记录
    actions = []  # 动作记录

    # 记录状态和动作（用于可解释性分析）【状态=>动作】
    states.append(env.first_ob())
    w2 = agent.predict(env.first_ob())
    actions.append(w2)

    # 获取初始状态信息
    info = env.step(None, None)
    # 解析信息
    r, contin, s, w1, p, risk = parse_info(info)



    # 更新交易摘要
    stocktrader.update_summary(0, r, 0, 0, w1, p)

    w1 = w2 # w1存的是上一个动作

    # 主循环：直到回合结束
    while contin:
        # 使用当前状态预测动作（投资组合权重）
        w2 = agent.predict(s)

        # NAN bug？？？
        # 检查预测结果是否为NaN
        if np.any(np.isnan(w2)) or np.any(np.isinf(w2)):
            print("🚨 预测到NaN权重，使用均匀分布并跳过本轮训练")
            w2 = np.ones((1, M)) / M  # 使用均匀权重
            # 可以选择重置网络或跳过训练
            if trainable == "True":
                print("🔄 检测到NaN，清空经验缓冲区")
                agent.buffer = []  # 清空无效经验
                break  # 跳出当前回合

        # 记录状态和动作（用于可解释性分析）【状态=>动作】
        states.append(s)
        actions.append(w2)

        # 如果启用噪声，处理动作
        if noise_flag == 'True':
            # 随着训练进行，噪声比例逐渐减小
            # w2 = stocktrader.action_processor(w2, (epochs - epoch) / epochs) # 直线下降
            w2 = stocktrader.action_processor(w2, math.exp(-100 * epoch / epochs)) # 曲线下降（数字越大降得越快）【不妨取10~1000】

        # 在环境中执行动作，获取新信息
        env_info = env.step(w1, w2)
        # 解析新信息
        r, contin, s_next, w2, p, risk = parse_info(env_info)

        # 保存状态转移经验（用于训练）
        # 论文对应：第3.3节"保存状态-动作对"
        agent.save_transition(s, w2, r - 0)

        # 初始化训练变量
        loss, q_value, actor_loss = 0, 0, 0

        # DDPG框架的训练逻辑
        if framework == 'DDPG':
            if trainable == "True":  # 如果可训练
                # 训练代理
                agent_info = agent.train(method, epoch)
                # 获取训练信息
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        # PPO框架的训练逻辑
        elif framework == 'PPO':
            # PPO在回合结束时才进行训练
            if not contin and trainable == "True":
                agent_info = agent.train(method, epoch)
                actor_loss,loss, q_value = agent_info["actor_loss"],agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        # 更新交易摘要
        stocktrader.update_summary(loss, r, q_value, actor_loss, w1, p)
        w1=w2
        # 更新状态
        s = s_next

    # 保存状态-动作对到文件（用于可解释性分析）
    # 论文对应：为SHAP/LIME分析提供数据【每次训练/测试都会覆盖文件，保留的是最新的状态-行为对】
    if epoch == -1:
        save_state_actions(f"{RESULT_FOLDER}/test-state_actions.csv", states, actions)
    save_state_actions(f"{RESULT_FOLDER}/state_actions/epoch{epoch}-state_actions.csv", states, actions)


# 模块作用：解析配置文件参数
def     parse_config(config, mode):
    """
    解析配置文件参数

    Args:
        config: 配置字典
        mode: 运行模式（train/test）

    Returns:
        所有解析后的配置参数
    """
    # 从配置中提取参数
    codes = config["session"]["codes"]  # 股票代码列表
    start_date = config["session"]["start_date"]  # 开始日期
    end_date = config["session"]["end_date"]  # 结束日期
    features = config["session"]["features"]  # 使用的特征
    agent_config = config["session"]["agents"]  # 代理配置
    market = config["session"]["market_types"]  # 市场类型
    # 各种标志
    noise_flag, record_flag, plot_flag = config["session"]["noise_flag"], config["session"]["record_flag"], \
    config["session"]["plot_flag"]

    # 代理相关参数
    predictor, framework, window_length = agent_config
    global RESULT_FOLDER
    RESULT_FOLDER = f'{market}-result(depend-{window_length}-days)'
    os.makedirs(f'./{RESULT_FOLDER}/train', exist_ok=True)
    os.makedirs(f'./{RESULT_FOLDER}/state_actions', exist_ok=True)
    reload_flag, trainable = config["session"]['reload_flag'], config["session"]['trainable']
    method = config["session"]['method']

    # 更新全局变量
    global epochs
    epochs = int(config["session"]["epochs"])

    # 测试模式下的参数覆盖
    if mode == 'test':
        record_flag = 'True'  # 强制记录
        noise_flag = 'False'  # 关闭噪声
        plot_flag = 'True'  # 强制绘图
        reload_flag = 'True'  # 强制重载模型
        trainable = 'False'  # 不可训练
        method = 'model_free'  # 使用无模型方法

    # 打印训练状态信息
    print("*--------------------Training Status-------------------*")
    print('Codes:', codes)
    print("Date from", start_date, ' to ', end_date)
    print('Features:', features)
    print("Agent:Noise(", noise_flag, ')---Recoed(', noise_flag, ')---Plot(', plot_flag, ')')
    print("Market Type:", market)
    print("Predictor:", predictor, "  Framework:", framework, "  Window_length:", window_length)
    print("Epochs:", epochs)
    print("Trainable:", trainable)
    print("Reloaded Model:", reload_flag)
    print("Method", method)
    print("Noise_flag", noise_flag)
    print("Record_flag", record_flag)
    print("Plot_flag", plot_flag)

    # 返回所有参数
    return codes, start_date, end_date, features, agent_config, market, predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method


# 模块作用：主会话函数，协调整个训练或测试过程
def session(config, mode):
    """
    主会话函数，协调训练或测试过程

    Args:
        config: 配置字典
        mode: 运行模式
    """
    # 动态导入Environment类（避免循环导入）
    from data.environment import Environment
    # 解析配置
    codes, start_date, end_date, features, agent_config, market, predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method = parse_config(
        config, mode)
    # 创建环境实例
    env = Environment(start_date, end_date, codes, features, int(window_length), market)

    # 更新全局资产数量
    global M
    M = len(codes) + 1

    # 根据框架类型加载对应的代理
    pre = 0
    if framework == 'DDPG':
        print("*-----------------Loading DDPG Agent---------------------*")
        from agents.ddpg import DDPG
        agent = DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,
                     trainable)

    elif framework == 'PPO':
        print("*-----------------Loading PPO Agent---------------------*")
        from agents.ppo import PPO
        agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,
                    trainable)
    elif framework == 'PPO_PLUS':
        print("*-----------------Loading PPO Agent---------------------*")
        framework ='PPO'
        pre = 1
        from agents.ppo_plus import PPO
        agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,
                    trainable)

    # 创建股票交易员
    stocktrader = StockTrader()

    # 训练模式
    if mode == 'train':
        if pre == 1:
            print("🎯 开始Critic预训练阶段(对奖励值有个大概估计)...")
            pretrain_epochs = 1  # 预训练1个epoch

            for pretrain_epoch in range(pretrain_epochs):
                print(f"Critic预训练 Epoch {pretrain_epoch + 1}/{pretrain_epochs}")
                s = env.first_ob()
                w2 = agent.predict(env.first_ob())
                r = env.test_first_step()["reward"]
                agent.save_transition(s, w2, r-1)
                agent.train(method, -5)



        print("Training with {:d}".format(epochs))
        # 遍历每个训练回合
        for epoch in range(epochs):
            print("Now we are at epoch", epoch)
            # 执行一个完整回合
            traversal(stocktrader, agent, env, epoch, noise_flag, framework, method, trainable)

            # 如果启用记录，写入结果
            if record_flag == 'True':
                stocktrader.write(epoch)

            # 如果启用绘图，显示结果
            if plot_flag == 'True':
                stocktrader.plot_result()

            # 打印结果并保存模型
            stocktrader.print_result(epoch, agent)
            # 重置交易员状态
            stocktrader.reset()

    # 测试模式
    elif mode == 'test':
        # 执行一个测试回合
        traversal(stocktrader, agent, env, -1, noise_flag, framework, method, trainable)
        # 写入结果
        stocktrader.write(-1)
        # 显示图形
        stocktrader.plot_result()
        # 打印结果
        stocktrader.print_result(-1, agent)


# 模块作用：构建命令行参数解析器【只使用两个：python main.py --mode=train/test】
def build_parser():
    """
    构建命令行参数解析器

    Returns:
        配置好的参数解析器
    """
    parser = ArgumentParser(
        description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    # 添加模式参数
    parser.add_argument("--mode", dest="mode", help="download(China), train, test", metavar="MODE", default="train",
                        required=True)
    # 添加模型参数
    parser.add_argument("--model", dest="model", help="DDPG,PPO", metavar="MODEL", default="PPO", required=False)
    return parser


# 模块作用：程序主函数，程序入口点
def main():
    """
    程序主函数，整个程序的入口点
    """
    # 构建参数解析器
    parser = build_parser()
    # 解析命令行参数并转换为字典
    args = vars(parser.parse_args())
    print(args)

    # 读取配置文件
    with open('config.json') as f:
        config = json.load(f)
        # 如果是下载模式【数据已经下好了，不执行这个】
        if args['mode'] == 'download':
            from data.download_data import DataDownloader
            data_downloader = DataDownloader(config)
            data_downloader.save_data()
        else:
            # 否则运行训练或测试会话
            session(config, args['mode'])


# Python语法：如果这个文件是直接运行的（不是被导入的），则执行main函数
if __name__ == "__main__":
    main()
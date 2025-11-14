"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
一个单线程实现的简化版近端策略优化算法

论文对应：第3.1节中描述的PPO算法实现
"""

import tensorflow as tf  # 深度学习框架
import numpy as np       # 数值计算库
import json             # JSON数据处理
import time             # 时间相关操作
import math             # 数学函数
import pandas as pd     # 数据分析库
from argparse import ArgumentParser  # 命令行参数解析

# # 超参数定义
# EP_MAX = 1000      # 最大训练回合数【epoch】
# # EP_LEN = 500       # 每个回合的最大步数【每个epoch训练多少条数据，有503条就多训点呗】
# GAMMA = 0.9        # 折扣因子，平衡即时奖励和未来奖励
# A_LR = 10e-4       # Actor网络学习率
# C_LR = 10e-4       # Critic网络学习率
# BATCH = 256         # 批处理大小
# A_UPDATE_STEPS = 20  # Actor网络更新步数【同样的数据反复训练10次】
# C_UPDATE_STEPS = 20  # Critic网络更新步数
# 超参数定义 - 金融优化版【改进0】
GAMMA = 0.97        # 金融数据需要更长期的视角
A_LR = 9e-4         # 更保守的学习率
C_LR = 9e-4         # Critic学习率稍低
A_UPDATE_STEPS = 7   # 金融数据噪声大，减少更新步数
C_UPDATE_STEPS = 13   # Critic可以多更新几次
CLIP_EPSILON = 0.17   # 更严格的策略裁剪

# PPO优化方法选择
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL散度惩罚方法
    dict(name='clip', epsilon=0.2),                 # 裁剪替代目标方法（论文使用的方法）
][1]        # 选择第二种方法（裁剪方法），论文中表现更好

# 模块作用：构建卷积神经网络层
# def con2d(x, scope, trainable):
#     """
#     构建卷积神经网络层，用于提取状态特征
#
#     Args:
#         x: 输入张量，形状为 [batch_size, M, L, N]
#         scope: 变量作用域名称
#         trainable: 是否可训练
#
#     Returns:
#         out: 卷积层输出
#     """
#     with tf.variable_scope(scope):
#        # 第一层卷积层
#         # tf.truncated_normal: 截断正态分布初始化权重
#         # [1, 3, int(x.shape[3]), 2]: [高度, 宽度, 输入通道数, 输出通道数]
#         con_W_1 = tf.Variable(tf.truncated_normal([1, 3, int(x.shape[3]), 2], stddev=0.5), trainable=trainable)
#         # tf.nn.conv2d: 2D卷积操作，padding='SAME'保持输出尺寸不变
#         layer = tf.nn.conv2d(x, con_W_1, padding='SAME', strides=[1, 1, 1, 1])  # 使用'SAME'填充
#         # 批归一化层，加速训练和提高稳定性
#         norm = tf.layers.batch_normalization(layer, training=trainable)
#         # ReLU激活函数，引入非线性
#         x = tf.nn.relu(norm)
#
#         # 第二层卷积层
#         con_W_2 = tf.Variable(tf.truncated_normal([1, 3, int(x.shape[3]), 48], stddev=0.5), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_2, padding='SAME', strides=[1, 1, 1, 1])  # 使用'SAME'填充
#         norm = tf.layers.batch_normalization(layer, training=trainable)
#         x = tf.nn.relu(norm)
#
#         # 第三层卷积层
#         con_W_3 = tf.Variable(tf.truncated_normal([1, 3, 48, 1], stddev=0.5), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_3, padding='SAME', strides=[1, 1, 1, 1])  # 使用'SAME'填充
#         norm = tf.layers.batch_normalization(layer, training=trainable)
#         out = tf.nn.relu(norm)
#
#     return out
# 【改进1】
def con2d(x, scope, trainable):
    """
    增强的CNN特征提取
    """
    with tf.variable_scope(scope):
        # 第一层：更大的卷积核捕获长期模式
        con_W_1 = tf.Variable(tf.truncated_normal([1, 7, int(x.shape[3]), 72], stddev=0.5), trainable=trainable)
        layer = tf.nn.conv2d(x, con_W_1, padding='SAME', strides=[1, 1, 1, 1])
        norm = tf.layers.batch_normalization(layer, training=trainable)
        x = tf.nn.leaky_relu(norm, alpha=0.1)  # 改用LeakyReLU

        # 第二层：注意力机制
        con_W_2 = tf.Variable(tf.truncated_normal([1, 5, 72, 72], stddev=0.5), trainable=trainable)
        layer = tf.nn.conv2d(x, con_W_2, padding='SAME', strides=[1, 1, 1, 1])
        norm = tf.layers.batch_normalization(layer, training=trainable)
        x = tf.nn.leaky_relu(norm, alpha=0.1)


        con_W_2copy = tf.Variable(tf.truncated_normal([1, 3, 72, 24], stddev=0.5), trainable=trainable)
        layer = tf.nn.conv2d(x, con_W_2copy, padding='SAME', strides=[1, 1, 1, 1])
        norm = tf.layers.batch_normalization(layer, training=trainable)
        x = tf.nn.leaky_relu(norm, alpha=0.1)

        # 第三层：特征压缩
        con_W_3 = tf.Variable(tf.truncated_normal([1, 3, 24, 1], stddev=0.5), trainable=trainable)
        layer = tf.nn.conv2d(x, con_W_3, padding='SAME', strides=[1, 1, 1, 1])
        norm = tf.layers.batch_normalization(layer, training=trainable)
        out = tf.nn.leaky_relu(norm, alpha=0.1)

        # 全局平均池化替代展平
        # out = tf.reduce_mean(out, axis=2)  # 在时间维度上平均池化

    return out
# def con2d(x, scope, trainable):
#     """
#     深度可分离卷积 + 大感受野
#     """
#     with tf.variable_scope(scope):
#         # 深度可分离卷积 - 深度卷积
#         depthwise_W = tf.Variable(tf.truncated_normal([1, 7, int(x.shape[3]), 1], stddev=0.1), trainable=trainable)
#         depthwise = tf.nn.depthwise_conv2d(x, depthwise_W, strides=[1, 1, 1, 1], padding='SAME')
#         depthwise = tf.layers.batch_normalization(depthwise, training=trainable)
#         depthwise = tf.nn.leaky_relu(depthwise, alpha=0.1)
#
#         # 逐点卷积
#         pointwise_W = tf.Variable(tf.truncated_normal([1, 1, int(x.shape[3]), 64], stddev=0.1), trainable=trainable)
#         pointwise = tf.nn.conv2d(depthwise, pointwise_W, strides=[1, 1, 1, 1], padding='SAME')
#         pointwise = tf.layers.batch_normalization(pointwise, training=trainable)
#         x = tf.nn.leaky_relu(pointwise, alpha=0.1)
#
#         # 标准卷积层
#         con_W_2 = tf.Variable(tf.truncated_normal([1, 5, 64, 32], stddev=0.1), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_2, padding='SAME', strides=[1, 1, 1, 1])
#         layer = tf.layers.batch_normalization(layer, training=trainable)
#         x = tf.nn.leaky_relu(layer, alpha=0.1)
#
#         # 输出层
#         con_W_3 = tf.Variable(tf.truncated_normal([1, 3, 32, 1], stddev=0.1), trainable=trainable)
#         out = tf.nn.conv2d(x, con_W_3, padding='SAME', strides=[1, 1, 1, 1])
#         out = tf.layers.batch_normalization(out, training=trainable)
#         out = tf.nn.leaky_relu(out, alpha=0.1)
#
#     return out

# 模块作用：构建全连接层
def dense(x, out_dim, activation, scope, trainable):
    """
    构建全连接层

    Args:
        x: 输入张量
        out_dim: 输出维度
        activation: 激活函数类型
        scope: 变量作用域名称
        trainable: 是否可训练

    Returns:
        out: 全连接层输出
    """
    with tf.variable_scope(scope):
        # 权重矩阵初始化
        t1_w = tf.Variable(tf.truncated_normal([int(x.shape[1]), out_dim], stddev=0.1), trainable=trainable)
        # 偏置项初始化
        t1_b = tf.Variable(tf.constant(0.1, shape=[out_dim]), trainable=trainable)
        # 线性变换: Wx + b
        out = tf.matmul(x, t1_w) + t1_b

        # 应用激活函数
        if activation == 'relu':
            out = tf.nn.relu(out)
        elif activation == 'tanh':
            out = tf.nn.tanh(out)      # 输出范围[-1,1]，适合动作输出
        elif activation == 'softplus':
            out = tf.nn.softplus(out)  # 输出范围(0,∞)，适合标准差
        elif activation == 'sigmoid':
            out = tf.nn.sigmoid(out)   # 输出范围[0,1]
        else:
            print('fail to build up')
    return out

# 模块作用：构建TensorBoard摘要操作
def build_summaries():
    """
    构建训练过程的可视化摘要

    Returns:
        summary_ops: 摘要操作
        summary_vars: 摘要变量列表
    """
    # 定义摘要变量
    critic_loss = tf.Variable(0.)
    reward = tf.Variable(0.)
    ep_ave_max_q = tf.Variable(0.)
    actor_loss = tf.Variable(0.)

    # 创建标量摘要
    tf.summary.scalar('Critic_loss', critic_loss)
    tf.summary.scalar('Reward', reward)
    tf.summary.scalar('Ep_ave_max_q', ep_ave_max_q)
    tf.summary.scalar('Actor_loss', actor_loss)

    summary_vars = [critic_loss, reward, ep_ave_max_q, actor_loss]
    summary_ops = tf.summary.merge_all()  # 合并所有摘要
    return summary_ops, summary_vars

# 模块作用：PPO算法主类
class PPO(object):
    """
    近端策略优化算法实现
    论文对应：第3.1节中描述的PPO算法
    """

    # 模块作用：PPO类构造函数
    def __init__(self, predictor, M, L, N, name, load_weights, trainable):
        """
        初始化PPO算法

        Args:
            predictor: 预测器（未使用）
            M: 资产数量（股票数+现金）
            L: 时间窗口长度
            N: 特征数量
            name: 模型名称
            load_weights: 是否加载预训练权重
            trainable: 是否可训练
        """
        self.sess = tf.Session()  # 创建TensorFlow会话
        # 状态占位符：[批次大小, 资产数, 时间窗口, 特征数]
        self.tfs = tf.placeholder(tf.float32, [None, M, L, N], 'state')
        self.name = name

        # 环境参数
        self.M = M  # 资产数量
        self.L = L  # 时间窗口长度
        self.N = N  # 特征数量

        self.gamma = 0.99  # 折扣因子

        # Critic网络：评估状态价值
        with tf.variable_scope('critic'):
            l1 = con2d(self.tfs, 'critic', True)[:, :, 0, 0]

            # 🔥 增强Critic网络
            l2 = dense(l1, 200, 'relu', 'l2', True)
            l3 = dense(l2, 100, 'relu', 'l3', True)

            # 🔥 移除tanh限制，让Critic自由学习
            self.v = dense(l3, 1, None, 'v_output', True)  # 线性输出

            # 折扣回报占位符
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            # 优势函数：A(s,a) = Q(s,a) - V(s)
            self.advantage = self.tfdc_r - self.v
            # Critic损失：均方误差
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # # 优化操作：使用指数衰减学习率
            # global_step = tf.Variable(0, trainable=False)
            # C_learning_rate = tf.train.exponential_decay(C_LR, global_step,
            #                                            decay_steps=2000,
            #                                            decay_rate=0.9, staircase=False)
            # self.ctrain_op = tf.train.GradientDescentOptimizer(C_learning_rate).minimize(self.closs, global_step=global_step)
            # 优化操作：Adam + 梯度裁剪【改进2】
            global_step = tf.Variable(0, trainable=False)
            C_learning_rate = tf.train.exponential_decay(C_LR, global_step,
                                                         decay_steps=2000,
                                                         decay_rate=0.9, staircase=False)
            self.ctrain_op = tf.train.GradientDescentOptimizer(C_learning_rate).minimize(self.closs,
                                                                                         global_step=global_step)
            optimizer = tf.train.AdamOptimizer(C_learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.closs)
            clipped_grads = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in grads_and_vars]
            self.ctrain_op = optimizer.apply_gradients(clipped_grads)

        # Actor网络：生成动作策略
        # 当前策略网络
        pi, pi_params = self._build_anet('pi', trainable=True)
        # 旧策略网络（用于计算重要性采样比率）
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        with tf.variable_scope('sample_action'):
            # 采样动作：从策略分布中采样
            self.sample_op = pi.sample(1)[0]  # 采样一个动作

        with tf.variable_scope('update_oldpi'):
            # 更新旧策略：将当前策略参数复制到旧策略
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # 动作和优势占位符
        self.tfa = tf.placeholder(tf.float32, [None, self.M], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # 重要性采样比率：π(a|s) / π_old(a|s)
                # ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                # 替代目标函数
                surr = ratio * self.tfadv

            if METHOD['name'] == 'kl_pen':
                # KL散度惩罚方法
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)  # 计算KL散度
                self.kl_mean = tf.reduce_mean(kl)
                # Actor损失：带KL约束的目标函数
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # 裁剪方法（论文使用的方法）
                # 裁剪替代目标，防止策略更新过大
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,  # 原始替代目标
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv  # 裁剪后的目标
                ))


        with tf.variable_scope('atrain'):
            # Actor优化器
            A_learning_rate = tf.train.exponential_decay(A_LR, global_step,
                                                         decay_steps=2000,
                                                         decay_rate=0.9, staircase=False)
            self.atrain_op = tf.train.GradientDescentOptimizer(A_learning_rate).minimize(self.aloss)
            # Actor优化器 - Adam + 梯度裁剪【改进3】
            # Actor优化器 - 完整的Adam + 梯度裁剪
            # A_learning_rate = tf.train.exponential_decay(A_LR, global_step,
            #                                              decay_steps=2000,
            #                                              decay_rate=0.9, staircase=False)

            # # 🔥 修复：使用正确的优化器配置
            # optimizer = tf.train.AdamOptimizer(
            #     learning_rate=A_learning_rate,
            #     beta1=0.9,
            #     beta2=0.999,
            #     epsilon=1e-8
            # )
            #
            # # 计算梯度
            # grads_and_vars = optimizer.compute_gradients(self.aloss)
            #
            # # 应用梯度裁剪
            # clipped_grads = []
            # for grad, var in grads_and_vars:
            #     if grad is not None:
            #         # 更温和的梯度裁剪
            #         grad = tf.clip_by_value(grad, -1.0, 1.0)
            #         grad = tf.clip_by_norm(grad, clip_norm=1.0)
            #     clipped_grads.append((grad, var))
            #
            # self.atrain_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

        # 模型保存器
        self.saver = tf.train.Saver(max_to_keep=3)

        # 加载预训练权重
        if load_weights == "True":
            print("Loading Model")
            try:
                checkpoint = tf.train.get_checkpoint_state(self.result_save_path)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                    print("Successfully loaded:", checkpoint.model_checkpoint_path)
                else:
                    print("Could not find old network weights")
            except:
                print("Could not find old network weights")
                self.sess.run(tf.global_variables_initializer())
        else:
            # 初始化所有变量
            self.sess.run(tf.global_variables_initializer())

        # TensorBoard摘要记录
        if trainable:
            self.summary_writer = tf.summary.FileWriter("./summary/PPO", self.sess.graph)
            self.summary_ops, self.summary_vars = build_summaries()

        # 初始化经验回放缓冲区
        self.buffer = []

    # 模块作用：更新Actor和Critic网络
    def update(self, s, a, r,pretrain=0):
        """
        更新PPO网络参数

        Args:
            s: 状态批次
            a: 动作批次
            r: 折扣回报批次

        Returns:
            critic_loss: Critic网络损失
        """
        if(pretrain == 1):
            # 更新Critic网络（多次更新）
            critic_loss = 0
            for _ in range(C_UPDATE_STEPS):
                closs, _ = self.sess.run([self.closs, self.ctrain_op], {self.tfs: s, self.tfdc_r: r})
                # print('*--------------------*', closs)
                critic_loss += closs
            avg_critic_loss = critic_loss / C_UPDATE_STEPS
            # print(f"critic_loss: {avg_critic_loss:.6f}")  # 使用f-string
            return 0, avg_critic_loss




        if (np.any(np.isnan(s)) or np.any(np.isnan(a)) or np.any(np.isnan(r)) or
                np.any(np.isinf(s)) or np.any(np.isinf(a)) or np.any(np.isinf(r))):
            print("🚨 训练数据包含NaN/Inf，跳过本轮更新")
            return 0.0, 0.0


        # 更新旧策略网络
        self.sess.run(self.update_oldpi_op)
        # 计算优势函数
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        if np.any(np.isnan(adv)) or np.any(np.isinf(adv)):
            print("🚨 优势函数包含NaN/Inf，跳过本轮更新")
            return 0.0, 0.0

        # ✅ 检查是否有任何元素超出范围
        if np.any(adv < -5) or np.any(adv > 20):
            print(f"优势函数范围: min={np.min(adv):.3f}, max={np.max(adv):.3f}")
            # 限制优势函数范围
            # print(adv)
            # adv = np.clip(adv, -5.0, 20.0)
            print(f"不裁剪，保持: min={np.min(adv):.3f}, max={np.max(adv):.3f}")

        # 更新Actor网络（多次更新）
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # 🔥 修复：安全的Actor更新
        # actor_losses = []
        # for i in range(min(A_UPDATE_STEPS, 5)):  # 减少更新步数
        #     try:
        #         aloss, _ = self.sess.run([self.aloss, self.atrain_op],
        #                                  {self.tfs: s, self.tfa: a, self.tfadv: adv})
        #         if not (np.isnan(aloss) or np.isinf(aloss)):
        #             actor_losses.append(aloss)
        #         else:
        #             print(f"⚠️ Actor损失出现NaN，跳过剩余更新")
        #             break
        #     except Exception as e:
        #         print(f"⚠️ Actor更新出错: {e}")
        #         break
        # avg_actor_loss = np.mean(actor_losses)

        # actor_loss = 0
        # for _ in range(A_UPDATE_STEPS):
        #     aloss, _ = self.sess.run([self.aloss,self.atrain_op], {self.tfs: s, self.tfa: a, self.tfadv: adv})
        #     actor_loss += aloss
        # avg_actor_loss = actor_loss / A_UPDATE_STEPS

        # 更新Critic网络（多次更新）
        critic_loss = 0
        for _ in range(C_UPDATE_STEPS):
            closs, _ = self.sess.run([self.closs, self.ctrain_op], {self.tfs: s, self.tfdc_r: r})
            # print('*--------------------*', closs)
            critic_loss += closs
        avg_critic_loss = critic_loss / C_UPDATE_STEPS
        # print(f"critic_loss: {avg_critic_loss:.6f}")  # 使用f-string
        return 0,avg_critic_loss

    # 模块作用：构建Actor网络
    # def _build_anet(self, name, trainable):
    #     """
    #     构建Actor策略网络
    #
    #     Args:
    #         name: 网络名称
    #         trainable: 是否可训练
    #
    #     Returns:
    #         norm_dist: 动作概率分布（高斯分布）
    #         params: 网络参数
    #     """
    #     with tf.variable_scope(name):
    #         # CNN特征提取
    #         input = con2d(self.tfs, 'critic', trainable)[:, :, 0, 0]
    #         # 全连接层
    #         l1 = dense(input, 100, 'relu', 'critic', trainable)
    #         # 均值输出（使用tanh激活，范围[-1,1]）
    #         mu = dense(l1, self.M, 'tanh', 'critic', trainable)
    #         # 标准差输出（使用softplus激活，确保正值）
    #         sigma = dense(l1, self.M, 'softplus', 'critic', trainable)
    #         # 创建高斯分布
    #         norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
    #
    #     # 获取网络参数
    #     params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    #     return norm_dist, params
    # 【改进4】
    def _build_anet(self, name, trainable):
        """
        彻底修复Actor网络数值稳定性
        """
        with tf.variable_scope(name):
            # # CNN特征提取
            # input = con2d(self.tfs, 'critic', trainable)[:, :, 0, 0]
            #
            # # 全连接层 - 添加dropout和权重初始化
            # l1 = dense(input, 100, 'relu', 'l1', trainable)
            #
            # # 添加dropout防止过拟合
            # if trainable:
            #     l1 = tf.nn.dropout(l1, keep_prob=0.8)

            # CNN特征提取
            input = con2d(self.tfs, 'critic', trainable)[:, :, 0, 0]

            # 更深的网络结构
            l1 = dense(input, 256, 'relu', 'l1', trainable)
            l1 = tf.nn.dropout(l1, keep_prob=0.7) if trainable else l1

            l2 = dense(l1, 128, 'relu', 'l2', trainable)
            l2 = tf.nn.dropout(l2, keep_prob=0.7) if trainable else l2

            # 均值输出 - 更稳定的初始化
            mu_logits = dense(l2, self.M, None, 'mu_logits', trainable)
            mu = tf.tanh(mu_logits)  # [-1, 1]范围

            # 标准差输出 - 固定的小方差
            sigma = tf.constant(0.1, shape=[self.M])  # 固定标准差

            # 创建高斯分布
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # 模块作用：根据状态预测动作
    def predict(self, s):
        """
        根据当前状态预测投资组合权重

        Args:
            s: 状态张量

        Returns:
            a: 动作（投资组合权重），经过softmax归一化
        """
        # 从策略分布中采样动作
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            print("🚨 s包含NaN/Inf，返回均匀权重")
            print("s", s)
            s = np.ones_like(s)

        a = self.sess.run(self.sample_op, {self.tfs: s})[0]

        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            print("🚨 a包含NaN/Inf，返回均匀权重")
            print("a", a)
            a = np.ones_like(a)


        # 指数运算确保权重为正
            # 🔥 改进：只在超出范围时才裁剪
        a_min, a_max = np.min(a), np.max(a)
        if a_min < -10 or a_max > 10:
            print(f"🔍 a范围超出: [{a_min:.3f}, {a_max:.3f}]，执行裁剪")
            a_clipped = np.clip(a, -10, 10)
            print(f"✅ 裁剪后: [{np.min(a_clipped):.3f}, {np.max(a_clipped):.3f}]")
        else:
            a_clipped = a  # 保持原值

        # 安全的指数运算
        a_exp = np.exp(a_clipped)
        # print("a_exp", a_exp)
        # softmax归一化，确保权重和为1
        a = a_exp / np.sum(a_exp)
        # print("softmax_a", a)
        # 增加批次维度
        a = a[np.newaxis, :]
        # print("final_a", a)
        return a

    # 模块作用：评估状态价值
    def get_v(self, s):
        """
        获取状态s的价值V(s)

        Args:
            s: 状态张量

        Returns:
            value: 状态价值
        """
        if s.ndim < 2:
            s = s[np.newaxis, :]  # 确保有批次维度
        return self.sess.run(self.v, {self.tfs: s})[0]

    # 模块作用：写入TensorBoard摘要
    def write_summary(self, Loss, reward, ep_ave_max_q, actor_loss, epoch):
        """
        记录训练摘要到TensorBoard

        Args:
            Loss: Critic损失
            reward: 奖励值
            ep_ave_max_q: 平均最大Q值
            actor_loss: Actor损失
            epoch: 训练轮数
        """
        summary_str = self.sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: Loss,
            self.summary_vars[1]: reward,
            self.summary_vars[2]: ep_ave_max_q,
            self.summary_vars[3]: actor_loss
        })
        self.summary_writer.add_summary(summary_str, epoch)

    # 模块作用：保存模型
    def save_model(self, epoch):
        """保存模型到文件"""
        self.saver.save(self.sess, './saved_network/PPO/' + self.name, global_step=epoch)

    # 模块作用：保存经验转移
    def save_transition(self, s, w, r):
        """
        保存经验到回放缓冲区

        Args:
            s: 当前状态
            w: 动作（权重）
            r: 奖励
            contin: 是否继续（未使用）
            s_next: 下一状态（未使用）
            action_precise: 精确动作（未使用）
        """
        self.buffer.append([s, w, r])

    # 模块作用：训练PPO网络
    def train(self, method, epoch):
        """
        训练PPO网络

        Args:
            method: 训练方法（未使用）
            epoch: 训练轮数

        Returns:
            info: 训练信息字典
        """
        info = dict()
        # 计算最后一个状态的价值
        v = self.get_v(self.buffer[-1][0])
        # 计算折扣回报
        discounted_r = []
        # 反转奖励序列，从后往前计算
        rs = [transition[2]-1 for transition in self.buffer[::-1]]
        for r in rs:
            v = r + self.gamma * v  # 贝尔曼方程
            discounted_r.append(v)
        discounted_r.reverse()  # 反转回原始顺序
        discounted_r = np.array(discounted_r)

        print("discounted_r、v",discounted_r[0])
        # print(self.v[0][0])

        # 准备训练数据
        mini_batch_s = np.vstack([transition[0] for transition in self.buffer[::-1]])
        mini_batch_a = np.vstack([transition[1] for transition in self.buffer[::-1]])

        v_values = self.sess.run(self.v, {self.tfs: mini_batch_s})
        print(v_values[0])
        # 更新网络
        actor_loss,critic_loss = self.update(mini_batch_s, mini_batch_a, discounted_r,pretrain=1 if epoch == -5 else 0)
        # # 清空缓冲区
        # self.buffer = []
        #
        # # 返回训练信息
        # info["critic_loss"] = critic_loss
        # info["q_value"] = 0
        # info["actor_loss"] = 0
        # return info
        # 【改进5】
        # 计算实际的Actor损失和Q值
        # 获取优势函数来计算实际损失
        # adv = self.sess.run(self.advantage, {self.tfs: mini_batch_s, self.tfdc_r: discounted_r})
        # actor_loss_val = self.sess.run(self.aloss, {self.tfs: mini_batch_s, self.tfa: mini_batch_a, self.tfadv: adv})

        # 计算平均Q值（使用Critic的价值估计）
        q_values = self.sess.run(self.v, {self.tfs: mini_batch_s})
        avg_q_value = np.mean(q_values)

        # 清空缓冲区
        self.buffer = []

        # 返回训练信息 - 修正这里
        info["critic_loss"] = critic_loss
        info["q_value"] = avg_q_value
        info["actor_loss"] = actor_loss
        return info
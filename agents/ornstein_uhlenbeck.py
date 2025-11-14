"""
Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
"""

import numpy as np

# Ornstein-Uhlenbeck过程噪声类，用于在DRL中生成符合物理系统特性的随机噪声
class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck过程噪声生成器
    用于在深度强化学习中为连续动作空间添加时间相关的随机噪声
    这种噪声具有均值回归特性，比高斯噪声更适合物理系统
    """

    def __init__(self, mu, sigma=0.17, theta=.73, dt=1e-2, x0=None):
        """
        初始化Ornstein-Uhlenbeck噪声生成器

        参数:
        mu: 噪声的长期均值，通常设为0，表示噪声围绕0值波动【噪声均值】
        sigma: 噪声的波动幅度，控制随机项的强度，默认0.03【值小一点，降低一点噪声】
        theta: 均值回归速率，控制噪声回归到均值的速度，值越大回归越快，默认0.1【让噪声在均值附近波动】
        dt: 时间步长，模拟连续时间过程的离散化步长，默认0.01【0.01s 多久更新一次噪声，值越小，噪声变化越平滑】
        x0: 初始状态值，如果为None则初始化为与mu相同形状的零向量
        """
        self.theta = theta      # 均值回归速率参数
        self.mu = mu            # 长期均值
        self.sigma = sigma      # 波动率/噪声强度
        self.dt = dt            # 时间步长
        self.x0 = x0            # 初始状态
        self.reset()            # 重置噪声状态到初始值

    def __call__(self):
        """
        生成下一个噪声值
        实现Ornstein-Uhlenbeck过程的离散形式：dx = theta*(mu - x)*dt + sigma*dW

        返回:
        x: 下一个噪声值，与mu具有相同的形状
        """
        # Ornstein-Uhlenbeck过程的核心公式：
        # 均值回归项 + 随机波动项
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # 更新前一个状态为当前状态，为下一次调用做准备
        self.x_prev = x
        # 返回生成的噪声值
        return x

    def reset(self):
        """
        重置噪声状态到初始值
        在DRL训练中，通常在每个episode开始时调用，确保噪声过程重新开始
        """
        # 如果提供了初始值x0，使用x0；否则初始化为与mu相同形状的零向量
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        """
        返回对象的字符串表示，用于调试和日志记录

        返回:
        描述噪声参数的字符串
        """
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
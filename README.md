# XDRL-for-Finance: Explainable Deep Reinforcement Learning for Portfolio Management

基于论文《Explainable Post hoc Portfolio Management Financial Policy of a Deep Reinforcement Learning agent》的复现项目，使用可解释深度强化学习进行投资组合管理。

总结项目内容

已有股票数据csv，训练DRL让它做投资组合得到尽可能多的回报。

在2017年的新数据上测试效果

利用SHAP、LIME、特征重要性可视化解释状态-动作对

## 项目复现概述

本项目实现了一个基于 PPO (Proximal Policy Optimization) 算法的深度强化学习投资组合管理系统，并集成 SHAP、LIME 和特征重要性分析等可解释AI技术，在交易时解释DRL智能体的投资决策。

- **原论文**: Escudero et al. "Explainable Post hoc Portfolio Management Financial Policy of a Deep Reinforcement Learning agent"
- **代码Github**: [XDRL-for-finance](https://github.com/aleedelarica/XDRL-for-finance)
- **PPO框架**: [Reinforcement learning in portfolio management](https://github.com/deepcrypto/Reinforcement-learning-in-portfolio-management)

### 核心特性
- 🎯 **PPO算法**：用于投资组合权重分配的深度强化学习
- 📊 **多维度解释**：SHAP、LIME、特征重要性三种可解释技术
- 💰 **实战投资**：在真实股票数据上进行训练和测试

## 环境配置

### 创建 Conda 环境
```bash
conda create -n money python=3.6
conda activate money
```

### 安装依赖
```bash
pip install -r requirements.txt

# 额外安装可解释性工具包，例如
pip install shap lime
```

## 项目结构
![项目架构图](https://cdn.nlark.com/yuque/0/2025/png/36186524/1763115514276-a1ae30c8-6275-4e09-8854-eb81113f36dc.png?x-oss-process=image%2Fformat%2Cwebp)

## 快速开始

### 1. 数据准备
确保在 `data/raw/` 目录中包含以下股票的OHLCV数据CSV文件：
- AAPL (Apple)
- ADBE (Adobe) 
- BABA (Alibaba)
- SNE (Sony)
- V (Visa)

### 2. 训练模型
```bash
python main.py --mode=train
```

**训练配置** (`config.json`):
```json
{
    "session": {
        "start_date": "2015-01-01",
        "end_date": "2016-12-31",
        "market_types": "America",
        "codes": ["AAPL","ADBE","BABA","SNE","V"],
        "features": ["close","high","low","open"],
        "agents": ["CNN","PPO","30"],
        "epochs": "10000",
        "noise_flag": "True",
        "record_flag": "True", 
        "plot_flag": "False",
        "reload_flag": "False",
        "trainable": "True",
        "method": "model_free"
    }
}
```

### 3. 测试模型
```bash
python main.py --mode=test
```

**测试配置**:
```json
{
    "session": {
        "start_date": "2017-01-01",
        "end_date": "2017-12-31", 
        "market_types": "America",
        "codes": ["AAPL","ADBE","BABA","SNE","V"],
        "features": ["close","high","low","open"],
        "agents": ["CNN","PPO","30"], 
        "epochs": "-1",
        "noise_flag": "False",
        "record_flag": "True",
        "plot_flag": "True",
        "reload_flag": "True",
        "trainable": "False",
        "method": "model_free"
    }
}
```

### 4. 解释模型决策

#### 步骤1: 准备解释数据
将测试生成的状态-动作对数据文件放入 `explainability/data/` 目录。

#### 步骤2: 数据清洗
修改 `cleaner.py` 中的文件名，然后运行：
```bash
python explainability/cleaner.py
```
生成 `cleaned_state_actions.csv`。

#### 步骤3: 可视化解释

将要解释的状态-动作对数据文件放到explainability.data下

运行cleaner.py文件（注意在代码内修改文件名）得到cleand_state_actions.csv

运行easy_explain.ipynb可视化解释

## 可解释性技术

### 1. **特征重要性分析**
- 全局特征重要性排序
- 各资产特征贡献度分析
- 技术指标重要性比较

### 2. **SHAP (SHapley Additive exPlanations)**
- 个体预测解释
- 特征贡献力可视化
- 多输出模型支持（6个资产权重分配）

### 3. **LIME (Local Interpretable Model-agnostic Explanations)**
- 局部预测解释
- 具体时间点的决策分析
- 正负贡献特征识别

## 实验设置

### 数据周期
- **训练期**: 2015-01-01 至 2016-12-31
- **测试期**: 2017-01-01 至 2017-12-31

### 资产组合
5支科技股: AAPL, ADBE, BABA, SNE, V

### 特征空间
24维状态空间（5支股票+1支现金 × 4个OHLC特征）

## 核心算法

### PPO (Proximal Policy Optimization)
- 策略梯度算法，训练稳定
- 裁剪目标函数，防止策略更新过大
- 适用于连续动作空间（投资组合权重分配）

### 神经网络架构
- 卷积神经网络 (CNN) 用于特征提取
- Actor-Critic 框架
- 多输出层，每个资产对应一个权重输出

## 已知问题与注意事项

⚠️ **项目完善度说明**

本项目并不完善。包括但不限于：
1，配置文件以及参数的解析
2，PPO的底层代码设计，权重NAN
3，投资组合的财富变化、奖励设计
4，回报率不高，但测试时变高(可能因为股票本来就在上涨)
5，环境类、main文件代码杂乱无章


## 许可证

本项目基于原论文和参考代码实现，遵循相应的学术使用规范。

---

**注意**: 本项目主要用于学术研究和算法验证，不构成实际投资建议。金融市场投资存在风险，请谨慎决策。







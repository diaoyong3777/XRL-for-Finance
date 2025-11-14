# 导入必要的Python库
# pandas：数据处理和分析库，提供DataFrame数据结构
import pandas as pd
# numpy：数值计算库，提供数组和矩阵运算
import numpy as np
# ast：抽象语法树，用于安全地解析字符串为Python表达式
import ast
# json：JSON数据处理库
import json


# 模块作用：清洗状态-动作CSV文件，将其从复杂的数组格式转换为规整的表格格式
# 论文对应：第4节中为可解释性分析准备数据，将DRL代理的状态和动作转换为适合SHAP/LIME分析的格式
def clean_state_action_csv(input_file, output_file, config_file):
    """
    清洗状态-动作CSV文件

    主要功能：
    1. 解析原始的字符串格式的状态和动作数据
    2. 将多维数组展平为一维特征向量
    3. 为每个特征生成有意义的列名
    4. 保存为规整的CSV格式，便于后续分析

    Args:
        input_file: 原始状态-动作CSV文件路径
        output_file: 清洗后的输出文件路径
        config_file: 配置文件路径，包含资产和特征信息
    """

    # 加载config.json配置文件
    # 'r'表示以只读模式打开文件
    with open(config_file, 'r') as f:
        # json.load()将JSON文件内容解析为Python字典
        config = json.load(f)

    # 从配置文件中提取关键维度参数
    # M：状态空间的第一维度 = 资产数量 + 1（现金资产）
    M = len(config['session']['codes']) + 1
    # L：时间窗口长度，从agents配置的第三个元素获取
    L = int(config['session']['agents'][2])
    # N：特征数量
    N = len(config['session']['features'])

    # 使用pandas读取CSV文件
    # pd.read_csv()将CSV文件读取为DataFrame对象
    df = pd.read_csv(input_file)

    # 模块作用：辅助函数，解析字符串表示的数组
    def parse_array(array_str):
        """
        将字符串格式的数组解析为numpy数组

        Args:
            array_str: 包含数组数据的字符串

        Returns:
            numpy数组对象
        """
        # 替换字符串中的换行符为空字符串
        cleaned_str = array_str.replace('\n', '')
        # 将连续的空格替换为逗号（为后续解析做准备）
        cleaned_str = cleaned_str.replace('  ', ',')
        # 将单个空格替换为逗号
        cleaned_str = cleaned_str.replace(' ', ',')
        # 使用ast.literal_eval安全地评估字符串为Python表达式
        # 比eval()更安全，因为它只处理字面量结构
        array_data = ast.literal_eval(cleaned_str)
        # 将Python列表转换为numpy数组
        return np.array(array_data)

    # 调试用：打印DataFrame的前几行（注释状态）
    # print(df.head())

    # 应用解析函数到State和Action列
    # df['State'].apply()对State列的每个元素应用parse_array函数
    df['State'] = df['State'].apply(parse_array)
    df['Action'] = df['Action'].apply(parse_array)

    # 调试用：打印解析后的前几行（注释状态）
    # print(df.head())

    # 模块作用：展平状态和动作数据，将多维数组转换为一维特征向量
    # 初始化一个空列表来存储展平后的数据
    flattened_data = []

    # 遍历DataFrame的每一行
    # df.iterrows()返回索引和行数据的迭代器
    for _, row in df.iterrows():
        # 展平动作数组：从二维[M,1]展平为一维[M]
        # row['Action']是一个numpy数组，.flatten()将其展平为一维
        action = row['Action'].flatten()
        # 展平状态数组：从四维[1,M,L,N]展平为一维[M*L*N]
        state = row['State'].flatten()
        # 将动作和状态特征连接成一个长特征向量
        # np.concatenate()沿着现有轴连接数组序列
        flattened_row = np.concatenate((action, state))
        # 将展平后的行数据添加到列表中
        flattened_data.append(flattened_row)

    # 调试用：打印展平后的数据（注释状态）
    # print(flattened_data)

    # 模块作用：为展平后的特征生成有意义的列名
    # 创建动作列名列表
    action_columns = []
    # 遍历每个资产位置（0到M-1）
    for i in range(M):
        # 如果是第一个位置（i=0），对应现金资产
        # 否则从配置的股票代码列表中获取对应股票代码
        asset_name = config['session']['codes'][i - 1] if i > 0 else 'Cash'
        # 生成列名，格式：Action_资产名
        column_name = f'Action_{asset_name}'
        # 添加到动作列名列表
        action_columns.append(column_name)

    # 创建状态列名列表
    state_columns = []
    # 三层嵌套循环生成所有状态特征的列名
    # 第一层：遍历每个资产（M个）
    for i in range(M):
        # 第二层：遍历时间窗口中的每个时间点（L个）
        for j in range(L):
            # 第三层：遍历每个特征（N个）
            for k in range(N):
                # 获取特征名称
                feature_name = config['session']['features'][k]
                # 获取资产名称（现金或股票代码）
                asset_name = config['session']['codes'][i - 1] if i > 0 else 'Cash'
                # 生成列名，格式：State_资产名_特征名_L时间滞后
                # L{j+1}表示第j+1个时间滞后（从1开始计数）
                column_name = f'State_{asset_name}_{feature_name}_L{j + 1}'
                # 添加到状态列名列表
                state_columns.append(column_name)

    # 合并动作和状态列名
    columns = action_columns + state_columns

    # 调试用：打印列名（注释状态）
    # print(columns)

    # 创建新的DataFrame来存储清洗后的数据
    # pd.DataFrame()从二维数据创建DataFrame对象
    # flattened_data：二维列表，每行是一个样本的特征向量
    # columns：列名列表，指定每列的名称
    cleaned_df = pd.DataFrame(flattened_data, columns=columns)

    # 将清洗后的DataFrame保存到新的CSV文件
    # index=False表示不保存行索引
    cleaned_df.to_csv(output_file, index=False)


# 模块作用：合并清洗后的状态-动作数据与交易结果数据
# 论文对应：为可解释性分析创建完整的数据集，包含输入（状态-动作）和输出（回报、财富）
def merge_state_action_results(cleaned_state_action_file, results_file, output_file, config_file):
    """
    合并状态-动作数据与交易结果数据

    主要功能：
    1. 加载清洗后的状态-动作数据
    2. 加载交易结果数据（财富、回报、价格）
    3. 处理价格数据的格式
    4. 按行合并两个数据集

    Args:
        cleaned_state_action_file: 清洗后的状态-动作文件路径
        results_file: 交易结果文件路径
        output_file: 合并后的输出文件路径
        config_file: 配置文件路径
    """

    # 加载配置文件
    with open(config_file, 'r') as f:
        config = json.load(f)

    # 提取资产数量M
    M = len(config['session']['codes']) + 1

    # 加载清洗后的状态-动作CSV文件
    state_action_df = pd.read_csv(cleaned_state_action_file)

    # 加载交易结果CSV文件
    results_df = pd.read_csv(results_file)

    # 如果结果数据中包含'Weight'列，则删除该列
    # 因为状态-动作数据中已经包含了权重信息
    if 'Weight' in results_df.columns:
        results_df = results_df.drop(columns=['Weight'])

    # 处理价格数据：将逗号分隔的价格字符串拆分为多列
    price_columns = []
    # 为每个资产生成价格列名
    for i in range(M):
        # 获取资产名称
        asset_name = config['session']['codes'][i - 1] if i > 0 else 'Cash'
        # 生成价格列名，格式：Price_资产名
        price_columns.append(f'Price_{asset_name}')

    # 将Price列按逗号分隔并扩展为多列
    # str.split(',', expand=True)将字符串按逗号分割并扩展为多列
    results_df[price_columns] = results_df['Price'].str.split(',', expand=True)
    # 删除原始的Price列，因为已经拆分为多列
    results_df.drop('Price', axis=1, inplace=True)

    # 删除结果DataFrame的第一列（通常是索引列或不需要的列）
    # results_df.columns[0]获取第一列的列名
    results_df.drop(results_df.columns[0], axis=1, inplace=True)

    # 合并两个DataFrame
    # pd.concat()沿着轴连接pandas对象
    # axis=1表示按列连接（水平连接）
    # 假设两个DataFrame的行数相同且按时间顺序对齐
    merged_df = pd.concat([state_action_df, results_df], axis=1)

    # 将合并后的DataFrame保存到新的CSV文件
    merged_df.to_csv(output_file, index=False)


# 示例使用：清洗状态-动作数据
# 定义输入输出文件路径
input_file = '../explainability_data/epoch999-state_actions.csv'  # 原始状态-动作文件
output_file = '../explainability_data/cleaned_state_actions.csv'  # 清洗后的输出文件
config_file = '../config.json'  # 配置文件

# 调用清洗函数
clean_state_action_csv(input_file, output_file, config_file)

# 示例使用：合并状态-动作数据与交易结果数据（当前被注释）
# 定义文件路径
cleaned_state_action_file = '../explainability_data/cleaned_state_actions.csv'  # 清洗后的状态-动作文件
results_file = '../result/test-57.064914695339375.csv'  # 交易结果文件
output_file = '../explainability_data/cleaned_state_action_results.csv'  # 合并后的输出文件
config_file = '../config.json'  # 配置文件

# 调用合并函数（当前被注释，取消注释即可执行）
# merge_state_action_results(cleaned_state_action_file, results_file, output_file, config_file)
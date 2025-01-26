import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# 加载数据
athletes_data = pd.read_csv(r"C:\Users\123\Desktop\athlete_counts_by_country_sport.csv")  # 运动员参赛情况
medals_data = pd.read_csv(r"C:\Users\123\Desktop\summerOly_medal_counts.csv")  # 国家奖牌数量
total_medals_data = pd.read_csv(r"C:\Users\123\Desktop\gold_medals_by_year.csv")  # 总奖牌数

# 读取人口数据（假设包含 'Country', 'Year' 和 'Population' 列）
population_data = pd.read_csv(r"F:\pycharm\偏心受压构件的正截面承载力计算\数据处理\filtered_population_data.csv")  # 替换为人口数据路径

# 读取GDP数据（假设包含 'Country', 'Year' 和 'GDP' 列）
gdp_data = pd.read_csv(r"F:\pycharm\偏心受压构件的正截面承载力计算\数据处理\olympic_gdp_data.csv")  # 替换为GDP数据路径

# 读取主办方数据（假设包含 'Year' 和 'HostCountry' 列）
host_country_data = pd.read_csv(r"C:\Users\123\Desktop\summerOly_hosts.csv")  # 替换为主办国数据路径

# 合并数据：先将运动员数量和奖牌数据按年份和国家合并
data = pd.merge(athletes_data, medals_data, on=['Year', 'Team'])
data = pd.merge(data, total_medals_data, on='Year')

# 合并人口数据和GDP数据
data = pd.merge(data, population_data, how='left', on=['Year', 'Team'])
data = pd.merge(data, gdp_data, how='left', on=['Year', 'Team'])

# 合并主办方数据
data = pd.merge(data, host_country_data, how='left', on='Year')  # 假设 'HostCountry' 列包含主办国

# 创建主办方效应特征：如果当前年份的主办国为该国家，主办方效应为1，否则为0
data['host_country_effect'] = (data['Team'] == data['HostCountry']).astype(int)

# 创建滞后特征：往年五届的 `medal_count`
for i in range(1, 6):
    data[f'prev_{i}_medal_count'] = data.groupby('Team')['medal_count'].shift(i)

# 删除包含空值的行（因为新加入的滞后特征和主办方效应可能会导致前几年的数据缺失）
data = data.dropna(subset=['GDP', 'Population', 'host_country_effect'])

# 特征变量和目标变量
X = data[['athletes_count', 'Population', 'GDP', 'host_country_effect'] + [f'prev_{i}_medal_count' for i in range(1, 6)]]  # 输入变量：运动员数量，人口，GDP，主办方效应和五届奖牌数
y = data['medal_count']  # 目标变量：国家奖牌数

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练XGBoost模型
model = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 预测

# 保存模型到文件
model_filename = 'xgboost_model_with_host_country_effect.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

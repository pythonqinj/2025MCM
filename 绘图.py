import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 加载已保存的模型
model_filename = 'xgboost_model.pkl'
model = joblib.load(model_filename)
print("Model loaded successfully!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 加载数据
athletes_data = pd.read_csv(r"C:\Users\123\Desktop\athlete_counts_by_country_sport.csv")  # 运动员参赛情况
medals_data = pd.read_csv(r"C:\Users\123\Desktop\summerOly_medal_counts.csv")  # 国家奖牌数量
total_medals_data = pd.read_csv(r"C:\Users\123\Desktop\gold_medals_by_year.csv")  # 总奖牌数

# 假设每个表格有以下列：
# athletes_count.csv: ['year', 'Team', 'athletes_count']
# medals_count.csv: ['year', 'Team', 'medal_count']
# total_medals.csv: ['year', 'total_medals']

# 合并数据
# 先将运动员数量和奖牌数据按年份和国家合并
data = pd.merge(athletes_data, medals_data, on=['Year', 'Team'])

# 再将总奖牌数（总计所有国家的奖牌数）按年份合并
data = pd.merge(data, total_medals_data, on='Year')
# 加载现实世界数据（替换为你的实际文件路径）
real_world_data = data
print("Real-world data loaded successfully!")

# 选择五个感兴趣的国家
countries_of_interest = ['United States', 'China']  # 替换为你感兴趣的国家
filtered_data = real_world_data[real_world_data['Team'].isin(countries_of_interest)]

# 对筛选后的数据进行预测
X_real = filtered_data[['athletes_count', 'total_medals']]
filtered_data['predicted_medal_count'] = model.predict(X_real)

# 保存带有预测结果的现实数据
output_filename = 'filtered_predictions.csv'
filtered_data.to_csv(output_filename, index=False)
print(f"Filtered predictions saved to {output_filename}")


# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))

# 遍历五个国家，绘制每个国家的曲线
for country in countries_of_interest:
    country_data = filtered_data[filtered_data['Team'] == country]
    plt.plot(country_data['Year'], country_data['medal_count'], label=f'{country} Actual', marker='o')
    plt.plot(country_data['Year'], country_data['predicted_medal_count'], label=f'{country} Predicted', linestyle='--', marker='x')

plt.title('Actual vs Predicted Medal Count (Selected Countries)')
plt.xlabel('Year')
plt.ylabel('Medal Count')
plt.legend()
plt.grid()
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 计算预测值与实际值的误差
mae = mean_absolute_error(filtered_data['medal_count'], filtered_data['predicted_medal_count'])
mse = mean_squared_error(filtered_data['medal_count'], filtered_data['predicted_medal_count'])
# 计算误差比值 (Relative Error)
filtered_data['relative_error'] = abs(filtered_data['predicted_medal_count'] - filtered_data['medal_count']) / filtered_data['medal_count']

# 计算平均误差比值
average_relative_error = filtered_data['relative_error'].mean()

# 输出结果
print(f"Average Relative Error: {average_relative_error:.2%}")  # 输出百分比格式

# 输出误差结果


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

# 读取奥运会奖牌数据（包含每个国家每个项目的奖牌数）
medals_df = pd.read_csv(r"C:\Users\123\Desktop\sport-medal.csv")  # 替换为您的文件路径

# 读取奥运会项目设置数据（横向排列的年份在列头，项目在行）
events_df = pd.read_csv(r"C:\Users\123\Desktop\grouped_data.csv",index_col=0)  # 假设年份是列头，项目在行

# 转置项目设置数据，使其每一列代表一个项目，每一行代表一个年份
events_df = events_df.T  # 将年份列变成行，行变成列

# 查看转置后的项目设置数据
print(events_df.head())

# 筛选出感兴趣的国家的数据（以美国为例）
us_medals = medals_df[medals_df['Team'] == 'United States']
us_medals = us_medals.groupby('Year', as_index=False).sum()
print(us_medals)

medals_df['Year'] = medals_df['Year'].astype(int)
events_df.index = events_df.index.astype(int)
# 首先确保奖牌数据和项目设置数据有共同的 'Year' 列
merged_data = pd.merge(us_medals, events_df, left_on='Year', right_index=True, how='inner')

# 查看合并后的数据
# 选择特征（项目设置数量）和目标变量（美国奖牌数）
X = merged_data[events_df.columns]  # 特征：项目设置数量
print(X)
y = merged_data['Medal_Count']     # 目标变量：美国的奖牌数

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = Ridge(alpha=1.0)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

print(f"回归系数: {model.coef_}")
print(f"截距: {model.intercept_}")

# 输出模型的评估结果
print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred)}")
print(f"R^2: {r2_score(y_test, y_pred)}")
#
# # 保存模型（如果需要）
# import joblib
# joblib.dump(model, "us_medal_prediction_model.pkl")

# 获取回归系数和特征名称
import matplotlib.pyplot as plt
import numpy as np



# 获取测试集对应的年份
years_test = merged_data.iloc[X_test.index]['Year']

# 排序以确保年份按升序排列
sorted_indices = years_test.argsort()  # 获取按年份排序的索引
years_test_sorted = years_test.iloc[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(years_test_sorted, y_test_sorted, label="Actual Medal Count", marker='o', linestyle='-', color='blue')  # 真实数据
plt.plot(years_test_sorted, y_pred_sorted, label="Predicted Medal Count", marker='x', linestyle='--', color='orange')  # 预测数据

# 添加标题和标签
plt.title("Actual vs Predicted Medal Count (United States)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Medal Count", fontsize=12)

# 显示图例
plt.legend()

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
#
# # 添加网格
# plt.grid(True)
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 获取回归系数和特征名称
# coefficients = model.coef_
# features = events_df.columns
#
# # 创建条形图
# plt.figure(figsize=(12,6))
# plt.bar(features, coefficients, color='skyblue')
#
# # 添加标题和标签
# plt.title('Feature Importance (Weights) for Each Sport', fontsize=14)
# plt.xlabel('Sports', fontsize=12)
# plt.ylabel('Coefficient (Weight)', fontsize=12)
#
# # 旋转x轴标签以适应
# plt.xticks(rotation=90)
#
# # 显示图形
# plt.tight_layout()
# plt.show()

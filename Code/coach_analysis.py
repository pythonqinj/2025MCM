import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# # 设置随机种子
# np.random.seed(42)

# # 生成数据
# years = np.arange(1980,2024,4)
# countries = ['USA','China','Germany','Russia','UK','Japan','Australia','Brazil','France','India','Italy','Canada']

# data = {
#     'Year':np.repeat(years,len(countries)),
#     'Country':countries*len(years),
#     'Gold':np.random.randint(0,100,len(years)*len(countries)),
#     'Silver':np.random.randint(0,100,len(years)*len(countries)),
#     'Bronze':np.random.randint(0,100,len(years)*len(countries)),
#     'Total':np.random.randint(100,500,len(years)*len(countries)),
#     'event_count':np.random.randint(20,50,len(years)*len(countries)),
#     'Is_host':np.random.choice([0,1],len(years)*len(countries)),
#     'Coach_effect':np.random.choice([0,1],len(years)*len(countries))    # 教练效应,1表示有影响，0表示无影响
# }

# # 创建数据框
# df = pd.DataFrame(data)

# 读取csv文件
df = pd.read_csv('analysis_data1.csv',encoding='utf-8')

# 检查数据是否包含所需的列
required_columns = ['Year', 'Country', 'Gold', 'Silver', 'Bronze', 'Total', 'event_count', 'Host', 'Coach_effect']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 数据标准化
scaler = StandardScaler()
X = df[['Gold','Silver','Bronze','event_count','Is_host']]
X_scaled = scaler.fit_transform(X)

# 目标变量：金牌数
y_gold = df['Gold']

# 拆分数据集:70%训练集，30%测试集
X_train,X_test,y_train_gold,y_test_gold = train_test_split(X_scaled,y_gold,test_size=0.3,random_state=42)

# 线性回归模型
model = LinearRegression()
model.fit(X_train,y_train_gold)

# 预测金牌数
y_pred_gold = model.predict(X_test)

# 评估模型
mse_gold = mean_squared_error(y_test_gold,y_pred_gold)
mae_gold = mean_absolute_error(y_test_gold,y_pred_gold)

# 输出评估结果
print(f'MSE for Gold Medal Prediction: {mse_gold}')
print(f'MAE for Gold Medal Prediction: {mae_gold}')

# 可视化预测结果
# 1.教练效应与金牌数的关系散点图
plt.figure(figsize=(10,6))

# 对每个国家的金牌数进行聚合，计算平均金牌数，减少波动
df_avg_gold = df.groupby(['Country','Coach_effect'])['Gold'].mean().reset_index()

# 绘制散点图
sns.scatterplot(data=df_avg_gold,x='Coach_effect',y='Gold',hue='Coach_effect',palette='coolwarm',s=100,edgecolor='black',markers='o')   # s:点的大小

# 添加回归线来平滑数据
sns.regplot(data=df_avg_gold,x='Coach_effect',y='Gold',scatter=False,color='blue',line_kws={'linewidth':2})

# 添加标题和标签
plt.title('Effect of Coach Influence on Gold Medals (Smoothed)',fontsize=14)
plt.xlabel('Coach Effect(1 = Yes, 0 = No)',fontsize=12)
plt.ylabel('Average Gold Medals per Country',fontsize=12)
plt.legend(title='Coach Effect',labels=['No Influence','Influenced'],title_fontsize='12',fontsize='10')
plt.tight_layout()
plt.show()

# 可视化2：教练效应对不同国家金牌数的影响条形图
df_grouped = df.groupby(['Country','Coach_effect'])['Gold'].sum().unstack().fillna(0)
df_grouped.plot(kind='bar',stacked=True,figsize=(12,6),color=['blue','red'])
plt.title('Gold Medals by Country and Coach Effect',fontsize=14)
plt.xlabel('Country',fontsize=12)
plt.ylabel('Total Gold Medals',fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Coach Effect',labels=['No Influence','Influenced'],title_fontsize='12',fontsize='10')
plt.tight_layout()
plt.show()

# 可视化3：奖牌数随教练效应变化的折线图
df_line = df.groupby(['Year','Coach_effect'])['Gold'].sum().unstack().fillna(0)
df_line.plot(kind='line',figsize=(12,6),marker='o',color=['blue','red'])
plt.title('Gold Medals by Year and Coach Effect',fontsize=14)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Total Gold Medals',fontsize=12)
plt.legend(title='Coach Effect',labels=['No Influence','Influenced'],title_fontsize='12',fontsize='10')
plt.tight_layout()
plt.show()

# 可视化4：奖牌数随教练效应变化的热力图
df_pivot = df.pivot_table(index='Country',columns='Coach_effect',values='Gold',aggfunc='sum',fill_value=0)
plt.figure(figsize=(12,6))
sns.heatmap(df_pivot,annot=True,cmap='coolwarm',fmt='d',cbar=True,linecolor='white',linewidth=1)
plt.title('Gold Medals by Country and Coach Effect (Heatmap)',fontsize=14)
plt.xlabel('Coach Effect',fontsize=12)
plt.ylabel('Country',fontsize=12)
plt.tight_layout()
plt.show()

# 预测2028年的数据进行标准化
# 假设2028年奥运会的数据
X_2028 = np.array([[50,45,40,25,1]])    # 示例：金牌数50，银牌数45，铜牌数40，项目数25，主办国1

# 标准化数据
X_2028_scaled = scaler.transform(X_2028)

# 预测2028年金牌数
y_pred_2028_gold = model.predict(X_2028_scaled)
print(f'Predicted Gold Medals in 2028: {y_pred_2028_gold[0]:.0f}')

# 通过bootstrap方法估计2028年金牌数的不确定性
def bootstrap_predict(model,X_test,y_test,n_iterations=1000,percentile=95):
    predictions = np.zeros((n_iterations,len(y_test)))
    for _ in range(n_iterations):
        X_resampled,y_resampled = resample(X_test,y_test,replace=True,random_state=42)
        model.fit(X_resampled,y_resampled)
        y_pred = model.predict(X_2028_scaled)
        predictions[_,:] = model.predict(X_test)

    lower = np.percentile(predictions,(100-percentile)/2,axis=0)
    upper = np.percentile(predictions,100-(100-percentile)/2,axis=0)
    return lower,upper

# 估计2028年金牌数的不确定性
lower_gold,upper_gold = bootstrap_predict(model,X_test,y_test_gold)

# 可视化5：2028年金牌数的不确定性
plt.figure(figsize=(10,6))
sns.histplot(y_pred_gold,bins=30,kde=True,color='orange',label='Predicted Gold Medals')
plt.fill_between(np.arange(len(lower_gold)),lower_gold,upper_gold,color='orange',alpha=0.3,label='95% Confidence Interval')
plt.legend()
plt.title('Gold Medal Prediction with 95% Confidence Interval',fontsize=14)
plt.xlabel('Predicted Gold Medals',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.tight_layout()
plt.show()

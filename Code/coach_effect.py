import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
hosts_path = 'summerOly_hosts.csv'
medal_counts_path = 'summerOly_medal_counts.csv'

# 读取数据
hosts = pd.read_csv(hosts_path,encoding='utf-8')
medal_counts = pd.read_csv(medal_counts_path,encoding='ISO-8859-1')

# 修复列名中的特殊字符
print("Host columns before rename:",hosts.columns)
hosts.columns = hosts.columns.str.strip().str.replace('i>>?','',regex=True)  # 移除特殊字符
medal_counts.columns = medal_counts.columns.str.strip()

# 确保'Year'列存在且为整数
if 'Year' in hosts.columns:
    hosts['Year'] = hosts['Year'].astype(str).str.extract('(\d+)').astype(int)  # 提取年份数字并转换为整数
else:
    raise ValueError("Column 'Year' not found in hosts data. Please check the dataset.")

if 'Year' in medal_counts.columns:
    medal_counts['Year'] = medal_counts['Year'].astype(int)
else:
    raise ValueError("Column 'Year' not found in medal counts data. Please check the dataset.")

# 合并数据
data = pd.merge(medal_counts,hosts,on='Year',how='left')

# 检查合并后的数据
print("Merged data:/n",data.head())

# 模型部分
# 假设要预测Total与其他列之间的关系
# 选择特征列
features = ['Year','Gold','Silver','Bronze']
target = 'Total'

X = data[features].dropna()   # 删除缺失值
y = data[target].dropna()

# 划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 随机森林模型
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[5,10,None],
    'min_samples_split':[2,5,10]
}

# 网格搜索
grid_search = GridSearchCV(estimator=rf_model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=1,verbose=1)
grid_search.fit(X_train,y_train)

# 最优参数
best_params = grid_search.best_params_

# 最优模型
best_model = grid_search.best_estimator_

# 预测和评估
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f'Optimized Random Forst MSE: {mse:.2f}')
print(f'Optimized Random Forest R2: {r2:.2f}')

# 创建特征重要性数据框
feature_importances = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_}).sort_values('importance',ascending=False)

# 按重要性排序
feature_importances = feature_importances.sort_values('importance', ascending=False)

# 可视化
plt.figure(figsize=(8,6))
sns.barplot(x='importance',y='feature',data=feature_importances,palette='viridis')
plt.title('Feature Importance for Predicting Total Medals',fontsize=14)
plt.xlabel('Importance',fontsize=12)
plt.ylabel('Feature',fontsize=12)
plt.tight_layout()
plt.show()

# 显示结果
print(f'Future importance for predicting Total Medals:/n{feature_importances}')

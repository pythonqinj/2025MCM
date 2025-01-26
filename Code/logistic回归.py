from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("cleaned_data.csv")
# 选择特征列和目标列
X = df[[ 'athletes_count', 'sports_count', 'events_count']]  # 特征
y = df['Label']  # 目标标签
#'GDP_Percentage', 'Population_Percentage'
# 去除包含空值的行
X = X.dropna()
y = y[X.index]  # 确保目标列和特征列的行对齐

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型并训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

from sklearn.metrics import precision_recall_curve


precision, recall, _ = precision_recall_curve(y_test, y_pred)

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
# 获取标准化后的系数
from sklearn.preprocessing import StandardScaler

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X 为您选择的特征数据

# 重新训练模型
model.fit(X_scaled, y)

# 获取标准化后的系数
coefficients_scaled = model.coef_.flatten()

# 替换为实际的特征列名
feature_names = [ 'athletes_count', 'sports_count', 'events_count']

# 绘制标准化后的系数图
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients_scaled, color='lightgreen')
plt.xlabel('Standardized Coefficient Value')
plt.title('Feature Coefficients in Logistic Regression (Standardized)')
plt.grid(True)
plt.show()
#

import pandas as pd
import pandas as pd
file_path = r"C:\Users\123\Desktop\summerOly_athletes.csv"
# 假设已经加载的带标签数据 df
# df 应该包含 'high_level' 和 'medal'（包含 'Gold' 或 'Silver'）列
df = pd.read_csv(file_path)
# 假设 df 包含以下列：'Year', 'Team', 'Medal' (金牌、银牌或铜牌) 和 'high_level' (高水平标签)

# 1. 第一次登场就获奖的选手的概率
# 获取每个运动员第一次登场的年份
first_appearance = df.groupby('Name').first()  # 获取每个运动员第一次登场的记录

# 判断第一次登场时是否获得奖牌（Gold、Silver 或 Bronze）
first_appearance['first_medal'] = first_appearance['Medal'].isin(['Gold', 'Silver', 'Bronze'])

# 计算第一次登场就获得奖牌的频率
first_medal_prob = first_appearance['first_medal'].mean()
print(f"第一次登场就获奖的选手概率: {first_medal_prob:.2f}")

# 2. 所有选手每次的获奖概率
# 计算每个运动员在所有比赛中获得奖牌的概率
all_medal_prob = df['Medal'].isin(['Gold', 'Silver', 'Bronze']).mean()
print(f"所有选手每次的获奖概率: {all_medal_prob:.2f}")

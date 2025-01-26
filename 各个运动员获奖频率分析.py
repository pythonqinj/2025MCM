import pandas as pd
file_path = r"C:\Users\123\Desktop\athletes_with_level_intermediate.csv"
# 假设已经加载的带标签数据 df
# df 应该包含 'high_level' 和 'medal'（包含 'Gold' 或 'Silver'）列
df = pd.read_csv(file_path)
# 高水平运动员与普通运动员的获奖频率
high_level_athletes = df[df['high_level'] == 1]  # 高水平运动员
low_level_athletes = df[df['high_level'] == 0]   # 普通运动员

# 获奖频率：Gold 或 Silver 奖牌
high_level_award_freq = high_level_athletes['Medal'].isin(['Gold', 'Silver',"Brozen"]).mean()
low_level_award_freq = low_level_athletes['Medal'].isin(['Gold', 'Silver',"Brozen"]).mean()

# 计算高水平和普通运动员的获奖频率
print(f"高水平运动员获奖频率: {high_level_award_freq:.2f}")
print(f"普通运动员获奖频率: {low_level_award_freq:.2f}")

# 第一次登场的运动员获奖频率
# 找到每个运动员第一次参赛的年份
first_appearance = df.groupby('Name').first()  # 获取每个运动员第一次登场的记录

# 判断第一次登场时是否获得奖牌（Gold 或 Silver）
first_appearance['first_medal'] = first_appearance['Medal'].isin(['Gold', 'Silver',"Brozen"])

# 计算第一次登场就获得奖牌的频率
first_medal_freq = first_appearance['first_medal'].mean()

# 输出第一次登场的获奖频率
print(f"第一次登场就获得奖牌的运动员频率: {first_medal_freq:.2f}")

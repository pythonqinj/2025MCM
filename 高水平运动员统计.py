from tqdm import tqdm
import pandas as pd
import numpy as np
file_path = r"C:\Users\123\Desktop\yuanshi.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)
df = df.sort_values(by=['Name', 'Year'])

# 创建一个空列表，用于存储每个运动员的高低水平标签
high_level = []

# 用字典存储每个运动员历史获得的金银牌信息
athlete_medals = {}

# 计算总行数
total_rows = df.shape[0]

# 遍历所有行，首先标记哪些是金牌或银牌
for i, row in tqdm(df.iterrows(), total=total_rows, desc="Processing athlete medals"):
    athlete_id = row['Name']
    medal = row['Medal']

    # 如果是金牌或银牌，更新该运动员的历史记录
    if medal in ['Gold', 'Silver']:
        if athlete_id not in athlete_medals:
            athlete_medals[athlete_id] = []
        athlete_medals[athlete_id].append(row['Year'])

# 第二个for循环：遍历每个运动员的每条记录，判断是否为高水平运动员
total_athletes = len(df['Name'].unique())  # 运动员的总数

for idx, athlete_id in enumerate(tqdm(df['Name'].unique(), desc="Processing athlete levels")):
    athlete_data = df[df['Name'] == athlete_id]
    level_tag = 0  # 默认是低水平运动员

    # 只遍历该运动员的前30条记录
    athlete_data = athlete_data.head(30)  # 截取前30条记录

    # 查找该运动员历史是否有金银牌
    for i, row in athlete_data.iterrows():
        if athlete_id in athlete_medals:
            # 判断该运动员在该年之前是否有金银牌
            history_medals = [year for year in athlete_medals[athlete_id] if year < row['Year']]
            if history_medals and level_tag == 0:
                level_tag = 1  # 如果有金牌或银牌，标记为高水平运动员

        high_level.append(level_tag)

    # 每处理完20%的运动员，保存一次中间结果
    if (idx + 1) / total_athletes >= 0.95:  # 当前处理到20%
        print(f"Processing {int((idx + 1) / total_athletes * 100)}% - Saving intermediate results")
        high_level_padded = np.pad(high_level, (0, len(df) - len(high_level)), constant_values=0)
        # 将填充后的 high_level 放入 DataFrame 的 'high_level' 列
        df['high_level'] = high_level_padded
        df.to_csv('athletes_with_level_intermediate.csv', index=False)  # 保存中间结果
# 将高低水平标签添加到数据框中

# 显示结果
print(df)


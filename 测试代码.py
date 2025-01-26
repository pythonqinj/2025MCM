import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库

# 示例数据
data = {
    'athlete_id': [1, 1, 1, 2, 2, 3, 3],
    'country': ['USA', 'USA', 'USA', 'China', 'China', 'Russia', 'Russia'],
    'year': [2015, 2016, 2017, 2015, 2016, 2015, 2016],
    'medal': ['Bronze', 'Gold', 'Silver', 'None', 'Gold', 'Silver', 'None']
}

df = pd.DataFrame(data)

# 排序：确保按运动员 ID 和年份排序
df = df.sort_values(by=['athlete_id', 'year'])

# 创建一个空列表，用于存储每个运动员的高低水平标签
high_level = [0] * len(df)  # 初始化高低水平标签列表，长度与df相同

# 用字典存储每个运动员历史获得的金银牌信息
athlete_medals = {}

# 遍历所有行，首先标记哪些是金牌或银牌
for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing athlete medals"):
    athlete_id = row['athlete_id']
    medal = row['medal']

    # 如果是金牌或银牌，更新该运动员的历史记录
    if medal in ['Gold', 'Silver']:
        if athlete_id not in athlete_medals:
            athlete_medals[athlete_id] = []
        athlete_medals[athlete_id].append(row['year'])

# 遍历每个运动员的每条记录，判断是否为高水平运动员
for athlete_id in tqdm(df['athlete_id'].unique(), desc="Processing athlete levels"):
    athlete_data = df[df['athlete_id'] == athlete_id]
    level_tag = 0  # 默认是低水平运动员

    # 只遍历该运动员的前30条记录
    athlete_data = athlete_data.head(30)  # 截取前30条记录

    # 查找该运动员历史是否有金银牌
    for i, row in athlete_data.iterrows():
        if athlete_id in athlete_medals:
            # 判断该运动员在该年之前是否有金银牌
            history_medals = [year for year in athlete_medals[athlete_id] if year < row['year']]
            if history_medals and level_tag == 0:
                level_tag = 1  # 如果有金牌或银牌，标记为高水平运动员

        # 更新当前记录的标签
        high_level[row.name] = level_tag

# 将高低水平标签添加到数据框中
df['high_level'] = high_level

# 显示结果
print(df)

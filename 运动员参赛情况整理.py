import pandas as pd

# 读取CSV文件
file_path = r"C:\Users\123\Desktop\summerOly_athletes.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 确保列名正确并一致
df.columns = df.columns.str.strip()  # 去掉列名的多余空格
df['Team'] = df['Team'].str.replace(r'-\d+$', '', regex=True)
df['Team'] = df['Team'].str.split('/').str[0]
# 按年份、国家和运动类别分组并统计运动员数量
result = (
    df.groupby(['Year', 'Team'])
    .size()
    .reset_index(name='Athlete_Count')
)

# 将结果保存为CSV文件
output_path = 'athlete_counts_by_country_sport.csv'
result.to_csv(output_path, index=False)

print("统计完成！结果已保存到：", output_path)

# 打印前几行结果供查看
print(result.head())

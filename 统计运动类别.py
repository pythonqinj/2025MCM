import pandas as pd

# 读取CSV文件
file_path = r"C:\Users\123\Desktop\yuanshi.csv"  # 替换为您的CSV文件路径
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # 去掉列名的多余空格
df['Team'] = df['Team'].str.replace(r'-\d+$', '', regex=True)
df['Team'] = df['Team'].str.split('/').str[0]
# 按年份和国家统计类别数量
result = (
    df.groupby(['Year', 'Team'])['Sport']
    .nunique()
    .reset_index(name='Sport_count')
)

# 将结果保存为新的CSV文件
output_file = 'result.csv'
result.to_csv(output_file, index=False)

# 打印结果
print(result)


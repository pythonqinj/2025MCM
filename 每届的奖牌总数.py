import pandas as pd

# 假设CSV文件名为 medals.csv
# CSV 格式：
# 年份,国家/地区,金牌,银牌,铜牌
# 2000,中国,28,16,15
# 2000,美国,37,24,32
# 2004,中国,32,17,14

# 读取CSV文件
file_path = r"C:\Users\123\Desktop\summerOly_medal_counts.csv"
data = pd.read_csv(file_path)

# 按年份计算金牌总数
gold_medals_by_year = data.groupby("Year")["Gold"].sum()

# 输出结果
print("历年金牌总数：")
print(gold_medals_by_year)

# 如果需要将结果保存到文件
output_path = "gold_medals_by_year.csv"
gold_medals_by_year.to_csv(output_path, header=["金牌总数"])


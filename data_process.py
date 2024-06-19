import pandas as pd

#初始化参数，数据标准化
filename = '内蒙古碳排放指标统计数据.xlsx'
filename1 = '离差标准化后的数据.xlsx'
data = pd.read_excel(filename,index_col ="年份（内蒙古）")
data = (data-data.min())/(data.max()-data.min())
data = data.reset_index()
data.to_excel(filename1,index = False)
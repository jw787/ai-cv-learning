## 总结实操
# 1.数据载入

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

red_df = pd.read_csv('Classes_train_annotation.csv', index_col=0)
white_df = pd.read_csv('val_data.csv', index_col=0)

print(red_df)

# 列举出索引值前五的行的内容
red_df.head()
# 列举出索引值前10的行的内容
white_df.head(n=10)

red_df.shape
white_df.shape

# # 2.添加列&合并
# red_df['color'] = ['red']*red_df.shape[0]
# red_df.head()
# white_df['color'] =[1]*white_df.shape[0]
# white_df.head()
#
# # 这是为了显示所有的列名
# red_df.columns
#
# # 用于判断对应列的标签是否一样
# print(red_df.columns==white_df.columns)
# # 用于判断对应列的标签所有相同
# print(all(red_df.columns==white_df.columns))
#
# df = pd.concat([red_df, white_df], axis="index")
#
# df['color']
#
# df.shape
#
# from collections import Counter
#
# wine_color_counter = Counter(df['color']) #{'red':xxx, 1:xxx}
# cdf = pd.DataFrame.from_dict(wine_color_counter, orient='index')
# cdf.head()
#
# cdf.plot(kind='bar')
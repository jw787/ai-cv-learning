import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(np.random.randn(4, 4), index=['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
                    columns=['one', 'two', 'three', 'four'])
# 用Debug来查看
data.drop('Beijing')
data.drop(['Shanghai', 'Shenzhen'])
data.drop('one', axis=1)
data

data.drop('Beijing')

data.drop(['Shanghai', 'Shenzhen'])

data.drop('one', axis=1)

s1 = pd.Series(np.arange(3), index=['a', 'b', 'c'])
s2 = pd.Series(np.arange(3), index=['c', 'd', 'e'])

print(s1)

print(s2)

s1 + s2

print(s1 + s2)

s1.add(s2, fill_value=0)

print(s1.add(s2, fill_value=0))

df1 = pd.DataFrame(np.random.randn(3, 3), columns=list('bcd'), index=list('123'))
df2 = pd.DataFrame(np.random.randn(3, 3), columns=list('ade'), index=list('369'))

df1

df2
# 用debug中的命令行来显示
df1 + df2

print(df1 + df2)

print(df1.add(df2))

# 将两个数据表拥有的所有数据填充生成新的数据表,共同有值的就直接相加1再加1，被填充进去的值也加1
df = df1.add(df2, fill_value=1)

df

df1 + np.ones((3, 3))

# 用debug中的命令行来显示
np.abs(data)

# 将函数作用与一维数据帧

f = lambda x: x.max() - x.min()
data.apply(f)  # apply处理的单元不是pandas结构中的单个元素，而是整行或整列

data.apply(f, axis=1)

fs = lambda x: pd.Series([x.min(), x.max(), f(x)], index=['min', 'max', 'range'])

data.apply(fs)

data.apply(fs, axis="columns")  # 和axis=1的效果是一样的

# 将map作用于Series即数据序列，只能作用于Series
s = pd.Series(np.random.randn(10), index=np.arange(10))

s

import math

s.map(lambda x: x - math.floor(x))

# 将applymap作用于数据帧中的每一个元素，只能作用于DataFrame

df = pd.DataFrame(np.random.randn(3, 3))
df

df3 = df.describe()

print(df3)
a1 = df.applymap(lambda x: "{:.4E}".format(x))
print(a1)

a2 = df.applymap(lambda x: "{:.4E}".format(x)).describe()
print(a2)

## 数据装换

# 2.1 数据缺失
# 1.删除，填补

data = pd.DataFrame([[1, 2, 4],
                     [np.nan, np.nan, 2],
                     [np.nan, 3, 3]])

data


# 将数据表中的含有NAN的值的行全部删除，即删除带有NAN值的行
data.dropna()
# 将数据表中的含有NAN的值的列全部删除，即删除带有NAN值的列
data.dropna(axis=1)
# 将数据表中的全为NAN的值的列全部删除，即删除全为NAN值的列
data.dropna(axis=1, how='all')
# 保留至少有3个非NaN的数据的行
data.dropna(thresh=3)
# 保留至少有2个非NaN的数据的行
data.dropna(thresh=2)
# 保留至少有1个非NaN的数据的行
data.dropna(thresh=1)
# 将NAN用‘10’来替换
data.fillna('10')
# 将NAN用10来替换，并且对应数据表中最高精度
data.fillna(10)
# 字典的键代表列的序号，用不同值填充
data.fillna({0:0, 1:1})
# 用前一行的数据填充, 本身有的值保持不变
data.fillna(method='ffill')
# 用后一行的数据填充，本身有的值保持不变
data.fillna(method='bfill')

# 2.2 数据转化
data = pd.DataFrame({'k1':['one','two']*3 + ['two'], 'k2':[1,1,2,3,3,4,4]})

data

# 查看魔航是否为重复行
data.duplicated()
# 删除重复行
data.drop_duplicates()
# 将括号中的“one”替换成“three”
data.replace('one', 'three')
# 查看列中的标题，即索引
data.columns
print(data.columns)

data.columns = data.columns.map(lambda x:x + "_coloumn_name")

print(data.columns)

data

ages = [20,22,25,27,21,23,27,37,31,61,45,41,32,18] # 待切割的数据
bins = [18, 25, 35, 60, 100] # 切割的范围标准
# 将这些数字自动分配bins的区间范围,区间是左开右闭
cuts = pd.cut(ages, bins)

print(cuts)
# 将区间从小到大从0到3进行标号
cuts.codes

print(type(cuts))

print(cuts.categories)

print(pd.value_counts(cuts))

s = pd.Series(np.random.randn(10)*10)


# q为组数，即自动帮划分区间，有多少个label值就划分多少个区间，依旧从小到大
pd.qcut(s, q=4, labels=['a','b','c','d'])

## 2.3 数据整理
# 合并
df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'], 'data1': range(7)})

df2 = pd.DataFrame({'key':['b','a','e'], 'data2': range(3)})
# 数据表中共同只合并key相同的值，并补填上各自没有的数据，形成新的表格（看Debug更直观一些）
pd.merge(df1, df2)
# 和上面的一样功能，即求并集
pd.merge(df1, df2, how='inner')
# 将所有的key值求并集，两个数据表中没有的数据，即key中不包含data1和data2数据用NAN代替
pd.merge(df1, df2, how='outer')
# 依照左边的数据表中的key值来赋值，没有的值用NaN代替
pd.merge(df1, df2, how='left')
# 依照右边的数据表中的key值来赋值，没有的值用NaN代替
pd.merge(df1, df2, how='right')
# 拼接
pd.concat([df1, df2], sort=False)

# 2.3.2 split，Apply，Combine

df = pd.DataFrame({'key1': ['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1': np.random.randn(5),
                   'data2': np.random.randn(5)})
df
# 按照groupby里的key进行分组
grouped = df['data1'].groupby(df['key1'])
# grouped = df['data1'].groupby('key1') # 这是不可以的
print(grouped.groups)
# data1中的数据按照分组的结果将key相同所对应的值相加求和
grouped.sum()

# data1中的数据照分组的结果将key相同所对应的值求均值，注意这里有两个key，所以要同时相同才进行求平均计算，若无则保持原值
means = df['data1'].groupby([df['key1'],df['key2']]).mean()
# 转化为表格形式
means.unstack()
# 这里循环的内容是从df中的key1对应的值来的，所以有多少类key1的值就有多少次循环
for name, group in df.groupby('key1'):
    print(name)
    print(group)

for name, group in df.groupby(['key1','key2']):
    print(name)
    print(group)

# 按照groupby所对应的方程来进行n，d值一一对应，按照索引顺序，先输出值，在输出键
for n, d in df.groupby(df.dtypes, axis=1):
    print(n)
    print(d)

df.groupby(['key1','key2'])[['data2']].mean()
print(df.groupby(['key1','key2'])[['data2']].mean())

df['data2'].groupby([df['key1'],df['key2']]).mean()
print(df['data2'].groupby([df['key1'],df['key2']]).mean())

df.groupby(['key1','key2'])['data2'].mean()

print(df.groupby(['key1','key2'])['data2'].mean())

test_df = pd.DataFrame(np.concatenate([np.random.randn(100,10),
                                       np.random.randint(0, 3, (100,1))],
                                      axis=1),
                       columns=[f'key{i}' for i in range(11)])
def top(df, n=5, columns="key0"):
    return df.sort_values(by=columns)[-n:]

test_df

test_df.groupby('key10').apply(top)



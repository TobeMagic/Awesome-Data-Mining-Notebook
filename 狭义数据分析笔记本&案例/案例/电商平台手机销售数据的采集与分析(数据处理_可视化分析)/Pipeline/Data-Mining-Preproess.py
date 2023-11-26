import pandas as pd
import re

# 3.3数据探索与预处理
# 3.3.1 数据信息探索
# 读取依据爬取好的原始数据文件
all_sales_data = pd.read_csv('../tmp/手机销售数据.csv', encoding='gbk')
after_sales_data = pd.read_csv('../tmp/手机售后数据.csv', encoding='utf-8')
all_sales_data['商品编号']=all_sales_data['商品编号'].astype(str)
# 自定义analysis函数，实现数据信息探索的描述性统计分析和缺失值分析
def analysis(data):
    print('描述性统计分析结果为：\n', data.describe())
    print('各属性缺失值占比为：\n', 100*(data.isnull().sum() / len(data)))
# 手机销售数据
print(analysis(all_sales_data))
# 手机售后数据
print(analysis(after_sales_data))



# 3.3.2缺失值处理
# 删除店铺名称、手机配色、手机内存属性的缺失值
all_sales_data.dropna(axis=0, subset = ['店铺名称'], inplace=True)
after_sales_data.dropna(axis=0, subset = ['手机配色', '手机内存'], inplace=True)
# 填充CPU型号、后摄主摄像素、前摄主摄像素、系统属性的缺失值
null_data = all_sales_data.loc[((all_sales_data['CPU型号'].isnull() == True) | 
                                (all_sales_data['前摄主摄像素'].isnull() == True) |
                                (all_sales_data['后摄主摄像素'].isnull() == True)), 
                               '商品名称'].drop_duplicates()

for j in null_data:
    for i in ['CPU型号', '后摄主摄像素', '前摄主摄像素', '系统']:
        d = all_sales_data[all_sales_data['商品名称'] == j]
        g = d[d[i].isnull() == False]
        if len(g) != 0 :
            t = list(g[i])[0]
            all_sales_data.loc[((all_sales_data['商品名称'] == j) & (all_sales_data[i].isnull())), i] = t
        else :
            all_sales_data.loc[((all_sales_data['商品名称'] == j) & (all_sales_data[i].isnull())), i] = '其他'

# 3.3.3 重复值处理
print('删除重复值前的手机销售数据维度：', all_sales_data.shape)
print('删除重复值前的手机售后数据维度：', after_sales_data.shape)

# 保留首条、删除其他条重复数据
all_sales_data = all_sales_data.drop_duplicates()
print('删除重复值后的手机销售数据维度：', all_sales_data.shape)
after_sales_data = after_sales_data.drop_duplicates()
print('删除重复值后的手机售后数据维度：', after_sales_data.shape)



# 3.3.4文本处理
# 清洗手机销售数据中的手机品牌、商品名称属性的文本内容
# 选取非括号本身及其内容的其它数据信息
all_sales_data['手机品牌'] = [i.split('（')[0] for i in all_sales_data['手机品牌']]
# 选取非【】、5G、4G、新品、手机本身及其连带的其它数据信息
all_sales_data['商品名称'] = [re.split('【.*】|5G.*|4G.*|新品.*|手机.*',i)[0] for i in all_sales_data['商品名称']]
# 将其他OS修改为其他
all_sales_data['系统'] = all_sales_data['系统'].str.replace('其他OS', '其他')


# 清洗手机售后数据中的评论文本属性的文本内容
# 删除换行符
after_sales_data['评论文本'] = after_sales_data['评论文本'].str.replace('\n', '')
# 删除html语言下的表情符号（以&开头，中间为字母，以;结束），只是处理文本中的表情符号，并不删除文本
after_sales_data['评论文本'] = after_sales_data['评论文本'].str.replace('&[a-z]+;', '')


# 清洗手机售后数据中的默认好评的文本内容
after_sales_data = after_sales_data[after_sales_data['评论文本'] !='您没有填写内容，默认好评']

# 写出数据
all_sales_data.to_csv('../tmp/处理后的手机销售数据.csv', index=False, encoding='gbk')
after_sales_data.to_csv('../tmp/处理后的手机售后数据.csv', index=False, encoding='utf-8')


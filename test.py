# -*- coding: utf-8 -*-
"""
@date: 20240616
@author: FHR
"""
'''
import pandas as pd
#初始化参数，数据标准化
filename = '内蒙古碳排放指标统计数据.xls'
filename1 = '离差标准化后的数据.xls'
data = pd.read_excel(filename,index_col ="年份（内蒙古）")
data = (data-data.min())/(data.max()-data.min())
data = data.reset_index()
data.to_excel(filename1,index = False)
'''
'''
import pandas as pd
import numpy as np
from sklearn import linear_model

filename = '内蒙古碳排放指标统计数据.xls'
data = pd.read_excel(filename,index_col ="年份（内蒙古）")
data['二氧化碳排放量/万吨'] = np.log(data['二氧化碳排放量/万吨'])
data['经济水平（人均GDP）/万元'] = np.log(data['经济水平（人均GDP）/万元'])
data['产业结构（第二产业占比）/%'] = np.log(data['产业结构（第二产业占比）/%'])
data['人口规模/万人'] = np.log(data['人口规模/万人'])
data['城镇化率/%'] = np.log(data['城镇化率/%'])
data['能源结构(煤炭占比)/%'] = np.log(data['能源结构(煤炭占比)/%'])
data['能源强度（吨标准煤/万元）'] = np.log(data['能源强度（吨标准煤/万元）'])**2
clf = linear_model.Ridge (alpha = .5)
print(data['能源强度（吨标准煤/万元）'])
#print(data.head())
'''
# -*- coding: utf-8 -*-
"""
@date: 20240616
@author: FHR
"""
import xlrd
import numpy as np
import numpy
from sklearn.linear_model import LinearRegression,Ridge 
from matplotlib.font_manager import  FontProperties
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  

trained_model = None

def read_xlrd():
    excel1 = xlrd.open_workbook(r'离差标准化后的数据.xls')
    sheet2_name = excel1.sheet_names()[0]
    sheet2 = excel1.sheet_by_name('Sheet1')
    row_list = []
    #print (sheet2.name,sheet2.nrows,sheet2.ncols)
    for i in range(sheet2.nrows):
        #print(sheet2.row_values(i))
        if i != 0:
            row_list.append(sheet2.row_values(i))
    #print (row_list)
    row_list = numpy.array(row_list,dtype='float')
    #print (row_list)
    '''
    list1 = numpy.log(row_list[:,2:7])
    list2 = numpy.log(row_list[:,7])
    print(list1)
    print(list2)
    X = numpy.insert(list1,0,values = list2,axis=1)
    print(X)  
    '''
    X = row_list[:,2:]
    Y = row_list[:,1]
    Z = row_list[:,0]
    # print(X,Y)
    return X,Y,Z     
def Linear():
    global trained_model
    X,Y,Z = read_xlrd()    
    model = Ridge(alpha = 0.000001)#设置线性回归
    model.fit(X, Y)     # 训练模型
    trained_model = model  # 保存训练好的模型  
    a = model.coef_     #各个参数的权重
    b = model.intercept_#截距b的值 Y=a1*x1+a2*x2+a3*x3+b
    # 输出错误差值平方均值 
    print('各个参数的权重：',a)
    # print('截距：',b)
    x_test = X[20:]#测试数据的X数组的值
    y_test = Y[20:]#测试数据的Y的值
    print(y_test)
    predictions = model.predict(x_test)#选择后11个数据作为测试数据
    for i, prediction in enumerate(predictions):
        print('预测: %s, 结果: %s' % (prediction, y_test[i]))
    c = np.mean((model.predict(x_test) - y_test) ** 2)
    print('错误差值平方均值',c)
    a = model.score(x_test,y_test)
    print('得分：%.2f' %a)#模型得分 
def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
def matplotlab_train_samples():  
    X, Y, Z = read_xlrd()  
  
    plt.figure(figsize=(10, 5))  
    plt.ylabel('二氧化碳排放量/万吨', fontproperties=getChineseFont())  
    plt.xlabel("年份", fontproperties=getChineseFont())  
    # 训练样本（蓝色三角）  
    plt.scatter(Z[:20], Y[:20], color='blue', marker='^')  
    plt.title('训练样本', fontproperties=getChineseFont())  
    plt.show()  
  
def matplotlab_test_samples():  
    X, Y, Z = read_xlrd()  
  
    plt.figure(figsize=(10, 5))  
    plt.ylabel('二氧化碳排放量/万吨', fontproperties=getChineseFont())  
    plt.xlabel("年份", fontproperties=getChineseFont())  
    # 测试样本（红色星号）  
    plt.scatter(Z[20:], Y[20:], color='red', marker='*')  
    plt.title('测试样本', fontproperties=getChineseFont())  
    plt.show()  
  
def matplotlab_regression_line():  
    X, Y, Z = read_xlrd()  
      
    plt.ylabel('二氧化碳排放量/万吨', fontproperties=getChineseFont())  
    plt.xlabel("年份", fontproperties=getChineseFont())  
      
    # 绘制训练样本（蓝色三角）  
    plt.scatter(Z[:20], Y[:20], color='blue', marker='^', label='训练样本')  
      
    # 绘制测试样本（红色星号）  
    plt.scatter(Z[20:], Y[20:], color='red', marker='*', label='测试样本')  
    
    # 定义新的插值点，从0到30均匀分布100个点  
    new_points = np.linspace(0, 30, 100)  
    
    # 初始化一个新的100x6的矩阵来存储插值结果  
    X_interpolated = np.zeros((100, 6))  
    
    # 对每一列进行插值  
    for i in range(6):  
        f = interp1d(np.arange(31), X[:, i], kind='cubic')  # 使用三次立方插值  
        X_interpolated[:, i] = f(new_points) 
    # print(X_interpolated.shape)

    # 绘制回归线（绿色实线）  
    if trained_model is not None:  
        # 注意：这里的line_X_features构造方式可能需要根据实际数据的特征进行调整  
        line_X = np.linspace(Z.min(), Z.max(), 100).reshape(-1, 1)  
        line_X_features = np.hstack((line_X, np.zeros((line_X.shape[0], X.shape[1] - 1))))  
        line_Y_pred = trained_model.predict(X_interpolated)  
        plt.plot(line_X, line_Y_pred, color='green', linestyle='-', label='回归线')  
      
    plt.legend(prop=getChineseFont())  # 添加图例，并使用中文字体  
    plt.title('年份与碳排放的关系及回归线图', fontproperties=getChineseFont())  
    plt.show()

def test_model():
    X,Y,Z = read_xlrd()    
    print(X)
    
    predictions = trained_model.predict(X)#选择后11个数据作为测试数据
    print(predictions)
    print(Y)


if __name__=='__main__':
    Linear()  # 先训练模型  
    # matplotlab_train_samples()  # 绘制训练样本图  
    # matplotlab_test_samples()  # 绘制测试样本图  
    matplotlab_regression_line()  # 绘制回归线图
    # test_model()
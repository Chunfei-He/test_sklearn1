# -*- coding: utf-8 -*-
"""
@date: 20240616
@author: FHR
@description: 读取内蒙古碳排放指标统计数据.xls文件，使用sklearn库中的线性回归模型进行线性回归分析
"""
"""
@date: 20240626
@description: 
"""
import xlrd
import numpy
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontManager, FontProperties
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor 
def read_xlrd():
    excel1 = xlrd.open_workbook(r'./内蒙古碳排放指标统计数据.xls')
    sheet2_name = excel1.sheet_names()[0]
    sheet2 = excel1.sheet_by_name('Sheet1')
    # sheet的名称，行数，列数
    row_list = []
    #print (sheet2.name,sheet2.nrows,sheet2.ncols)
    for i in range(sheet2.nrows):
        #print(sheet2.row_values(i))
        if i != 0:
            row_list.append(sheet2.row_values(i))
    #print (row_list)
    row_list = numpy.array(row_list,dtype='float')
    #print (row_list)
    X = row_list[:,2:]
    Y = row_list[:,1]
    Z = row_list[:,0]
    #print(X,Y)
    return X,Y,Z     # 梯度下降算法实现  
def gradient_descent(X, Y, learning_rate=0.01, iterations=1000):  
    m, n = X.shape  
    theta = np.zeros(n)  
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # 添加截距项  
      
    for i in range(iterations):  
        prediction = np.dot(X, theta)  
        error = prediction - Y  
        gradient = (1/m) * np.dot(X.T, error)  
        theta -= learning_rate * gradient  
          
    return theta  
  
def SGDLinear():  
    X, Y, Z = read_xlrd()  
      
    # 设置SGDRegressor模型，并给出迭代次数和步长  
    model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)  
    model.fit(X, Y)  # 训练模型  
      
    # 获取训练后的参量  
    a = model.coef_  # 各个特征的权重  
    b = model.intercept_  # 截距b的值 Y = a1*x1 + a2*x2 + ... + an*xn + b  
    print('各个参数的权重：', a)  
    print('截距：', b)  
  
def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

def matplotlab1_with_regression_line():  
    X, Y, Z = read_xlrd()  
      
    # 训练SGDRegressor模型（这部分可以与SGDLinear函数中的模型训练部分合并，以避免重复训练）  
    model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)  
    model.fit(X, Y)  
      
    # 使用模型预测整个X范围的值，以绘制回归线  
    X_for_prediction = numpy.linspace(X.min(axis=0), X.max(axis=0), 100)  # 生成一个用于预测的X值范围  
    Y_predicted = model.predict(X_for_prediction)  # 预测Y值  
      
    # 绘制原始数据点  
    plt.subplot(3, 1, 3)  
    plt.ylabel('二氧化碳排放量/万吨', fontproperties=getChineseFont())  
    plt.xlabel("年份", fontproperties=getChineseFont())  
    plt.plot(Z, Y, color='blue', marker='o', linestyle='none')  # 修改linestyle为'none'以仅显示数据点  
      
    # 绘制回归线  
    plt.plot(Z[0:len(Y_predicted)], Y_predicted, color='green', linestyle='solid')  # 假设Z与X有相同的范围和间隔  
      
    plt.title('年份与碳排放的关系图', fontproperties=getChineseFont())  
      
    # ...（其他子图的代码保持不变）  
      
    plt.show()  
if __name__=='__main__':
    SGDLinear()
    matplotlab1_with_regression_line()
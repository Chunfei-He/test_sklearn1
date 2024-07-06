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
from scipy.interpolate import interp1d  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.model_selection import train_test_split, cross_val_score  
  

trained_model = None

def read_xlrd():
    excel1 = xlrd.open_workbook(r'C:\private\代码\test_sklearn1\平时作业2\离差标准化后的数据.xls')
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
    global trained_model
    X, Y, Z = read_xlrd()  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  
    
    # 设置SGDRegressor模型，并给出迭代次数和步长  
    model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)  
    model.fit(X_train, Y_train)  # 训练模型  
    
    # 使用测试集进行预测  
    Y_pred = model.predict(X_test)  
    
    # 计算MSE和R² 分数  
    mse = mean_squared_error(Y_test, Y_pred)  
    r2 = r2_score(Y_test, Y_pred)  
    
    print('均方误差（MSE）:', mse)  
    print('R² 分数:', r2)  

    trained_model = model
    # 获取训练后的参量  
    a = model.coef_  # 各个特征的权重  
    b = model.intercept_  # 截距b的值 Y = a1*x1 + a2*x2 + ... + an*xn + b  
    print('各个参数的权重：', a)  
    print('截距：', b)  
  
def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

def matplotlab1_with_regression_line():  
    X, Y, Z = read_xlrd()  
    # print(X.shape)
    
    # 定义新的插值点，从0到30均匀分布100个点  
    new_points = np.linspace(0, 30, 100)  
    
    # 初始化一个新的100x6的矩阵来存储插值结果  
    X_interpolated = np.zeros((100, 6))  
    
    # 对每一列进行插值  
    for i in range(6):  
        f = interp1d(np.arange(31), X[:, i], kind='cubic')  # 使用三次立方插值  
        X_interpolated[:, i] = f(new_points) 
    
    # print(X_interpolated)

    # 使用模型预测整个X范围的值，以绘制回归线  
    # X_for_prediction = numpy.linspace(X.min(axis=0), X.max(axis=0), 100)  # 生成一个用于预测的X值范围  
    Y_predicted = trained_model.predict(X_interpolated)  # 预测Y值  
    # print(Y_predicted)
    # 绘制原始数据点  
    plt.ylabel('二氧化碳排放量/万吨', fontproperties=getChineseFont())  
    plt.xlabel("年份", fontproperties=getChineseFont())  
    plt.plot(Z, Y, color='blue', marker='o', linestyle='none')  # 修改linestyle为'none'以仅显示数据点  
    
    line_X = np.linspace(Z.min(), Z.max(), 100).reshape(-1, 1)
    
    # 绘制回归线  
    plt.plot(line_X, Y_predicted, color='green', linestyle='solid')  # 假设Z与X有相同的范围和间隔  
      
    plt.title('年份与碳排放的关系图', fontproperties=getChineseFont())  
      
    # ...（其他子图的代码保持不变）  
    plt.savefig('2.2.3年份与碳排放的关系图.png')
    plt.show()  
if __name__=='__main__':
    SGDLinear()
    matplotlab1_with_regression_line()
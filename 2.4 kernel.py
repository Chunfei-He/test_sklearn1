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
from sklearn.kernel_ridge import KernelRidge 
import matplotlib.pyplot as plt

trained_model = None  

def read_xlrd():
    excel1 = xlrd.open_workbook(r'内蒙古碳排放指标统计数据.xls')
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
    return X,Y,Z     

def KernelRidgeRegression():  
    global trained_model
    X, Y, Z = read_xlrd()  
      
    # 使用KernelRidge模型  
    model = KernelRidge(alpha=1.0, kernel='linear')  # 也可以使用其他核，比如'rbf'、'poly'等  
    model.fit(X, Y)  
    trained_model = model
    # 由于KernelRidge不提供直接的coef_和intercept_属性，因此无法直接打印权重和截距  
    # 但可以通过模型对训练数据进行预测来评估模型性能  
      
    x_test = X[20:]  # 测试数据的X数组的值  
    y_test = Y[20:]  # 测试数据的Y的值  
    predictions = model.predict(x_test)  # 选择后部分数据作为测试数据  
      
    # 打印预测结果与实际结果的对比  
    for i, prediction in enumerate(predictions):  
        print(f'预测: {prediction}, 结果: {y_test[i]}')  
      
    # 计算模型得分  
    score = model.score(x_test, y_test)  
    print(f'得分：{score:.2f}')  # 模型得分  
  

def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
def matplotlab1():
    X,Y,Z = read_xlrd()
    # 生成用于绘图的等距X值  
    X_plot = numpy.linspace(X[:, 0].min(), X[:, 0].max(), 100)[:, None]  
    Y_plot = trained_model.predict(X)  # 使用模型预测这些X值对应的Y值  
  
    # 绘制原始数据点（可选）  
    plt.scatter(X[:, 0], Y, c='blue', label='实际数据')  
  
    # 绘制回归线  
    plt.plot(X_plot[:, 0], Y_plot, color='green', label='Kernel线性回归')  
  
    plt.xlabel('X值')  
    plt.ylabel('Y值')  
    plt.title('Kernel线性回归拟合结果')  
    plt.legend()  
    plt.show()  
  
    plt.show()
#read_xlrd()
if __name__=='__main__':
    KernelRidgeRegression()  
    matplotlab1()
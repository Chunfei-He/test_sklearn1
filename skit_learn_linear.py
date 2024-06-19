# -*- coding: utf-8 -*-
"""
@date: 20240616
@author: FHR
@description: 读取内蒙古碳排放指标统计数据.xls文件，使用sklearn库中的线性回归模型进行线性回归分析
"""
import xlrd
import numpy
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontManager, FontProperties
import matplotlib.pyplot as plt
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
    return X,Y,Z     
def Linear():
    X,Y,Z = read_xlrd()    
    model = LinearRegression()#设置线性回归
    model.fit(X, Y)     # 训练模型
    a = model.coef_     #各个参数的权重
    b = model.intercept_#截距b的值 Y=a1*x1+a2*x2+a3*x3+b
    print('各个参数的权重：',a)
    print('截距：',b)
    x_test = X[20:]#测试数据的X数组的值
    y_test = Y[20:]#测试数据的Y的值
    predictions = model.predict(x_test)#选择后11个数据作为测试数据
    for i, prediction in enumerate(predictions):
        print('预测: %s, 结果: %s' % (prediction, y_test[i]))
    a = model.score(x_test,y_test)
    print('得分：%.2f' %a)#模型得分 
def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
def matplotlab1():
    X,Y,Z = read_xlrd()
    plt.subplot(3,1,3)
    plt.ylabel('二氧化碳排放量/万吨',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())  
    plt.plot(Z,Y, color='blue', marker='o', linestyle='solid')
    plt.title('年份与碳排放的关系图', fontproperties=getChineseFont())
    plt.subplot(3,3,1)
    plt.ylabel('经济水平（人均GDP）/万元',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与人均GDP的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,0], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,2)
    plt.ylabel('产业结构（第二产业占比）/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与第二产业占比%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,1], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,3)
    plt.ylabel('人口规模/万人',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与人口规模/万人的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,2], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,4)
    plt.ylabel('城镇化率/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与城镇化率/%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,3], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,5)
    plt.ylabel('能源结构(煤炭占比)/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与能源结构(煤炭占比)/%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,4], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,6)
    plt.ylabel('能源强度（吨标准煤/万元）',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与能源强度（吨标准煤/万元）的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,5], color='red', marker='o', linestyle='solid')
    plt.show()
#read_xlrd()
if __name__=='__main__':
    Linear()
    matplotlab1()
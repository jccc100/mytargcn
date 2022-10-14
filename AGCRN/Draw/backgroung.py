# _*_ coding: utf-8 _*_
# @time     :2022/3/22 14:11
# @Author   :jc
# @File     :backgroung.py
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# 插入背景图片
path=r'C:\Users\jc\Desktop\毕业相关\bg_img'
img=plt.imread(path+'\PEMS08_MAE.png')
fig,ax=plt.subplots()
# 指定图片的高度和宽度
# ax.imshow(img)

# x = [10,20,30,40,50,60,70,80,90,100]
# y1 = [7,17,27,37,43,49,57,65,71,77]
# y2 = [7,17,27,37,45,54,59,67,75,83]
# y3 = [8,18,28,38,47,56,64,73,80,89]
# y4 = [10,20,30,40,50,60,70,80,90,100]
# plt.plot(x,y1,color='grey',linewidth=2.0,linestyle='-')
# plt.plot(x,y2,color='orange',linewidth=2.0,linestyle='-')
# plt.plot(x,y3,color='blue',linewidth=2.0,linestyle='-')
# plt.plot(x,y4,color='red',linewidth=2.0,linestyle='-')
plt.tick_params(right=True, top=True,labelsize=10, direction='in', pad=3, length=4)
x_ticks = np.linspace(-5, 5, 13)#发现如果是np.arrange（5，-5,8）就不能画上去
x_labels = ['0', '5', '10', '15', '20', '25','30','35','40','45','50','55','60']
y_ticks = np.linspace(-3, 2, 9)
y_labels = ['12.5', '15.0', '17.5', '20.0', '22.5', '25.0','27.5','30.0','32.5']
plt.xticks(x_ticks, x_labels, fontsize=9)
plt.yticks(y_ticks, y_labels, fontsize=9)

plt.xlabel('Xlabel')
plt.ylabel('Ylabel')

# 设置小图标
plt.legend(['A','B','C','D'],loc='upper left',fontsize = 10)
plt.show()

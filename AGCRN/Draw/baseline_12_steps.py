import matplotlib.pyplot as plt
import numpy as np


class PeMSD8():
    def __init__(self):
        super(PeMSD8, self).__init__()

    def MAE(self):
        VAR  = [17.78, 18.93, 19.52, 20.45, 21.26, 22, 22.97, 23.72, 24.45, 25.04, 25.6, 26.19]
        SAR  = [14.48, 16.26, 17.94,19.4, 20.92, 22.33, 23.98, 25.55, 27.03, None,None,None]
        DCRNN  = [13.5,14.81,15.66,16.46,17.18,17.88,18.66,19.36,20.19,20.84,21.64,22.38]
        ASTGCN  = [14.35,15.59,16.48,17.24,18.05,18.69,19.36,20,20.62,21.15,21.89,22.41]
        STSGCN  = [14.45,15.28,15.67,16.08,16.5,16.95,17.38,17.81,18.14,18.61,18.94,19.49]
        STGCN  = [13.13,14.29,15.38,16.11,16.9,17.73,18.52,19.32,20.16,20.87,21.51,22.36]
        AGCRN  = [14.12,14.45,15.01,15.35,15.68,16.02,16.5,16.85,17.16,17.47,17.91,18.53]
        # Ours_model=[13.46,13.85,14.25,14.58,14.87,15.15,15.42,15.70,15.95,16.17,16.46,16.86] # 622
        Ours_model=[13.43, 13.87, 14.25, 14.35, 14.67, 15.00, 15.28, 15.50, 15.85, 16.10, 16.52, 16.86]
        # Ours_model=[13.57,13.86,14.22,14.49,14.73,14.99,15.28,15.58,15.81,15.95,16.13,16.52] # 712
        average = 0
        for i in Ours_model:
            average = average + i
        print("RMSE:", average / 12)

        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 MAE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAE for PEMSD8")

        plt.plot(my_x_ticks,VAR [:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR [:],'o',color='lime')

        plt.plot(my_x_ticks,SAR [:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR [:], 'go')

        plt.plot(my_x_ticks,DCRNN [:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN [:], 'bo')

        plt.plot(my_x_ticks,ASTGCN [:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN [:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN [:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN [:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN [:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN [:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN [:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN [:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')



        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        VAR = [26.99,28.77,29.77,31.09,32.19,33.41,34.51,35.71,36.62,37.47,38.28,39]
        SAR = [22.16,25.17,27.74,30.27,31.41,34.53,36.68,38.86,40.95,None,None,None]
        DCRNN= [20.75,22.98,24.81,25.77,26.62,27.92,28.97,29.91,31.12,31.91,32.96,34.72]
        ASTGCN = [21.77,23.65,25.38,26.51,27.32,28.19,29.18,30.2,30.78,31.65,32.49,33.75]
        STSGCN= [22.3,23.85,24.34,25.24,25.92,26.59,27.3,28,28.48,29.06,29.62,30.51]
        STGCN = [20.38,22.28,23.78,25.24,26.28,27.3,28.49,29.62,30.7,31.65,32.69,33.72]
        AGCRN = [21.95,22.78,23.61,24.21,24.88,25.58,26.22,26.8,27.33,27.84,28.5,29.4]

        # Ours_model=[21.05,21.85,22.66,23.25,23.72,24.22,24.85,25.25,25.6,26.01,26.38,26.94]
        # Ours_model=[21.16,21.85,22.55,23.13,23.63,24.12,24.62,25.11,25.54,25.86,26.20,26.73]# 712
        Ours_model =[21.01,21.97,22.85,23.62,24.20,24.65,25.20,25.64,26.12,26.48,26.84,27.35]
        average_RMSE=0
        for i in Ours_model:
            average_RMSE=average_RMSE+i
        print("RMSE:",average_RMSE/12)
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 RMSE")
        plt.xlabel("Prediction interval")
        plt.ylabel("RMSE for PEMSD8")

        plt.plot(my_x_ticks,VAR[:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'o',color='lime')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()
    def MAPE(self):
        VAR = [11.21,12.18,12.52,13.16,13.69,14.31,14.9,15.43,15.94,16.42,16.89,17.32]
        SAR = [9.05,10.11,11.08,12,13,14.05,15.09,16.14,17.32,None,None,None]
        DCRNN= [8.86,9.5,10,10.46,10.89,11.34,11.82,12.32,12.75,13.12,13.63,14.21]
        ASTGCN = [9.95,10.57,11.03,11.48,11.82,12.25,12.7,13.1,13.52,13.92,14.5,15.23]
        STSGCN= [9.48,9.85,10.14,10.35,10.57,10.81,11.07,11.28,11.48,11.66,11.91,12.27]
        STGCN = [8.66,9.28,9.77,10.23,10.61,11.05,11.46,11.85,12.26,12.64,12.95,13.34]
        AGCRN = [9.27,9.45,9.7,9.82,10.02,10.25,10.51,10.74,10.85,11.08,11.33,11.69]
        # Ours_model=[8.73,8.93,9.15,9.32,9.51,9.68,9.91,10.12,10.27,10.50,10.71,11.01]
        # Ours_model=[9.03,9.13,9.32,9.47,9.57,9.75,9.89,10.04,10.16,10.29,10.42,10.69]
        Ours_model=[9.10, 9.31, 9.43, 9.62, 9.77, 9.91, 10.08, 10.18, 10.27, 10.38, 10.45, 10.65]
        average=0
        for i in Ours_model:
            average=average+i
        print("MAPE:",average/12)
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 MAPE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAPE for PEMSD8")

        plt.plot(my_x_ticks,VAR[:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'o',color='lime')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()

class PeMSD4():
    def __init__(self):
        super(PeMSD4, self).__init__()

    def MAE(self):
        VAR = [19.52,20.93,21.97,22.72,23.23,23.72,24.21,24.66,25.46,25.79,26.15,26.95]
        SVR = [18.84,20.67,22.5,24.23,25.94,27.72,29.51,None,None,None,None,None]
        DCRNN  = [17.62,19.05,20.3,21.28,22.19,23.17,24.17,25.22,26.16,27.07,28.14,29.32]
        ASTGCN  = [18.15,19.23,20.15,20.82,21.43,22.15,22.8,23.41,23.58,24.76,25.68,26.55]
        STSGCN  = [17.75,18.59,19.16,19.63,20.16,20.6,21.09,21.55,21.95,22.34,22.86,23.38]
        STGCN  = [16.8,18.14,19.23,20.28,21.06,21.83,22.74,23.72,24.86,25.48,26.19,26.97]
        AGCRN  = [18.8,18.95,19.11,19.28,19.41,19.66,19.84,20.08,20.27,20.42,20.81,21.44]

        # Ours_model=[18.05,18.18,18.45,18.71,18.92,19.08,19.26,19.58,19.8,19.95,20.16,20.55] #622
        Ours_model=[17.8,18.02,18.35,18.65,18.86,19.03,19.21,19.40,19.54,19.66,19.86,20.28] # 712
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']
        # plt.yticks([0,18,20,22,24,26,28,30])

        plt.title("PEMSD4 MAE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAE for PEMSD4")

        plt.plot(my_x_ticks,VAR [:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR [:],'o',color='lime')

        plt.plot(my_x_ticks,SVR [:], 'green',label="SVR")
        plt.plot(my_x_ticks,SVR [:], 'go')

        plt.plot(my_x_ticks,DCRNN [:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN [:], 'bo')

        plt.plot(my_x_ticks,ASTGCN [:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN [:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN [:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN [:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN [:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN [:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN [:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN [:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')



        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        VAR = [31.27,33.2,34.4,35.39,35.97,36.6,37.21,37.89,38.47,39.01,39.7,40.35]
        # SVR = [29.95,32.65,35.31,37.81,40.15,42.77,45.08,47.5,50,53.89,57.2,62]
        SVR = [29.95,32.65,35.31,37.81,40.15,42.77,45.00,None,None,None,None,None]
        DCRNN= [28.09,30.28,32.15,33.47,34.76,36.37,37.58,39.02,40.39,41.71,43.28,44.93]
        ASTGCN = [28.55,30.27,31.48,32.46,33.31,34.14,35.28,36.41,36.95,37.81,38.95,40.15]
        STSGCN= [28.88,29.95,31.52,32.14,32.76,33.59,34.28,34.81,35.75,36.58,37.22,37.85]
        STGCN = [27.12,29.06,30.72,32.07,33.21,34.28,35.51,36.77,38,39.06,40.02,41.15]
        AGCRN = [29.83,30.22,30.72,31.24,31.72,31.99,32.56,32.98,33.32,33.74,34.12,34.97]

        # Ours_model=[29.62,30.09,30.62,31.25,31.67,31.84,32.28,32.78,33.18,33.48,33.86,34.38]
        Ours_model=[29.14,29.69,30.31,30.83,31.28,31.62,31.99,32.34,32.64,32.90,33.21,33.73]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD4 RMSE")
        plt.xlabel("Prediction interval")
        plt.ylabel("RMSE for PEMSD4")

        plt.plot(my_x_ticks,VAR[:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'o',color='lime')

        plt.plot(my_x_ticks,SVR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SVR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()
    def MAPE(self):
        VAR = [14.09,15.4,16.37,17,17.52,17.97,18.56,19.03,19.5,19.96,20.47,20.96]
        # SVR = [12.27,13.46,14.69,15.97,16.96,18.64,19.57,20.83,21.18,23.66,25.37,26.73]
        SVR = [12.27,13.46,14.69,15.97,16.96,18.64,19.57,20.83,21.18,None,None,None]
        DCRNN= [11.8,12.97,13.8,14.37,15.05,15.27,16.4,17.03,17.89,18.46,19.23,20.08]
        ASTGCN = [12.8,13.44,14.07,14.58,15.08,15.52,16.08,16.6,17.07,17.7,18.37,19.14]
        STSGCN= [11.97,12.45,12.83,13.11,13.33,13.57,13.83,14.17,14.47,14.78,14.95,15.53]
        STGCN = [11.36,12.17,12.87,13.36,13.76,14.16,14.63,15.17,15.61,16.03,16.4,16.84]
        AGCRN = [12.12,12.18,12.35,12.48,12.54,12.69,12.94,13.17,13.32,13.48,13.76,14.12]

        Ours_model=[12.11,12.23,12.36,12.43,12.55,12.69,12.86,12.97,13.00,13.05,13.14,13.46]# 712
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD4 MAPE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAPE for PEMSD4")

        plt.plot(my_x_ticks,VAR[:], 'lime',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'o',color='lime')

        plt.plot(my_x_ticks,SVR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SVR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'red',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='red')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()

class PeMSD7():
    def __init__(self):
        super(PeMSD7, self).__init__()

    def MAE(self):
        VAR = []
        SAR = []
        DCRNN  = [18.55,20.52,21.33,22.05,22.65,23.38,24.03,24.69,25.43,26.11,26.81,27.56]
        ASTGCN  = [19.22,21.21,22.59,23.74,24.88,25.59,26.62,27.59,28.43,29.44,30.72,31.98]
        STSGCN  = [18.38,20,21.21,22.1,22.95,23.74,24.57,25.34,26.14,26.86,27.65,28.62]
        STGCN  = [17.65,19.92,21.93,23.61,25.09,26.62,28.19,29.92,31.74,33.19,34.69,36.33]
        AGCRN  = [19.02,19.4,19.98,20.52,20.99,21.34,21.65,21.95,22.32,22.45,22.85,23.64]

        Ours_model=[18.27,19,19.69,20.23,20.71,21.07,21.37,21.68,21.96,22.25,22.47,22.85]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMS07 MAE")
        plt.xlabel("Time")
        plt.ylabel("MAE")

        # plt.plot(my_x_ticks,VAR [:], 'red',label="VAR")
        # plt.plot(my_x_ticks,VAR [:],'ro')

        # plt.plot(my_x_ticks,SAR [:], 'green',label="SVR")
        # plt.plot(my_x_ticks,SAR [:], 'go')

        plt.plot(my_x_ticks,DCRNN [:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN [:], 'bo')

        plt.plot(my_x_ticks,ASTGCN [:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN [:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN [:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN [:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN [:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN [:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN [:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN [:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'lime',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='lime')



        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        # VAR = [31.27,33.2,34.4,35.39,35.97,36.6,37.21,37.89,38.47,39.01,39.7,40.35]
        # SAR = [29.95,32.65,35.31,37.81,40.15,42.77,45.08,47.5,50,53.89,57.2,62]
        DCRNN= [28.8,31.34,32.96,34.16,35.23,36.28,37.26,38.1,39,39.86,40.92,42.04]
        ASTGCN = [29.5,32.62,34.64,36.34,37.68,38.96,40.36,41.56,42.82,44.16,45.78,47.32]
        STSGCN= [29.3,32.18,34.1,35.84,37.36,38.8,40.36,41.48,42.64,44.02,45.36,46.76]
        STGCN = [27.54,30.92,33.38,35.7,37.6,39.58,41.6,43.66,45.84,47.62,49.52,51.56]
        AGCRN = [30.53,31.63,32.68,33.69,34.45,35.09,35.89,36.44,36.97,37.65,38.33,38.98]

        Ours_model=[29.41,31.1,32.47,33.61,34.57,35.36,36.06,36.72,37.3,37.78,38.16,38.66]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMS07 RMSE")
        plt.xlabel("Time")
        plt.ylabel("RMSE")

        # plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        # plt.plot(my_x_ticks,VAR[:],'ro')

        # plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        # plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'lime',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='lime')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()
    def MAPE(self):
        # VAR = [14.09,15.4,16.37,17,17.52,17.97,18.56,19.03,19.5,19.96,20.47,20.96]
        # SAR = [12.27,13.46,14.69,15.97,16.96,18.64,19.57,20.83,21.18,23.66,25.37,26.73]
        DCRNN= [8.42,8.92,9.26,9.61,9.88,10.16,10.46,10.8,11.03,11.34,11.68,12.12]
        ASTGCN = [8.65,9.3,9.91,10.39,10.88,11.42,11.96,12.61,13.21,13.88,14.65,15.46]
        STSGCN= [7.85,8.5,8.96,9.34,9.67,10.02,10.3,10.7,11,11.29,11.66,12.1]
        STGCN = [7.8,8.88,9.73,10.45,11.11,11.8,12.66,13.49,14.26,14.95,15.61,16.3]
        AGCRN = [8.1,8.19,8.4,8.6,8.75,8.91,9.1,9.21,9.32,9.39,9.53,9.83]

        Ours_model=[8.04,8.23,8.39,8.56,8.74,8.89,8.99,9.11,9.23,9.39,9.51,9.7]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMS07 MAPE")
        plt.xlabel("Time")
        plt.ylabel("MAPE(%)")

        # plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        # plt.plot(my_x_ticks,VAR[:],'ro')

        # plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        # plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN[:], 'o',color='black')

        plt.plot(my_x_ticks,Ours_model[:], 'lime',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='lime')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()

class ablation_result():
    def __init__(self):
        super(ablation_result,self).__init__()

    def MAE(self):
        # TA_GCN_linear  = [17.04,17.16,17.35,17.66,17.97,18.33,18.81,19.27,19.56,19.92,20.45,21.24]
        # TA_GCN_SA  =     [14.47,15.2,15.8,16.23,16.57,16.86,17.23,17.59,17.9,18.21,18.65,19.33]
        # TA_GCN_noTA  =   [13.79,14.37,14.64,14.97,15.43,15.62,15.95,16.25,16.56,16.82,17.29,17.64]
        # TARGCN=         [13.46,13.85,14.25,14.58,14.87,15.15,15.42,15.70,15.95,16.17,16.46,16.86]
        TA_GCN_linear = [14.14,14.87,15.57,16.17,16.70,17.25,17.78,18.27,18.66,19.07,19.55,20.23]
        TA_GCN_SA = [13.78,14.55,15.24,15.86,16.43,16.99,17.55,18.10,18.64,19.16,19.75,20.54]
        # TA_GCN_noTA = [13.49, 14.01, 14.33, 14.81, 15.12, 15.45, 15.86, 16.22, 16.49, 16.78, 17.12, 17.55] # static
        TA_GCN_noTA = [13.88,14.26,14.51,14.88,15.21,15.72,15.98,16.23,16.65,16.92,17.22,17.43] # nostatic
        # TA_GCN_noTA = [14.00,14.26,14.38,14.55,14.82,15.32,15.68,16.23,16.49, 16.78, 17.12, 17.55] # nostatic
        TARGCN =      [13.43, 13.87, 14.25, 14.35, 14.67, 15.00, 15.28, 15.50, 15.85, 16.10, 16.52, 16.86]
        linear = 0
        sa = 0
        nota = 0
        our = 0
        for i in range(12):
            linear = linear + TA_GCN_linear[i]
            sa = sa + TA_GCN_SA[i]
            nota = nota + TA_GCN_noTA[i]
            our = our + TARGCN[i]
        print("MAE:")
        print("linear:", linear / 12, "sa:", sa / 12, "nota:", nota / 12, "our:", our / 12)
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 MAE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAE for PEMSD8")

        plt.plot(my_x_ticks,TA_GCN_linear [:], 'peru',label="TARGCN-linear")
        plt.plot(my_x_ticks,TA_GCN_linear [:],'o',color='peru')

        plt.plot(my_x_ticks,TA_GCN_SA [:], 'green',label="TARGCN-SA")
        plt.plot(my_x_ticks,TA_GCN_SA [:], 'go')

        plt.plot(my_x_ticks,TA_GCN_noTA [:],'b',label="TARGCN-noTA")
        plt.plot(my_x_ticks,TA_GCN_noTA [:], 'bo')

        plt.plot(my_x_ticks,TARGCN [:], 'red',label="TARGCN")
        plt.plot(my_x_ticks,TARGCN [:], 'o',color='red')





        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        # TA_GCN_linear = [26.37,26.63,27,27.49,27.97,28.45,29.01,29.51,29.86,30.28,30.93,31.89]
        # TA_GCN_SA = [22.48,23.74,24.75,25.47,26.02,26.49,27.02,27.52,27.97,28.43,29.08,30.04]
        # TA_GCN_noTA = [21.68,22.45,23.15,23.92,24.37,24.74,25.46,25.97,26.46,26.82,27.23,27.94]
        # TA_GCN = [21.05,21.85,22.66,23.25,23.72,24.22,24.85,25.25,25.6,26.01,26.38,26.94]
        TA_GCN_linear = [22.04,23.50,24.79,25.92,26.90,27.83,28.72,29.54,30.23,30.94,31.73,32.77]
        TA_GCN_SA = [21.34,22.79,24.06,25.15,26.10,27.02,27.93,28.81,29.63,30.45,31.33,32.45]
        # TA_GCN_noTA = [21.29,22.17,22.99,23.88,24.35,24.95,25.62,25.98,26.55,26.95,27.38,27.86]# static
        TA_GCN_noTA = [21.56,22.41,23.46,24.23,24.74,25.25,25.79,26.25,26.62,27.05,27.42,27.87]# static
        TA_GCN =      [21.01,21.97,22.85,23.62,24.20,24.65,25.20,25.64,26.12,26.48,26.84,27.35]
        linear = 0
        sa = 0
        nota = 0
        our = 0
        for i in range(12):
            linear = linear + TA_GCN_linear[i]
            sa = sa + TA_GCN_SA[i]
            nota = nota + TA_GCN_noTA[i]
            our = our + TA_GCN[i]
        print("RMSE:")
        print("linear:", linear / 12, "sa:", sa / 12, "nota:", nota / 12, "our:", our / 12)
        my_x_ticks = np.arange(5, 65, 5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 RMSE")
        plt.xlabel("Prediction interval")
        plt.ylabel("RMSE for PEMSD8")

        plt.plot(my_x_ticks, TA_GCN_linear[:], 'peru', label="TARGCN-linear")
        plt.plot(my_x_ticks, TA_GCN_linear[:], 'o',color='peru')

        plt.plot(my_x_ticks, TA_GCN_SA[:], 'green', label="TARGCN-SA")
        plt.plot(my_x_ticks, TA_GCN_SA[:], 'go')

        plt.plot(my_x_ticks, TA_GCN_noTA[:], 'b', label="TARGCN-noTA")
        plt.plot(my_x_ticks, TA_GCN_noTA[:], 'bo')

        plt.plot(my_x_ticks, TA_GCN[:], 'red', label="TARGCN")
        plt.plot(my_x_ticks, TA_GCN[:], 'o', color='red')


        plt.legend()  # 显示图例

        plt.show()

    def MAPE(self):
        # TA_GCN_linear = [11.23,11.23,11.28,11.41,11.56,11.77,12.14,12.67,12.99,13.44,13.76,14.37]
        # TA_GCN_SA = [9.49,9.82,10.13,10.40,10.62,10.84,11.09,11.32,11.55,11.79,12.14,12.65]
        # TA_GCN_noTA = [9.12,9.41,9.56,9.77,9.98,10.12,10.34,10.56,10.72,10.83,11.02,11.30]
        # TA_GCN = [8.73,8.93,9.15,9.32,9.51,9.68,9.91,10.12,10.27,10.50,10.71,11.01]
        TA_GCN_linear = [9.35,9.85,10.32,10.72,11.08,11.44,11.68,12.03,12.31,12.52,12.85,13.33]
        TA_GCN_SA = [8.80,9.20,9.54,9.88,10.19,10.56,10.97,11.38,11.75,11.99,12.37,12.87]
        # TA_GCN_noTA = [8.85,9.04,9.28,9.48,9.66,9.95,10.36,10.45,10.65,10.82,10.93,11.22] # static
        TA_GCN_noTA = [9.25,9.41,9.61,9.80,9.94,10.15,10.25,10.42,10.58,10.62,10.79,10.98] # no static
        TA_GCN = [9.10,9.31,9.43,9.62,9.77,9.91,10.08,10.18,10.27,10.38,10.45,10.65]
        linear=0
        sa=0
        nota=0
        our=0
        for i in range(12):
            linear=linear+TA_GCN_linear[i]
            sa=sa+TA_GCN_SA[i]
            nota=nota+TA_GCN_noTA[i]
            our=our+TA_GCN[i]
        print("MAPE:")
        print("linear:",linear/12,"sa:",sa/12,"nota:",nota/12,"our:",our/12)

        my_x_ticks = np.arange(5, 65, 5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 MAPE")
        plt.xlabel("Prediction interval")
        plt.ylabel("MAPE for PEMSD8")

        plt.plot(my_x_ticks, TA_GCN_linear[:], 'peru', label="TARGCN-linear")
        plt.plot(my_x_ticks, TA_GCN_linear[:], 'o',color='peru')

        plt.plot(my_x_ticks, TA_GCN_SA[:], 'green', label="TARGCN-SA")
        plt.plot(my_x_ticks, TA_GCN_SA[:], 'go')

        plt.plot(my_x_ticks, TA_GCN_noTA[:], 'b', label="TARGCN-noTA")
        plt.plot(my_x_ticks, TA_GCN_noTA[:], 'bo')

        plt.plot(my_x_ticks, TA_GCN[:], 'red', label="TARGCN")
        plt.plot(my_x_ticks, TA_GCN[:], 'o', color='red')


        plt.legend()  # 显示图例
        # plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD8\3.1\Pred_True_{}_day{}_15min.png".format(str(node), str(day)))
        plt.show()


if __name__=='__main__':
    PeMSD8=PeMSD8()
    PeMSD4=PeMSD4()
    PeMSD7 = PeMSD7()
    ablation_result=ablation_result()
    # PeMSD8.MAE()
    # PeMSD8.RMSE()
    # PeMSD8.MAPE()
    # PeMSD4.MAE()
    # PeMSD4.RMSE()
    # PeMSD4.MAPE()
    # PeMSD7.MAE()
    # PeMSD7.RMSE()
    # PeMSD7.MAPE()
    ablation_result.MAE()
    ablation_result.RMSE()
    ablation_result.MAPE()

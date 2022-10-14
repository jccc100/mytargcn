# _*_ coding: utf-8 _*_
# @time     :2022/3/1 13:51
# @Author   :jc
# @File     :MAE-RMSE-MAPE.py
# PEMS04
import matplotlib.pyplot as plt


mae_PEMS04=[18.09,18.26,18.55,18.81,19.01,19.20,19.41,19.62,19.83,19.99,20.19,20.59]
rmse_PEMS04=[29.66,30.18,30.79,31.30,31.70,32.04,32.43,32.81,33.21,33.55,33.91,34.42]
mape_PEMS04=[12.0426,12.1142,12.3065,12.4865,12.5988,12.6812,12.8898,13.0155,13.1481,13.2297,13.3745,13.6024]
# mae
ax1=plt.figure()
plt.plot(range(1,13),mae_PEMS04,'ro',label="RA_GCN")
plt.plot(range(1,13),mae_PEMS04,"b")

# plt.ylabel("time steps")
plt.xlim([0,13])
plt.xlabel("Predict Interval")
plt.title("PEMS04 MAE")
plt.legend()
# plt.show()
# rmse
ax2=plt.figure()
plt.plot(range(1,13),rmse_PEMS04,'ro',label="RA_GCN")
plt.plot(range(1,13),rmse_PEMS04,"b")

# plt.ylabel("time steps")
plt.xlim([0,13])
plt.xlabel("Predict Interval")
plt.title("PEMS04 RMSE")
plt.legend()
# plt.show()
# mape
ax3=plt.figure()
plt.plot(range(1,13),mape_PEMS04,'ro',label="RA_GCN")
plt.plot(range(1,13),mape_PEMS04,"b")

# plt.ylabel("time steps")
plt.xlim([0,13])
plt.xlabel("Predict Interval")
plt.title("PEMS04 MAPE")
plt.legend()
plt.show()


# PEMS08
mae_PEMS08=[]
rmse_PEMS08=[]
mape_PEMS08=[]
# mae

# rmse

# mape


# PEMS03
mae_PEMS03=[]
rmse_PEMS03=[]
mape_PEMS03=[]
# mae

# rmse

# mape






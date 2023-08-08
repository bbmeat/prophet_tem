import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import warnings
import json
from prophet.serialize import model_to_json, model_from_json
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.ERROR)


mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# load model
with open('temp_max_pre.json', 'r') as fin:
    m = model_from_json(json.load(fin))

dataframe = pd.read_csv('./data/min_p.csv')
dataframe['ds'] = pd.to_datetime(dataframe['ds'], format='%Y/%m/%d')

train_size = int(len(dataframe)-1)
print(train_size)
x_de, y_de = pd.DataFrame(dataframe.iloc[:train_size, [0, 3]]), pd.DataFrame(dataframe.iloc[:train_size, 1])

forecast = m.predict(x_de)
print(forecast)
fig, ax = plt.subplots()
# 可视化预测值
forecast.plot(x="ds", y="yhat", style="b-", figsize=(14, 7),
                  label="预测值", ax=ax)
# 可视化出置信区间
ax.fill_between(forecast["ds"].values, forecast["yhat_lower"],
                forecast["yhat_upper"], color='b', alpha=.2,
                label="95%置信区间")

plt.legend(loc=2)
plt.grid()
plt.title("时间序列异常值检测结果")
plt.savefig('temp_min_err.png')
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv('./weather_further_min.csv', index=False)

if __name__ == "__main__":
    main()
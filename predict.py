import prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import datetime
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl
from prophet.plot import add_changepoints_to_plot
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
import json
from prophet.serialize import model_to_json, model_from_json

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

data = pd.read_csv('./data/sensor_min_p.csv')
# data.columns = ['ds', 'y']
data['ds'] = pd.to_datetime(data['ds'], format='%Y/%m/%d')

# data.set_index('ds', inplace=True)

# df_final = data.set_index('ds')

# fig = px.line(data, x="ds", y="y")
# fig.show()


train_size = int(len(data) - 31)
train = data.iloc[:train_size, :]
x_train, y_train = pd.DataFrame(data.iloc[:train_size, [0, 3]]), pd.DataFrame(data.iloc[:train_size, 1])
x_val, y_val = pd.DataFrame(data.iloc[train_size:, [0, 3]]), pd.DataFrame(data.iloc[train_size:, 1])
x_de, y_de = pd.DataFrame(data.iloc[:int(len(data)), [0, 3]]), pd.DataFrame(data.iloc[:int(len(data)), 1])

model = prophet.Prophet(
    changepoint_range=0.8,
    changepoint_prior_scale=0.5,
    daily_seasonality=False,
    yearly_seasonality=True,
    seasonality_prior_scale=10,
    # growth="liner",
    interval_width=0.95,
)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# model.add_country_holidays(country_name='CN')
# model.add_regressor('temp_max')
model.add_regressor('temp_min')

model.fit(train)

future = model.make_future_dataframe(periods=30, freq='D')


# future['cap'] = data.total_purchase_amt.values.max()
# future['floor'] = data.total_purchase_amt.values.min()
def outlier_detection(forecst):
    index = np.where((forecst["y"] <= forecst["yhat_lower"]) | (forecst["y"] >= forecst["yhat_upper"]), True, False)

    return index


forecast = model.predict(x_val)
err_forecast = model.predict(x_de)
err_forecast["y"] = data["y"].reset_index(drop=True)
err_forecast[["ds", "y", "yhat", "yhat_lower", "yhat_upper"]].tail()
outlier_index = outlier_detection(err_forecast)
outlier_df = data[outlier_index]
print("异常值：", np.sum(outlier_index))

fig, ax = plt.subplots()
# 可视化预测值
err_forecast.plot(x="ds", y="yhat", style="b-", figsize=(14, 7),
                  label="预测值", ax=ax)
# 可视化出置信区间
ax.fill_between(err_forecast["ds"].values, err_forecast["yhat_lower"],
                err_forecast["yhat_upper"], color='b', alpha=.2,
                label="95%置信区间")
err_forecast.plot(kind="scatter", x="ds", y="y", c="k",
                  s=20, label="原始数据", ax=ax)
# 可视化出异常值的点
outlier_df.plot(x="ds", y="y", style="rs", ax=ax,
                label="异常值")
plt.legend(loc=2)
plt.grid()
plt.title("时间序列异常值检测结果")
plt.savefig('temp_pre_min_err.png')
# plt.show()

print(forecast)

figa = model.plot_components(forecast)
fig1 = model.plot(forecast)
figa.savefig('temp_min_all.png')
fig1.savefig('temp_min_model.png')
# fig.show()
# fig1.show()
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv('./weather_further_min.csv', index=False)

# Save model
with open('temp_max_pre.json', 'w') as fout:
    json.dump(model_to_json(model), fout)

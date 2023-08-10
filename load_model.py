import pandas as pd
import numpy as np
from prophet import Prophet
import datetime
import re
import os
import logging
import warnings
import json
from prophet.serialize import model_to_json, model_from_json
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)


def change_arr(arr):
    for i in reversed(arr): pass
    return arr


# get yesterday weather link is weak
def get_yes_weather():
    response = requests.get('http://t.weather.itboy.net/api/weather/city/101030100')
    pattern = r'\d+'
    data = json.loads(response.text)
    # print(data)
    temp_data = data["data"]
    yes_data = temp_data['yesterday']
    yes_temp_max = yes_data['high']
    yes_temp_min = yes_data['low']

    temp_max = re.findall(pattern, yes_temp_max)[0]
    temp_min = re.findall(pattern, yes_temp_min)[0]
    return temp_max, temp_min


# get sensor max and min data
def get_sensor_data():
    now = datetime.datetime.now()
    yest = now + datetime.timedelta(days=-1)
    name = 'C364C4251B5A3625'
    startTime = yest.strftime('%Y-%m-%d')
    endTime = yest.strftime('%Y-%m-%d')
    yestTime = yest.strftime('%Y/%m/%d')
    pageNum = 1
    pageSize = 500
    response = requests.get(
        f'http://192.168.30.80:8082/api/storage/data/sac/history?name={name}&startTime={startTime}&endTime={endTime}&pageNum={pageNum}&pageSize={pageSize}'
    )
    data = json.loads(response.text)
    sensor_data_all = data["data"][0]
    sensor_data_list = sensor_data_all["sdlist"]
    sensor_temp = []

    for i in range(len(sensor_data_list)):
        temp = sensor_data_list[i]["data"][1]
        sensor_temp.append(temp)

    sensor_temp_max = max(sensor_temp)
    sensor_temp_min = min(sensor_temp)
    print("max", sensor_temp_max, "min:", sensor_temp_min)
    return yestTime, sensor_temp_max, sensor_temp_min


# create data to fit new model
def create_fit_data():
    # get yesterday weather
    temp = get_yes_weather()
    sensor_data = get_sensor_data()
    # create fit data
    temp_max_f = pd.DataFrame({
        'ds': sensor_data[0],
        'y': sensor_data[1],
        'temp_max': temp[0],
    })

    temp_min_f = pd.DataFrame({
        'ds': sensor_data[0],
        'y': sensor_data[2],
        'temp_min': temp[1],
    })
    return temp_max_f, temp_min_f


# get future weather in 7days
def search_weather():
    location = '101010100'
    key = 'd396c8b70d804c40a0f7a0a5593f017c'
    lan = 'zh'
    response = requests.get(f'https://devapi.qweather.com/v7/weather/7d?location={location}&key={key}&lang={lan}')
    # print(response)
    data = json.loads(response.text)
    # print(data["daily"])
    weather_data = data["daily"]
    date = []
    temp_max = []
    temp_min = []
    for i in range(len(weather_data)):
        date_weather = weather_data[i]["fxDate"]
        date_temp_max = weather_data[i]["tempMax"]
        date_temp_min = weather_data[i]["tempMin"]
        d = date_weather.replace("-", "/")
        date.append(d)
        temp_max.append(date_temp_max)
        temp_min.append(date_temp_min)

    date = change_arr(date).replace('-', '/')
    temp_max = change_arr(temp_max)
    temp_min = change_arr(temp_min)
    print(len(date))
    print(date)
    temp_data = pd.DataFrame({
        'ds': date,
        'temp_max': temp_max,
        'temp_min': temp_min,
    })
    print(temp_data)


# load model
def load_model(model_file):
    with open(model_file, 'r') as fin:
        m = model_from_json(json.load(fin))
    return m


def save_new_model(model, model_file):
    with open(model_file, 'w') as fout:
        json.dump(model_to_json(model), fout)

def plt_set():
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['font.serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# judge model type
def is_in(file_name, key_word):
    # Pair keywords using regular matching
    if re.findall(key_word, file_name):
        return True
    else:
        return False


def get_data(file_path):
    # load data
    dataframe = pd.read_csv(file_path)
    dataframe['ds'] = pd.to_datetime(dataframe['ds'], format='%Y/%m/%d')

    # data size
    train_size = int(len(dataframe) - 1)
    print(train_size)
    train_size_t = int(len(dataframe)) - train_size
    x_de, y_de = pd.DataFrame(dataframe.iloc[train_size_t:, [0, 2]]), pd.DataFrame(dataframe.iloc[train_size_t:, 1])
    return x_de


def predict(model, data, file_path):
    # predict data
    forecast = model.predict(data)
    print("predict data:\n")
    print(forecast)

    # judge data type
    file_name = os.path.basename(file_path).split('.')[0]
    if is_in(file_name, "min"):
        predict_file_name = "temp_min_"
    else:
        predict_file_name = "temp_max_"

    fig, ax = plt.subplots()
    # 可视化预测值
    forecast.plot(x="ds", y="yhat", style="b-", figsize=(14, 7), label="预测值", ax=ax)
    # 可视化出置信区间
    ax.fill_between(forecast["ds"].values, forecast["yhat_lower"],
                    forecast["yhat_upper"], color='b', alpha=.2,
                    label="95%置信区间")

    plt.legend(loc=2)
    plt.grid()
    plt.title("时间序列异常值检测结果")
    plt.savefig(fr'{predict_file_name}err.png')
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(fr'.{predict_file_name}data.csv', index=False)


def model_init(model):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = model.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = model.params[pname][0]

    return res


def main():
    # plt setting
    plt_set()

    min_data_file = r'./data/min_p1-7.csv'
    max_model_file = r'temp_max_pre.json'
    min_model_file = r'temp_min_pre.json'
    max_model = load_model(max_model_file)
    min_model = load_model(min_model_file)
    data = get_data(min_data_file)
    # predict(max_model, data, max_model_file)
    predict(min_model, data, min_model_file)
    temp_data = create_fit_data()
    # fit new model
    # max temp model
    max_model_n = Prophet().fit(temp_data[0], init=model_init(max_model))
    min_model_n = Prophet().fit(temp_data[1], init=model_init(min_model))

    # save model
    model_time = datetime.datetime.now().strftime('%Y%m%d')
    max_model_file_path = f'./model/{model_time}_max.json'
    min_model_file_path = f'./model/{model_time}_min.json'


if __name__ == "__main__":
    main()

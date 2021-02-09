import numpy as np
import time
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU


n_steps = 10 # input step
n_features = 6 # num of features
n_predict_fe = 2 # predict Temperature
pre_len = 1 # predict step

def plotfig(nu,X,Y,file_name,label1,label2,xlabel,ylabel):
	pyplot.figure(num=nu, figsize=(6, 4))  # 创建画图，序号为1，图片大小为2.7*1.8
	pyplot.rcParams['axes.unicode_minus'] = False  # 使用上标小标小一字号
	# pyplot.rcParams['font.sans-serif'] = ['Times New Roman']
	pyplot.rcParams['font.sans-serif'] = ['Arial']
	# 设置全局字体，可选择需要的字体替换掉‘Times New Roman’
	# 使用黑体作为全局字体，可以显示中文
	# plt.rcParams['font.sans-serif']=['SimHei']
	font1 = {'family': 'Arial', 'weight': 'light', 'size': 12}  # 设置字体模板，
	font2 = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}  # 设置字体模板，
	# wight为字体的粗细，可选 ‘normal\bold\light’等
	# size为字体大小
	# pyplot.title("Title", fontdict=font2)  # 标题
	if nu==1 or nu==2:
		pyplot.plot(X, lw=1.5, color='#E96363', marker='.', label=label1)
		pyplot.plot(Y, lw=1.5, color='#0F7FFE', marker='*', label=label2)
	elif nu==3:
		pyplot.plot(X, lw=1.5, color='#E96363', marker='.', label=label1)
	pyplot.xlabel(xlabel, fontdict=font1)
	pyplot.ylabel(ylabel, fontdict=font1)
	pyplot.grid(linestyle='-.')
	# pyplot.legend(loc="best", scatterpoints=1, prop=font1, shadow=True，frameon=True)  # 添加图例，\
	pyplot.legend(loc="best", scatterpoints=1, prop=font1)  # 添加图例，\
	# loc控制图例位置，“best”为最佳位置，“bottom”,"top"，“topringt"等，\
	# shadow为图例边框阴影，frameon控制是否有边框
	pyplot.minorticks_on()  # 开启小坐标
	# 设置图框与图片边缘的距离
	# 设置x轴
	pyplot.tick_params( \
		axis='x',  # 设置x轴
		direction='in',  # 小坐标方向，in、out
		which='both',  # 主标尺和小标尺一起显示，major、minor、both
		bottom=True,  # 底部标尺打开
		top=False,  # 上部标尺关闭
		labelbottom=True,  # x轴标签打开
		labelsize=12)  # x轴标签大小
	pyplot.tick_params( \
		axis='y',
		direction='in',
		which='both',
		left=True,
		right=False,
		labelbottom=True,
		labelsize=12)
	pyplot.ticklabel_format(axis='both', style='sci')  # sci文章的风格
	pyplot.tight_layout(rect=(0, 0, 1, 1))  # rect=[left,bottom,right,top]
	pyplot.savefig(file_name+".pdf", format="pdf", dpi=300)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
interval = '600s'
names=["date","time","epoch","moteid","temperature","humidity","light","voltage"]
dataset = read_csv("/Users/usr/Downloads/data.txt",header=None,delim_whitespace=True,names=names,parse_dates=[[0,1]],index_col="date_time")
pre_data = dataset[dataset.moteid == 1 & (dataset.temperature < 28)]
# pre_data = dataset[dataset.moteid == 1 & (dataset.temperature < 28) & (dataset.temperature > 10)]
pre_data = pre_data.resample(interval).last()
pre_data = pre_data.fillna(method='ffill')
values = pre_data.values



values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours

# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps, pre_len)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_steps = 3000
train = values[:n_train_steps, :]
test = values[n_train_steps:, :]
# split into input and outputs
n_obs = n_steps * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features+n_predict_fe]
test_X, test_y = test[:, :n_obs], test[:, -n_features+n_predict_fe]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(pre_len))
model.compile(loss='mae', optimizer='adam')

start = time.clock()
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=False)
end = time.clock()
print(model.summary())
print("Model fitting time:",str(end-start))
# plot history
plotfig(1,history.history['loss'],history.history['val_loss'],'mae_loss','train','Test','Epochs','MAE Loss')
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_steps*n_features))
# invert scaling for forecast
inv_yhat = concatenate((test_X[:, -n_features:-n_features+n_predict_fe],yhat), axis=1)
if -n_features+n_predict_fe+1 != 0:
	inv_yhat = concatenate((inv_yhat, test_X[:, -n_features+n_predict_fe+1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,n_predict_fe]
# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, -n_features:-n_features+n_predict_fe],test_y), axis=1)
if -n_features+n_predict_fe+1 != 0:
	inv_y = concatenate((inv_y, test_X[:, -n_features+n_predict_fe+1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,n_predict_fe]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

mp = mape(inv_y, inv_yhat)
print('Test MAPE: %.3f' % mp)

plotfig(2,inv_yhat,inv_y,'compare','Predict Value','Ture Value','Step','Temperature °C')
pyplot.show()

plotfig(3,np.abs((inv_y - inv_yhat)),None,'Error','Absolute Error',None,'Step','Error')
pyplot.show()
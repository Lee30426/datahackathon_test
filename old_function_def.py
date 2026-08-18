#function defining

import os
import sys
import numpy as np
import pandas as pd
# we will only import certain module from those libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import math
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from random import randrange
import argparse
import collections
import tensorflow as tf
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Add, Layer
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import Adam
import pydot
import pydotplus
from pydotplus import graphviz
from scipy import stats
from scipy import integrate
from scipy.optimize import fmin
from pyts.metrics import dtw
from pandas.plotting import register_matplotlib_converters

# array_attributes = the names of the columns we would like the histograms for
# cols_number = indicate how many columns for the grid
def showHistograms(data, histogram_attributes, cols_number, height, width):

 rows_number = math.ceil(len(histogram_attributes) / cols_number)
 # rows number calculated automatically
 fig = make_subplots(rows = rows_number, cols = cols_number,
        subplot_titles = (histogram_attributes))

 for i in range(len(histogram_attributes)):
    legend = histogram_attributes[i] + " count"
 # cols and rows start in 1, not in 0, thus we add one.
    row_number = i % cols_number + 1
    col_number = math.ceil((i+1) / cols_number)
    fig.append_trace(go.Histogram(name = legend, x = data[histogram_attributes[i]]),
        row_number, col_number)
 fig.update_layout(height = height, width = width,
    title_text = "Histograms of Selected Attributes")
 fig.show()

 # 히트맵 그리는 함수. 상관계수 수치가 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 나타냄.

# Procedure for Drawing Correlation Heatmap with Coefficient Text Printed
# Input parameter: the dataframe name, the width for heatmap graph, and the number of column
# to visualize.
# Output: correaltion heatmap of the input dataframe with size (width X width)
def plotCorrelationMatrixText(df, graphSize, n):
    df = df[[col for col in df if df[col].nunique() >1]] # keep columns where there are
 # more than 1 unique values
    df = df.iloc[:, :n+1] #get the first n columns
    if df.shape[1] <2: #check if there are more than one column in dataset.
 #If only one column in a dataset, print error message.
        print(f'No correlation plots shown: ')
        print(f'The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr() #calculate pearson correlation between all attributes in dataset

    heatmap = go.Heatmap(
    z=corr,
    x=corr.columns,
    y=corr.columns,
    text=corr,
    texttemplate="%{text:.2f}",
    textfont={"size":7}

    )

    layout = go.Layout(
        title_text ="Correlation Matrix",
        width=graphSize,
        height=graphSize,
    )
    fig = go.Figure(data=[heatmap], layout=layout)
    fig.show()

    # 직접 짤 수 있어야함. 추후 공부해보자.

# 이 산점도 행렬은 상관계수 수치만으로는 알 수 없는 비선형적 관계를 확인하는 방법임.
# 완벽한 직선은 오히려 같은 수치를 다른 도메인으로 관찰한 것으로 판단할 수 있고 일정한 방향성을 갖고 
# Procedure for Drawing Scatter Plot Matrix
# Input parameter: the dataframe name, the number of column to visualize, the size of graph,
# the text size of graph
# Output: Scatter matrix graph with specified size
def plotScatterMatrix(df, n, plotSize):
    df = df.select_dtypes(include = [np.number]) # keep only numerical columns
    df = df.dropna('columns') #drop column with missing data
    df = df[[col for col in df if df[col].nunique() > 1]]
    # keep columns where there are more than 1 unique values
    df = df.iloc[:, :n] #get the first n columns
    columnNames = list(df) #list of all attributes
    df = df[columnNames]
    layout = go.Layout(
        title_text = "Scatter matrix",
        width = plotSize,
        height = plotSize,)
    fig = px.scatter_matrix(df)
    fig.update_layout(
        title = '<span style="font-size: 16px;">Scatter and Density Plot</span>',
        width = plotSize, height=plotSize,
        font = dict(size = 8))
    fig.show()

def removeConstant(df, n):
 df = df[[col for col in df if df[col].nunique() > n]]
 return df

#칼럼 제거 후 데이터 이름 확인
#수치가 변하지 않고 하나인 칼럼을 제거함. 

def identify_outliers(df, c):
    #set the constant for IQR boundary
    constant = float(c)
    # calculate Q1 and Q3
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    # calculate the IQR
    IQR = Q3 - Q1
    # filter the dataset with the IQR
    IQR_outliers = df[((df.lt(Q1 - constant * IQR)) | (df.gt(Q3 + constant * IQR))).any(axis=1)]
    IQR_outliers = pd.DataFrame(IQR_outliers)
    return IQR_outliers

def remove_outliers(df, c):
    #find outliers
    df = pd.DataFrame(df)
    outliers = identify_outliers(df, c)
    #remove outliers
    df_out = pd.DataFrame(outliers)
    df.drop(df_out.index, inplace = True)
    return df

def handleMissingValue(data):
    df = data.copy()
    numeric_df = df.columns[df.dtypes != 'object']
    categorical_df = df.columns[df.dtypes == 'object']
    #handling missing value for categorical
    for i in categorical_df:
        df[i].fillna(df[i].mode()[0], inplace = True)

    df_flag_null = df.isnull()
    i, c = np.where(df_flag_null)
    #handling missing value for numerical
    for j in range(len(i)):
    #if missing value in the beginning
        if i[j] == 0:
            s = df.iloc[:, c[j]]
            id_s = s.notna().idxmax()
            val = df.iat[id_s, c[j]]
            df.iat[i[j], c[j]] = val
            #df.iloc[[i[j]], [c[j]]] = df.iloc[[id_s], [c[j]]]
            #print(type(val))
        #if missing value in the end
        elif i[j] == (len(df) - 1):
            s = df.iloc[:, c[j]]
            id_s = s.notna()[::-1].idxmax()
            val = df.iat[id_s, c[j]]
            df.iat[i[j], c[j]] = val
        #if missing value in the middle
        else:
            low = df.iat[i[j] - 1, c[j]]
            high = df.iat[i[j] + 1, c[j]]
            print(high)
            if(math.isnan(high)):
                val = low
            else:
                val = (low+high) / 2
                df.iat[i[j], c[j]] = val

    return df

def diff_smooth_df(df, lags_n, diffs_n, smooth_n, diffs_abs=False, abs_features=False):
 #Given a pandas dataframe preprocess it to take differences, add smoothing, and lags as specified.
    if diffs_n >=1:
    # take differences
        df = df.diff(diffs_n).dropna()
    # abs diffs if defined
        if diffs_abs ==True:
            df = abs(df)
    if smooth_n >=2:
    # apply a rolling average to smooth out the data a bit
        df = df.rolling(smooth_n).mean().dropna()
    if lags_n >=1:
 # for each dimension add a new columns for each of lags_n lags of the differenced and smoothed values for that dimension
        df_columns_new = [f'{col}_lag{n}'for n in range(lags_n+1) for col in df.columns]
        df = pd.concat([df.shift(n) for n in range(lags_n+1)], axis =1).dropna()
        df.columns = df_columns_new
 # sort columns to have lagged values next to each other for clarity when looking at the feature vectors
    df = df.reindex(sorted(df.columns), axis =1)
 # abs all features if specified
    if abs_features ==True:
        df = abs(df)
    return df

def time_segments_aggregate(X, interval, time_column, method=['mean']):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    X = X.sort_values(time_column).set_index(time_column)
    if isinstance(method, str):
        method = [method]
    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]
    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts-1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts
    return np.asarray(values), np.asarray(index)

def rolling_window_sequences(X, index, window_size, target_size, step_size,
 target_column, drop=None, drop_windows=False):
 """Create rolling window sequences out of time series data.
 The function creates an array of input sequences and an array of target sequences
 by rolling over the input sequence with a specified window. Optionally, certain
 values can be dropped from the sequences.
 Args:
 X (ndarray): N-dimensional sequence to iterate over.
 index (ndarray): Array containing the index values of X.
 window_size (int): Length of the input sequences.
 target_size (int): Length of the target sequences.
 step_size (int):
 Indicating the number of steps to move the window forward each round.
 target_column (int): Indicating which column of X is the target.
 drop (ndarray or None or str or float or bool):
 Optional. Array of boolean values indicating which values of X are invalid,
 or value indicating which value should be dropped. If not given,
 `None` is used.
 drop_windows (bool): Optional. Indicates whether the dropping
 functionality should be enabled. If not given, `False` is used.
 Returns:
 ndarray, ndarray, ndarray, ndarray:
 * input sequences.
 * target sequences.
 * first index value of each input sequence.
 * first index value of each target sequence.
 """
 out_X = list()
 out_y = list()
 X_index = list()
 y_index = list()
 target = X[:, target_column]
 if drop_windows:
    if hasattr(drop, '__len__') and (not isinstance(drop, str)):
        if len(drop) != len(X):
            raise Exception('Arrays `drop` and `X` must be of the same length.')
    else:
        if isinstance(drop, float) and np.isnan(drop):
            drop = np.isnan(X)
        else:
            drop = X == drop
 start =0
 max_start = len(X) - window_size - target_size + 1
 while start < max_start:
    end = start + window_size
    if drop_windows:
        drop_window = drop[start:end+target_size]
        to_drop = np.where(drop_window)[0]
        if to_drop.size:
            start += to_drop[-1] + 1
            continue
    out_X.append(X[start:end])
    out_y.append(target[end:end+target_size])
    X_index.append(index[start])
    y_index.append(index[end])
    start = start + step_size
 return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


#MTadGAN에 필요한 함수 중 하나로서 RandomWeightedAverage() 클래스를 정의하는데 이는 입력 데이터와 예측된 입력 데이터를 정해진 비율로 선형 결합하는 역할을 한다.
class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
 
        super().__init__()
        self.batch_size = batch_size
    def call(self, inputs, **kwargs):
        alpha = K.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1-alpha) * inputs[1])
print("RandomWeightedAverage() Class defined, Done! ")

def build_encoder_layer(input_shape, encoder_reshape_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
    Bidirectional(LSTM(units=win_size, return_sequences=True)),
    Flatten(),
    Dense(20), # 20 = self.critic_z_input_shape[0]
    Reshape(target_shape=encoder_reshape_shape)]) # (20, 1)
    return Model(x, model(x))

#MTadGAN의 generator 레이어를 생성하는 함수를 정의한다. 이는 숨은 공간으로부터 가상의 데이터를 생성하는 역할을 한다.
def build_generator_layer(input_shape, generator_reshape_shape):
 # input_shape = (20, 1) / generator_reshape_shape = (50, 1)
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Flatten(),
        Dense(win_size), # 50 originally
        Reshape(target_shape=generator_reshape_shape), # (50, 1)
        Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
        Dropout(rate=0.2),
        #UpSampling1D(size=2),
        UpSampling1D(size=1),
        Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
        Dropout(rate=0.2),
        TimeDistributed(Dense(features_dim)), # features_dim >=1, multiple features
        Activation(activation='tanh')]) # (None, 10, 1)
    return Model(x, model(x))
print("build generator layer defined, Done!")

def build_critic_x_layer(input_shape):
 x = Input(shape=input_shape)
 model = tf.keras.models.Sequential([
 Conv1D(filters=64, kernel_size=k_size),
 LeakyReLU(alpha=0.2),
 Dropout(rate=0.25),
 Conv1D(filters=64, kernel_size=k_size),
 LeakyReLU(alpha=0.2),
 Dropout(rate=0.25),
 Conv1D(filters=64, kernel_size=k_size),
 LeakyReLU(alpha=0.2),
 Dropout(rate=0.25),
 Conv1D(filters=64, kernel_size=k_size),
 LeakyReLU(alpha=0.2),
 Dropout(rate=0.25),
 Flatten(),
 Dense(units=1)])
 return Model(x, model(x))

def build_critic_z_layer(input_shape):
 x = Input(shape=input_shape)
 model = tf.keras.models.Sequential([
    Flatten(),
    Dense(units=100),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.2),
    Dense(units=100),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.2),
    Dense(units=1)])
 return Model(x, model(x))

#MTadGAN의 Wasserstein 로스 함수를 정의한다. Wasserstein 로스 함수는 두 확률 분포 사이의 거리를 주는 함수로서 판별자 네트워크의 학습에 쓰이는 로스함수이다.
def wasserstein_loss(y_true, y_pred):
 return K.mean(y_true*y_pred)


#critic_x 네트워크를 학습하는 과정을 수행하는 함수를 정의한다. 로스함수를 정의하고 gradient 계산함수, 그리고 Adam 최적화 함수를 포함한다.
def critic_x_train_on_batch(x, z, valid, fake, delta):
 with tf.GradientTape() as tape:
    (valid_x, fake_x, interpolated) = critic_x_model(inputs=[x, z], training=True)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic_x(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)[0]
    grads = tf.square(grads)
    ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
    gp_loss = tf.reduce_mean((ddx-1.0) **2)
    loss = tf.reduce_mean(wasserstein_loss(valid, valid_x))
    loss += tf.reduce_mean(wasserstein_loss(fake, fake_x))
    loss += gp_loss*10.0
 gradients = tape.gradient(loss, critic_x_model.trainable_weights)
 optimizer.apply_gradients(zip(gradients, critic_x_model.trainable_weights))
 return loss    


# critic_z 네트워크를 학습하는 과정을 수행하는 함수를 정의한다. 로스함수를 정의하고 gradient 계산함수, 그리고 Adam 최적화 함수를 포함한다
def critic_z_train_on_batch(x, z, valid, fake, delta):
 with tf.GradientTape() as tape:
    (valid_z, fake_z, interpolated) = critic_z_model(inputs=[x, z], training=True)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic_z(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)[0]
    grads = tf.square(grads)
    ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
    gp_loss = tf.reduce_mean((ddx -1.0) **2)
    loss = tf.reduce_mean(wasserstein_loss(valid, valid_z))
    loss += tf.reduce_mean(wasserstein_loss(fake, fake_z))
    loss += gp_loss*10.0
 gradients = tape.gradient(loss, critic_z_model.trainable_weights)
 optimizer.apply_gradients(zip(gradients, critic_z_model.trainable_weights))
 return loss

#encoder-generator 네트워크를 학습하는 과정을 수행하는 함수를 정의한다. 로스함수를 정의하고 gradient 함수, 그리고 Adam 최적화 함수를 포함한다.
def enc_gen_train_on_batch(x, z, valid):
 with tf.GradientTape() as tape:
    (fake_gen_x, fake_gen_z, x_gen_rec) = encoder_generator_model(inputs=[x, z], training=True)
    x = tf.squeeze(x)
    x_gen_rec = tf.squeeze(x_gen_rec)
    loss = tf.reduce_mean(wasserstein_loss(valid, fake_gen_x))
    loss += tf.reduce_mean(wasserstein_loss(valid, fake_gen_z))
    loss += tf.keras.losses.MSE(x, x_gen_rec)*10
    loss = tf.reduce_mean(loss)
 gradients = tape.gradient(loss, encoder_generator_model.trainable_weights)
 optimizer.apply_gradients(zip(gradients, encoder_generator_model.trainable_weights))
 return loss

def predict(X):
 """Predict values using the initialized object.
 Args:
 X (ndarray): N-dimensional array containing the input sequences for the model.
 Returns:
 ndarray:
 N-dimensional array containing the reconstructions for each input sequence.
 ndarray:
 N-dimensional array containing the critic scores for each input sequence.
 """
 X = X.reshape((-1, shape[0], feat_dim)) # feat_dim : feature dimension (modified from 1: yeesj)
 z_ = encoder.predict(X)
 y_hat = generator.predict(z_)
 critic = critic_x.predict(X)
 return y_hat, critic


# anomalies.py (1)
# """
# Time Series anomaly detection functions.
# Some of the implementation is inspired by the paper
# https://arxiv.org/pdf/1802.04431.pdf
# """
class Anomaly(object):
 def __init__(self):
    pass
 def _deltas(self, errors, epsilon, mean, std):
#  """Compute mean and std deltas.
#  delta_mean = mean(errors) - mean(all errors below epsilon)
#  delta_std = std(errors) - std(all errors below epsilon)
#  Args:
#  errors (ndarray): Array of errors.
#  epsilon (ndarray): Threshold value.
#  mean (float): Mean of errors.
#  std (float): Standard deviation of errors.
#  Returns:
#  float, float:
#  * delta_mean.
#  * delta_std.
#  """
    below = errors[errors <= epsilon]
    if not len(below):
        return 0, 0
    return mean - below.mean(), std - below.std()

 # anomalies.py (2)
 def _count_above(self, errors, epsilon):
#  """Count number of errors and continuous sequences above epsilon.
#  Continuous sequences are counted by shifting and counting the number
#  of positions where there was a change and the original value was true,
#  which means that a sequence started at that position.
#  Args:
#  errors (ndarray): Array of errors.
#  epsilon (ndarray): Threshold value.
#  Returns:
#  int, int:
#  * Number of errors above epsilon.
#  * Number of continuous sequences above epsilon.
#  """
    above = errors > epsilon
    total_above =len(errors[above])
    above = pd.Series(above)
    shift = above.shift(1)
    change = above != shift
    total_consecutive = sum(above & change)
    return total_above, total_consecutive

 # anomalies.py (3)
 def _z_cost(self, z, errors, mean, std):
#  """Compute how bad a z value is.
#  The original formula is::
#  (delta_mean/mean) + (delta_std/std)
#  ------------------------------------------------------
#  number of errors above + (number of sequences above)^2
#  which computes the \"goodness\" of `z`, meaning that the higher the value
#  the better the `z`.
#  In this case, we return this value inverted (we make it negative),
#  to convert it into a cost function, as later on we will use scipy.fmin
#  to minimize it.
#  Args:
#  z (ndarray): Value for which a cost score is calculated.
#  errors (ndarray): Array of errors.
#  mean (float): Mean of errors.
#  std (float): Standard deviation of errors.
#  Returns float: Cost of z.
#  """
    epsilon = mean + z * std
    delta_mean, delta_std =self._deltas(errors, epsilon, mean, std)
    above, consecutive =self._count_above(errors, epsilon)
    numerator =-(delta_mean / mean + delta_std / std)
    denominator = above + consecutive **2
    if denominator ==0:
        return np.inf
    return numerator / denominator
 # anomalies.py (4)
 def _find_threshold(self, errors, z_range):
#  """Find the ideal threshold.
#  The ideal threshold is the one that minimizes the z_cost function.
#  Scipy.fmin is used to find the minimum, using the values from z_range as
#  starting points.
#  Args:
#  errors (ndarray): Array of errors.
#  z_range (list):
#  List of two values denoting the range out of which the start points for
#  the scipy.fmin function are chosen.
#  Returns: float: Calculated threshold value.
#  """
    mean = errors.mean()
    std = errors.std()
    min_z, max_z = z_range
    best_z = min_z
    best_cost = np.inf
    for z in range(min_z, max_z):
        best = fmin(self._z_cost, z, args=(errors, mean, std), full_output=True,
        disp=False)
        z, cost = best[0:2]
        if cost < best_cost:
            best_z = z[0]
    return mean + best_z * std

 def _fixed_threshold(self, errors, k=3.0):
#  """Calculate the threshold.
#  The fixed threshold is defined as k standard deviations away from the mean.
#  Args:
#  errors (ndarray): Array of errors.
#  Returns:
#  float: Calculated threshold value.
#  """
    mean = errors.mean()
    std = errors.std()
    return mean + k * std

 # anomalies.py (5-1)
 def _find_sequences(self, errors, epsilon, anomaly_padding):
#  """Find sequences of values that are above epsilon.
#  This is done following this steps:
#  * create a boolean mask that indicates which values are above epsilon.
#  * mark certain range of errors around True values with a True as well.
#  * shift this mask by one place, filing the empty gap with a False.
#  * compare the shifted mask with the original one to see if there are
#  changes.
#  * Consider a sequence start any point which was true and has changed.
#  * Consider a sequence end any point which was false and has changed.
#  Args:
#  errors (ndarray): Array of errors.
#  epsilon (float): Threshold value. All errors above epsilon are considered an
#  anomaly.
#  anomaly_padding (int):
#  Number of errors before and after a found anomaly that are added to
#  the anomalous sequence.
#  Returns:
#  ndarray, float:
#  * Array containing start, end of each found anomalous sequence.
#  * Maximum error value that was not considered an anomaly.
#  """
    above = pd.Series(errors > epsilon)
    index_above = np.argwhere(above.values)
    for idx in index_above.flatten():
        above[max(0, idx - anomaly_padding):min(idx + anomaly_padding +1,
        len(above))] =True
    shift = above.shift(1).fillna(False)
    change = above != shift
    if above.all():
        max_below =0
    else:
        max_below = max(errors[~above])
    index = above.index
    starts = index[above & change].tolist()
    ends = (index[~above & change]-1).tolist()
    if len(ends) ==len(starts)-1:
        ends.append(len(above)-1)
    return np.array([starts, ends]).T, max_below

 # anomalies.py (5-2)
 def _get_max_errors(self, errors, sequences, max_below):
#  """Get the maximum error for each anomalous sequence.
#  Also add a row with the max error which was not considered anomalous.
#  Table containing a ``max_error`` column with the maximum error of each
#  sequence and the columns ``start`` and ``stop`` with the corresponding start
#  and stop indexes, sorted descendingly by the maximum error.
#  Args:
#  errors (ndarray): Array of errors.
#  sequences (ndarray): Array containing start, end of anomalous sequences
#  max_below (float): Maximum error value that was not considered an anomaly.
#  Returns:
#  pandas.DataFrame: DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
#  """
    max_errors = [{
        'max_error': max_below,
       'start': -1,
        'stop': -1
    }]
    for sequence in sequences:
        start, stop = sequence
        sequence_errors = errors[start: stop +1]
        max_errors.append({
        'start': start,
        'stop': stop,
        'max_error': max(sequence_errors)
        })
    max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)
    return max_errors.reset_index(drop=True)

 # anomalies.py (6)
 def _prune_anomalies(self, max_errors, min_percent):
#  """Prune anomalies to mitigate false positives.
#  This is done by following these steps:
#  * Shift the errors 1 negative step to compare each value with the next one.
#  * Drop the last row, which we do not want to compare.
#  * Calculate the percentage increase for each row.
#  * Find rows which are below ``min_percent``.
#  * Find the index of the latest of such rows.
#  * Get the values of all the sequences above that index.
#  Args:
#  max_errors (pandas.DataFrame): DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
#  min_percent (float):
#  Percentage of separation the anomalies need to meet between themselves and the
#  highest non-anomalous error in the window sequence.
#  Returns:
#  ndarray: Array containing start, end, max_error of the pruned anomalies.
#  """
    next_error = max_errors['max_error'].shift(-1).iloc[:-1]
    max_error = max_errors['max_error'].iloc[:-1]
    increase = (max_error-next_error) / max_error
    too_small = increase < min_percent
    if too_small.all():
        last_index =-1
    else:
        last_index = max_error[~too_small].index[-1]
    return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index+1].values

 # anomalies.py (7)
 def _compute_scores(self, pruned_anomalies, errors, threshold, window_start):
#  """Compute the score of the anomalies.
#  Calculate the score of the anomalies proportional to the maximum error in the sequence
#  and add window_start timestamp to make the index absolute.
#  Args:
#  pruned_anomalies (ndarray): Array of anomalies containing the start, end and max_error for all anomalies in the window.
#  errors (ndarray): Array of errors.
#  threshold (float): Threshold value.
#  window_start (int): Index of the first error value in the window.
#  Returns:
#  list: List of anomalies containing start-index, end-index, score for each anomaly.
#  """
    anomalies = list()
    denominator = errors.mean() + errors.std()
    for row in pruned_anomalies:
        max_error = row[2]
        score = (max_error-threshold) / denominator
        anomalies.append([row[0]+window_start, row[1]+window_start, score])
    return anomalies

 # anomalies.py (8)
 def _merge_sequences(self, sequences):
#     """Merge consecutive and overlapping sequences.
#  We iterate over a list of start, end, score triples and merge together
#  overlapping or consecutive sequences.
#  The score of a merged sequence is the average of the single scores,
#  weighted by the length of the corresponding sequences.
#  Args:
#  sequences (list): List of anomalies, containing start-index, end-index, score for each anomaly.
#  Returns:
#  ndarray: Array containing start-index, end-index, score for each anomaly after merging.
#  """
    if len(sequences) ==0:
        return np.array([])
    sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]
    weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]
    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]
        if sequence[0] <= prev_sequence[1] +1:
            score.append(sequence[2])
            weights.append(sequence[1] - sequence[0])
            weighted_average = np.average(score, weights=weights)
            new_sequences[-1] = (prev_sequence[0],
                max(prev_sequence[1],
                sequence[1]), weighted_average)
        else:
            score = [sequence[2]]
            weights = [sequence[1] - sequence[0]]
            new_sequences.append(sequence)
    return np.array(new_sequences)

 # anomalies.py (9)
 def _find_window_sequences(self, window, z_range, anomaly_padding, min_percent, window_start, fixed_threshold):
    if fixed_threshold: threshold =self._fixed_threshold(window)
    else:
        threshold =self._find_threshold(window, z_range)
    window_sequences, max_below =self._find_sequences(window, threshold,
    anomaly_padding)
    max_errors =self._get_max_errors(window, window_sequences, max_below)
    pruned_anomalies =self._prune_anomalies(max_errors, min_percent)
    window_sequences =self._compute_scores(pruned_anomalies, window, threshold,
        window_start)
    return window_sequences

 # anomalies.py (10-1)
 def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
    window_size_portion=None, window_step_size=None,
    window_step_size_portion=None, min_percent=0.1,
    anomaly_padding=50, lower_threshold=False,
    fixed_threshold=True):

    window_size = window_size or len(errors)
    if window_size_portion:
        window_size = np.ceil(len(errors) * window_size_portion).astype('int')
    window_step_size = window_step_size or window_size
    if window_step_size_portion:
        window_step_size = np.ceil(window_size*window_step_size_portion).astype('int')
    window_start =0
    window_end =0
    sequences = list()
    # anomalies.py (10-3)
    while window_end <len(errors):
        window_end = window_start + window_size
        window = errors[window_start:window_end]
        window_sequences =self._find_window_sequences(window,
            z_range,
            anomaly_padding,
            min_percent,
            window_start,
            fixed_threshold)
        sequences.extend(window_sequences)
        if lower_threshold:
    # Flip errors sequence around mean
            mean = window.mean()
            inverted_window = mean - (window - mean)
            inverted_window_sequences=self._find_window_sequences(inverted_window,
                z_range,
                anomaly_padding,
                min_percent,
                window_start,
                fixed_threshold)
            sequences.extend(inverted_window_sequences)
        window_start = window_start + window_step_size
    sequences =self._merge_sequences(sequences)
    anomalies = list()
    for start, stop, score in sequences:
        print("start", start)
        print("stop", stop)
        print("score", score)
        anomalies.append([index[int(start)], index[int(stop)], score])
    return anomalies

 # anomalies.py (11)
 def _compute_critic_score(self, critics, smooth_window):
#  """Compute an array of anomaly scores.
#  Args:
#  critics (ndarray): Critic values.
#  smooth_window (int): Smooth window that will be applied to compute smooth errors.
#  Returns:
#  ndarray: Array of anomaly scores.
#  """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)
    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) +1
    z_scores = pd.Series(z_scores).rolling(smooth_window, center=True,
        min_periods=smooth_window //2).mean().values
    return z_scores
 
 # anomalies.py (12)
 def _regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
#  """Compute an array of absolute errors comparing predictions and expected output.
#  If smooth is True, apply EWMA to the resulting array of errors.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values.
#  smoothing_window (float):
#  Optional. Size of the smoothing window, expressed as a proportion of the total
#  length of y. If not given, 0.01 is used.
#  smooth (bool):
#  Optional. Indicates whether the returned errors should be smoothed with EWMA.
#  If not given, `True` is used.
#  Returns:
#  ndarray: Array of errors.
#  """
    errors = np.abs(y-y_hat)[:, 0]
    if not smooth:
        return errors
    smoothing_window =int(smoothing_window*len(y))
    return pd.Series(errors).ewm(span=smoothing_window).mean().values
 
 # anomalies.py (13)
 def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
#  """Compute an array of absolute errors comparing predictions and expected output.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values.
#  smoothing_window (float):
#  Optional. Size of the smoothing window, expressed as a proportion of the total
#  length of y. If not given, 0.01 is used.
#  smooth (bool):
#  Optional. Indicates whether the returned errors should be smoothed with EWMA.
#  If not given, `True` is used.
#  Returns:
#  ndarray: Array of errors.
#  """
    errors = np.abs(y - y_hat)[:, 0]
    if not smooth:
        return errors
    smoothing_window =int(smoothing_window *len(y))
    return pd.Series(errors).ewm(span=smoothing_window).mean().values
 
 # anomalies.py (14)
 def _point_wise_error(self, y, y_hat):
#  """Compute point-wise error between predicted and expected values.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values.
#  Returns:
#  ndarray: An array of smoothed point-wise error.
#  """
    y_abs = abs(y-y_hat)
    y_abs_sum = np.sum(y_abs, axis=-1)
    return y_abs_sum
 
 def _area_error(self, y, y_hat, score_window=10):
#  """Compute area error between predicted and expected values.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values.
#  score_window (int):
#  Optional. Size of the window over which the scores are calculated.
#  If not given, 10 is used.
#  Returns:
#  ndarray: An array of area error.
#  """
    smooth_y = pd.Series(y).rolling(score_window,
        center=True,
        min_periods=score_window //2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(score_window,
        center=True,
        min_periods=score_window //2).apply(integrate.trapz)
    errors = abs(smooth_y-smooth_y_hat)
    return errors
 
 # anomalies.py (15)
 def _dtw_error(self, y, y_hat, score_window=10):
#  """Compute dtw error between predicted and expected values.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values.
#  score_window (int):
#  Optional. Size of the window over which the scores are calculated.
#  If not given, 10 is used.
#  Returns:
#  ndarray: An array of dtw error.
#  """
    length_dtw = (score_window //2) *2 +1
    half_length_dtw = length_dtw //2
    # add padding
    y_pad = np.pad(y,
        (half_length_dtw, half_length_dtw),
        'constant',
        constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat,
        (half_length_dtw, half_length_dtw),
        'constant',
        constant_values=(0, 0))
    i =0
    similarity_dtw = list()
    while i <len(y)-length_dtw:
        true_data = y_pad[i:i+length_dtw]
        true_data = true_data.flatten()
        pred_data = y_hat_pad[i:i+length_dtw]
        pred_data = pred_data.flatten()
        dist = dtw(true_data, pred_data)
        similarity_dtw.append(dist)
        i +=1
    errors = ([0] * half_length_dtw + similarity_dtw + [0] * (len(y)-len(similarity_dtw) -
        half_length_dtw))
    return errors
 
    # anomalies.py (16-1)
 def _reconstruction_errors(self, y, y_hat, step_size=1,
 score_window=10, smoothing_window=0.01,
 smooth=True, rec_error_type='point'):
#  """Compute an array of reconstruction errors.
#  Compute the discrepancies between the expected and the
#  predicted values according to the reconstruction error type.
#  Args:
#  y (ndarray): Ground truth
#  y_hat (ndarray): Predicted values. Each timestamp has multiple predictions.
#  step_size (int):
#  Optional. Indicating the number of steps between windows in the predicted values.
#  If not given, 1 is used.
#  score_window (int):
#  Optional. Size of the window over which the scores are calculated.
#  If not given, 10 is used.
#  smoothing_window (float or int):
#  Optional. Size of the smoothing window, when float it is expressed as a proportion
#  of the total length of y. If not given, 0.01 is used.
#  smooth (bool):
#  Optional. Indicates whether the returned errors should be smoothed.
#  If not given, `True` is used.
#  rec_error_type (str):
#  Optional. Reconstruction error types ``["point", "area", "dtw"]``.
#  If not given, "point" is used.
#  Returns:
#  ndarray: Array of reconstruction errors.
#  """
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)
    true=[]
    for i in range(len(y)):
        true.append(y[i][0])
    for it in range(len(y[-1]) -1): # contents of the last window included
        true.append(y[-1][it+1])

 # anomalies.py (16-2)
###################################################################
    predictions = []
    predictions_vs = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] -1)
    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i+1, pred_length)):
            intermediate.append(y_hat[i-j, j])
        ave_p = []
        if intermediate:
            predictions.append(np.average(intermediate, axis=0))
            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])
        true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)
 # Compute reconstruction errors
    if rec_error_type.lower() =="point":
        errors =self._point_wise_error(true, predictions)
    elif rec_error_type.lower() =="area":
        errors =self._area_error(true, predictions, score_window)
    elif rec_error_type.lower() =="dtw":
        errors =self._dtw_error(true, predictions, score_window)
 # Apply smoothing
    if smooth:
            errors = pd.Series(errors).rolling(smoothing_window,
                                               center=True,
                min_periods=smoothing_window //2).mean().values
    return errors, predictions_vs

 # anomalies.py (17-1)
 def score_anomalies(self, y, y_hat, critic, index,
    score_window=10, critic_smooth_window=None,
    error_smooth_window=None, smooth=True,
    rec_error_type="point", comb="mult", lambda_rec=0.5):
#  """Compute an array of anomaly scores.
#  Anomaly scores are calculated using a combination of reconstruction error and critic score.
#  Args:
#  y (ndarray): Ground truth.
#  y_hat (ndarray): Predicted values. Each timestamp has multiple predictions.
#  index (ndarray): time index for each y (start position of the window)
#  critic (ndarray): Critic score. Each timestamp has multiple critic scores.
#  score_window (int):
#  Optional. Size of the window over which the scores are calculated.
#  If not given, 10 is used.
#  critic_smooth_window (int):
#  Optional. Size of window over which smoothing is applied to critic.
#  If not given, 200 is used.
#  error_smooth_window (int):
#  Optional. Size of window over which smoothing is applied to error.
#  If not given, 200 is used.
#  smooth (bool):
#  Optional. Indicates whether errors should be smoothed.
#  If not given, `True` is used.
#  rec_error_type (str):
#  Optional. The method to compute reconstruction error. Can be one of
#  `["point", "area", "dtw"]`. If not given, 'point' is used.
#  comb (str):
#  Optional. How to combine critic and reconstruction error. Can be one
#  of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
#  lambda_rec (float):
#  Optional. Used if `comb="sum"` as a lambda weighted sum to combine
#  scores. If not given, 0.5 is used.
#  Returns:
#  ndarray: Array of anomaly scores.
#  """
 # anomalies.py (17-2)
    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0]*0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0]*0.01)
    step_size =1 # expected to be 1
    true_index = list(index) # no offset
    true = []
    for i in range(len(y)):
        true.append(y[i][0])
    for it in range(len(y[-1]) -1): # contents of the last window included
        true.append(y[-1][it+1])
        true_index.append((index[-1] + it +1))# extend the index (by inluding the last window part)
    true_index = np.array(true_index)# in order to cover the whole sequence of data
    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())
    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))
    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0]-1)

    # anomalies.py (17-3)
    for i in range(num_errors):
        critic_intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i+1, pred_length)):
            critic_intermediate.append(critic_extended[i - j, j])
        if len(critic_intermediate) >1:
            discr_intermediate = np.asarray(critic_intermediate)
            try:
                critic_kde_max.append(discr_intermediate[np.argmax(stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
            except np.linalg.LinAlgError:
                critic_kde_max.append(np.median(discr_intermediate))
        else:
            critic_kde_max.append(np.median(np.asarray(critic_intermediate)))
    # Compute critic scores
    critic_scores =self._compute_critic_score(critic_kde_max, critic_smooth_window)
    # Compute reconstruction scores
    rec_scores, predictions =self._reconstruction_errors(y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)
    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) +1
 # Combine the two scores
    if comb =="mult":
        final_scores = np.multiply(critic_scores, rec_scores)
    elif comb =="sum":
        final_scores = (1 - lambda_rec) * (critic_scores -1) + lambda_rec * (rec_scores -1)
    elif comb =="rec":
        final_scores = rec_scores
    else:
        raise ValueError('Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))
    true = [[t] for t in true]
    return final_scores, true_index, true, predictions


 
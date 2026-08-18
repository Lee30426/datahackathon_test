import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# we will only import certain module from those libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from random import randrange
from datetime import datetime
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.simplefilter(action='ignore', category=FutureWarning)

# the file path is the archive location of the file in our computer
file_path = './data/raw_data/train/1000_chg.csv'
data = pd.read_csv(file_path) # using pandas library (pd) to read the csv file.
# After reading the file, it will be used as a Pandas DataFrame.
# Pandas DataFrame is a special data structure from Pandas Library that is two-dimensional,
# size-mutable, potentially heterogeneous tabular data.
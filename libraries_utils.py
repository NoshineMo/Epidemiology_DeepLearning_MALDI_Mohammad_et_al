import os
import time
import pickle


import numpy as np

import dateutil
import ast
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import itertools
from math import sqrt

from scipy.signal import cwt
from scipy.signal import ricker

pd.options.display.max_colwidth = 1000

from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score, make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from scipy import special
from sklearn.metrics import confusion_matrix
from collections import Counter
import random
import collections
import scipy
from scipy import stats
from scipy.sparse import csr_matrix

# multi-class classification with Keras


import keras
import tensorflow as tf
from keras.models import Sequential
from keras import Model
from keras.layers import Dense
#from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten,  MaxPool1D , Conv1D, LayerNormalization, GlobalMaxPooling1D, AveragePooling1D, SpatialDropout1D, Activation, GRU, BatchNormalization, Bidirectional, Conv2D, Lambda, TimeDistributed, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import Concatenate, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.regularizers import Regularizer
from keras.regularizers import l1, l2
from keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from keras.layers import concatenate

from keras import backend
import tempfile 
from sklearn.model_selection import KFold, StratifiedKFold
#%load_ext tensorboard
from keras import layers
#%matplotlib inline
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

#!pip install keras-tqdm
#from keras_tqdm import TQDMNotebookCallback


import tensorflow_addons as tfa
from tensorflow_addons.layers import ESN


from packaging import version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

import tensorboard
tensorboard.__version__

from tcn import TCN, tcn_full_summary
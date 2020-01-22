import tensorflow as tf

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers
import tensorflow_datasets as tfds

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model

import  matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import os
import time
from glob import glob

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from matplotlib.patches import Rectangle
from skimage.transform import resize
from tensorflow.keras.losses import binary_crossentropy

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DenseFeatures
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.load_model('saved_model/myModel')

def createFeatureColumns(inputFeatures):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in inputFeatures])

testData = [98, 415, -7, -7, 80, 35]
testData = [testData]

print(model.predict(testData))

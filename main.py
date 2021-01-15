import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import Preprocessing
import Model1_MUL_Features
import Model2_POLY

songs_data = pd.read_csv('spotify_training.csv')
Songs_df = pd.DataFrame(songs_data)

Songs_df = Preprocessing.preprocess_Data(Songs_df)

X = Songs_df[['year', 'energy', 'loudness']]
Y = Songs_df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)
Model1_MUL_Features.Mul_features(X_train, y_train , X_test,  y_test)
Model2_POLY.Poly(X_train, y_train , X_test,  y_test)










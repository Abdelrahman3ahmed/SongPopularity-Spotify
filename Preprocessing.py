import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import seaborn as sns



def preprocess_Data(Songs_df):
    songs_data = pd.read_csv('spotify_training.csv')
    Songs_df = pd.DataFrame(songs_data)
    Songs_df.dropna(how='any', inplace=True)
    Songs_df = One_hot_encoder_KEY(Songs_df)
    Songs_df = Labeling_Artists(Songs_df)
    Correlation(Songs_df)

    Songs_df.drop(['release_date', 'name', 'id', 'key'], axis=1, inplace=True)

    return Songs_df

def One_hot_encoder_KEY(Songs_df):
    art = OneHotEncoder(handle_unknown='ignore')
    Artists_df = pd.DataFrame(art.fit_transform(Songs_df[['key']].values).toarray())
    Songs_df = Songs_df.join(Artists_df)
    return Songs_df
def Labeling_Artists(Songs_df):
    labeling = preprocessing.LabelEncoder()
    Songs_df['artists'] = (labeling.fit_transform(list(Songs_df['artists'].values)))
    return Songs_df
def Correlation(Songs_df):
    corr = Songs_df.corr()
    top_feature = corr.index[abs(corr['popularity'] > 0.1)]
    plt.subplots(figsize=(12, 8))
    top_corr = Songs_df[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

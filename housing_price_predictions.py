import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import fetch_california_housing

#  load the dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)


X = df.drop(columns=['PRICE'])  # This will be the features (we want to predict price)
y = df['PRICE']  # This will be price targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# More work to be done

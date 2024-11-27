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

df['PRICE'] = california.target

print(df)

# More work to be done

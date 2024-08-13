import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import keras 
import tensorflow as tf 

dataset=pd.read_csv('data.csv')
cols_not_needed=['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker']
ipl_data=dataset.drop(cols_not_needed,axis=1)
x = ipl_data.drop(['total'], axis =1)
y = ipl_data['total']


# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit and transform the categorical features with label encoding
x['venue'] = venue_encoder.fit_transform(x['venue'])
x['batting_team'] = batting_team_encoder.fit_transform(x['batting_team'])
x['bowling_team'] = bowling_team_encoder.fit_transform(x['bowling_team'])
x['batsman'] = striker_encoder.fit_transform(x['batsman'])
x['bowler'] = bowler_encoder.fit_transform(x['bowler'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Input( shape=(x_train_scaled.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])

# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # You can adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)

model.fit(x_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(x_test_scaled, y_test))

model_losses = pd.DataFrame(model.history.history)
model_losses.plot()

predictions = model.predict(x_test_scaled)

mean_absolute_error(y_test,predictions)


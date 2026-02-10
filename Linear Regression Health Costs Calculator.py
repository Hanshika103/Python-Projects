import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# --------------------------------
# Convert categorical data to numbers
# --------------------------------
dataset = dataset.copy()

dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['smoker'] = dataset['smoker'].map({'no': 0, 'yes': 1})
dataset['region'] = dataset['region'].map({
    'southwest': 0,
    'southeast': 1,
    'northwest': 2,
    'northeast': 3
})

# --------------------------------
# Split dataset (80% train, 20% test)
# --------------------------------
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# --------------------------------
# Separate labels
# --------------------------------
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# --------------------------------
# Normalize numerical data
# --------------------------------
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

# --------------------------------
# Build regression model
# --------------------------------
model = keras.Sequential([
    normalizer,
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae']
)

# --------------------------------
# Train model
# --------------------------------
model.fit(
    train_dataset,
    train_labels,
    epochs=100,
    validation_split=0.2,
    verbose=0
)

# --------------------------------
# Evaluate model
# --------------------------------
model.evaluate(test_dataset, test_labels, verbose=2)

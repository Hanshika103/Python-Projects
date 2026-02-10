import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --------------------------------
# Prepare training and test data
# --------------------------------
train_messages = train_data['message'].values
train_labels = train_data['label'].values

test_messages = test_data['message'].values
test_labels = test_data['label'].values

# --------------------------------
# Text vectorization
# --------------------------------
vectorizer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)

vectorizer.adapt(train_messages)

# --------------------------------
# Build neural network model
# --------------------------------
model = keras.Sequential([
    vectorizer,
    layers.Embedding(10000, 64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------------
# Train model
# --------------------------------
model.fit(
    train_messages,
    train_labels,
    epochs=10,
    validation_split=0.2,
    verbose=0
)

# --------------------------------
# Prediction function (REQUIRED)
# --------------------------------
def predict_message(message):
    prediction = model.predict([message], verbose=0)[0][0]
    label = "spam" if prediction >= 0.5 else "ham"
    return [float(prediction), label]

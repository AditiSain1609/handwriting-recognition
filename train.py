import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load dataset
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

# Split
y = data.iloc[:, 0].values
x = data.iloc[:, 1:].values

# Fix labels
y = y.astype('int')

# Normalize
x = x / 255.0

# Reshape
x = x.reshape(-1, 28, 28, 1)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),   # increased dropout

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(26, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train

model.fit(x, y, epochs=7, batch_size=128)
# Save
model.save("model.h5")

print("✅ Final Alphabet Model Ready!")
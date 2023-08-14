import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import os

# Load simulated data
def load_simulated_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    obstacles = data['obstacles']
    elements = data['elements']
    return obstacles, elements

# Define the neural network model architecture
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)  # Adding another hidden layer
    outputs = Dense(2, activation='linear')(x) # Adjust output dimensions based on your prediction task

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Train the model
def train_model(model, x_train, y_train, x_val, y_val, num_epochs=200, batch_size=64, learning_rate=0.0001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size, verbose=1)
    return history

if __name__ == "__main__":
    data_dir = "../UAV-RIS-Optimization/Model/data"
    data_filename = "simulated_data.npz"
    data_path = os.path.join(data_dir, data_filename)
    
    obstacles, elements = load_simulated_data(data_path)
    
    # Preprocess your data and create x_train and y_train
    # Flatten the obstacles array
    flattened_obstacles = obstacles.reshape(-1, 1)

    # Combine flattened obstacles and elements to create input features
    x_train = np.concatenate((flattened_obstacles, elements), axis=1)
    
    # Create corresponding target values (y_train) based on your problem
    num_samples = x_train.shape[0]
    y_train = np.random.random((num_samples, 2)) 
    
    # Split the data into training and validation sets
    split_ratio = 0.8  # Adjust as needed
    split_index = int(split_ratio * num_samples)
    x_train, x_val = x_train[:split_index], x_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]
    
    num_features = x_train.shape[1]
    model = create_model(input_shape=(num_features,))
    model.summary()
    
    # Train the model with validation data
    history = train_model(model, x_train, y_train, x_val, y_val)
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

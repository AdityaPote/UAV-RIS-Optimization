import os
import numpy as np
from model import create_model, train_model, load_simulated_data
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    # Data paths
    data_dir = "../UAV-RIS-Optimization/Model/data"
    models_dir = "../UAV-RIS-Optimization/Model/models"
    data_filename = "simulated_data.npz"
    data_path = os.path.join(data_dir, data_filename)
    
    # Load simulated data
    obstacles, elements = load_simulated_data(data_path)
    
    # Preprocess data and create x_train and y_train
    flattened_obstacles = obstacles.reshape(-1, 1)
    x_train = np.concatenate((flattened_obstacles, elements), axis=1)
    num_samples = x_train.shape[0]
    y_train = np.random.random((num_samples, 2))  # Adjust target values based on your problem
    
    num_features = x_train.shape[1]
    
    # Split the data into training and validation sets
    split_ratio = 0.8  # Adjust as needed
    split_index = int(split_ratio * num_samples)
    x_train, x_val = x_train[:split_index], x_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]
    
    # Create and compile the model
    model = create_model(input_shape=(num_features,))
    model.summary()
    
    # Train the model
    history = train_model(model, x_train, y_train, x_val, y_val)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save the trained model using model.save()
    model_save_path = os.path.join(models_dir, "trained_model")
    model.save(model_save_path)
    print("Trained model saved to:", model_save_path)

    
    # Plot training loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

if __name__ == "__main__":
    main()

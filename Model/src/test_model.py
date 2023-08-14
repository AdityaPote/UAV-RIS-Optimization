import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import load_simulated_data

def main():
    # Data paths
    data_dir = "../UAV-RIS-Optimization/Model/data"
    models_dir = "../UAV-RIS-Optimization/Model/models"
    data_filename = "simulated_data.npz"
    model_filename = "trained_model"  # Change this based on the saved model name
    data_path = os.path.join(data_dir, data_filename)
    
    # Load simulated data
    obstacles, elements = load_simulated_data(data_path)
    
    # Preprocess data and create x_test
    flattened_obstacles = obstacles.reshape(-1, 1)
    x_test = np.concatenate((flattened_obstacles, elements), axis=1)
    
    # Load the trained model
    model_load_path = os.path.join(models_dir, model_filename)
    loaded_model = load_model(model_load_path)
    
    # Perform predictions on test data
    predictions = loaded_model.predict(x_test)
    
    # Display some sample predictions
    num_samples = min(5, x_test.shape[0])  # Display predictions for up to 5 samples
    for i in range(num_samples):
        print("Sample:", i + 1)
        print("Input Features:", x_test[i])
        print("Predictions:", predictions[i])
        print()

if __name__ == "__main__":
    main()

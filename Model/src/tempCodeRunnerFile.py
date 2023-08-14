import os
import numpy as np
from model import create_model, train_model, load_simulated_data
import matplotlib.pyplot as plt

def main():
    # Data paths
    data_dir = "../UAV-RIS-Optimization/Model/data"
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
    
    # Create and compile the model
    model = create_model(input_shape=(num_features,))
    model.summary()
    
    # Train the model
    history = train_model(model, x_train, y_train)
    
    # Save the trained model
    model_save_path = os.path.join(data_dir, "trained_model.h5")
    model.save(model_save_path)
    print("Trained model saved to:", model_save_path)
    
    # Plot training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()

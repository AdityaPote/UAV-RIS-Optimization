import numpy as np
import os

# Simulate data for RIS elements' characteristics
def simulate_ris_data(num_elements, save_path):
    # Simulate data based on your existing code
    grid_size = 11
    num_obstacles = 50
    obstacles = [(np.random.randint(0, grid_size), np.random.randint(0, grid_size)) for _ in range(num_obstacles)]
    elements = np.random.normal(0.5, 0.1, (num_elements, 4))
    
    # Create the data directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # Save the data
    np.savez(save_path, obstacles=obstacles, elements=elements)

# Example usage
if __name__ == "__main__":
    num_elements = 100
    data_dir = "../UAV-RIS-Optimization/Model/data"  # Adjust the directory path based on your project structure
    save_path = os.path.join(data_dir, "simulated_data.npz")
    simulate_ris_data(num_elements, save_path)
    print("Simulated data saved to:", save_path)

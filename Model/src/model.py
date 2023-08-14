import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

def load_simulated_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    obstacles = data['obstacles']
    elements = data['elements']
    return obstacles, elements

def calculate_snr_power(energy_indices, signal_indices):
    snr = np.random.random()
    power = np.random.random()
    return snr, power


def genetic_algorithm(num_elements, num_generations, population_size, crossover_rate, mutation_rate, evaluate_fitness):
    population = [np.random.choice(range(num_elements), num_elements // 2, replace=False) for _ in range(population_size)]
    
    for _ in range(num_generations):
        new_population = []
        for parent1, parent2 in zip(population[::2], population[1::2]):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, num_elements // 2)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1 = parent1
                child2 = parent2
            
            for child in [child1, child2]:
                if np.random.rand() < mutation_rate:
                    mutation_index = np.random.randint(num_elements // 2)
                    child[mutation_index] = np.random.randint(num_elements)
            new_population.extend([child1, child2])
        
        population = new_population
        
        best_individual = min(population, key=lambda ind: evaluate_fitness(ind, list(set(range(num_elements)) - set(ind))))
        
        best_energy_indices = best_individual
        best_signal_indices = list(set(range(num_elements)) - set(best_individual))
        best_snr, best_power = calculate_snr_power(best_energy_indices, best_signal_indices)
        
    return best_energy_indices, best_signal_indices, best_snr, best_power


def smarter_optimization(initial_energy_indices, initial_signal_indices, calculate_snr_power):
    initial_snr, initial_power = calculate_snr_power(initial_energy_indices, initial_signal_indices)
    best_energy_indices = initial_energy_indices.copy()
    best_signal_indices = initial_signal_indices.copy()
    best_combined_improvement = 0
    
    for _ in range(100):  
        improved = False
        for energy_index in initial_energy_indices:
            for signal_index in initial_signal_indices:
                new_energy_indices = [idx for idx in initial_energy_indices if idx != energy_index]
                new_signal_indices = [idx for idx in initial_signal_indices if idx != signal_index]
                new_energy_indices.append(signal_index)
                new_signal_indices.append(energy_index)
                
                new_snr, new_power = calculate_snr_power(new_energy_indices, new_signal_indices)
                snr_improvement = new_snr - initial_snr
                power_reduction = initial_power - new_power
                combined_improvement_metric = snr_improvement - 0.5 * power_reduction
                
                if combined_improvement_metric > best_combined_improvement:
                    best_combined_improvement = combined_improvement_metric
                    best_energy_indices = new_energy_indices
                    best_signal_indices = new_signal_indices
                    improved = True
        
        if not improved:
            break
    
    return best_energy_indices, best_signal_indices

def optimize_ris_distribution(obstacles, elements):
    num_elements = elements.shape[0]
    num_generations = 50
    population_size = 20
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    
    def evaluate_fitness(individual, complement_individual):
        return np.sum(individual), np.sum(complement_individual)
    
    
    best_energy_indices, best_signal_indices, _, _ = genetic_algorithm(
        num_elements, num_generations, population_size, crossover_rate, mutation_rate, evaluate_fitness
    )
    
    
    def calculate_snr_power(energy_indices, signal_indices):
        
        
        energy_values = np.random.random(len(energy_indices))
        signal_values = np.random.random(len(signal_indices))
        
        total_energy = np.sum(energy_values)
        total_signal = np.sum(signal_values)
        
        snr = total_signal / total_energy
        power = total_signal
        
        return snr, power

    
    optimized_ris_positions = np.array([elements[i, :2] for i in best_energy_indices + best_signal_indices])

    return optimized_ris_positions


def calculate_channel_characteristics(ris_positions):
    
    freq = 2.4e9  
    c = 3e8  

    
    tx_position = np.array([0, 0, 0])  
    rx_position = np.array([10, 10, 2])  

    
    ris_positions = np.array(ris_positions).reshape(-1, 2)

    
    tx_positions = np.tile(tx_position[:2], (ris_positions.shape[0], 1))

    
    tx_distances = np.linalg.norm(ris_positions - tx_positions, axis=1)
    rx_distances = np.linalg.norm(ris_positions - rx_position[:2], axis=1)

    
    path_loss = (c / (4 * np.pi * freq)) ** 2 * (tx_distances * rx_distances) ** (-2)

    
    return path_loss


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)  
    outputs = Dense(2, activation='linear')(x)  

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, x_train, y_train, x_val, y_val, num_epochs=200, batch_size=64, learning_rate=0.0001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size, verbose=1)
    return history

def main():
    data_dir = "../UAV-RIS-Optimization/Model/data"
    models_dir = "../UAV-RIS-Optimization/Model/models"
    data_filename = "simulated_data.npz"
    data_path = os.path.join(data_dir, data_filename)
    
    obstacles, elements = load_simulated_data(data_path)
    
    optimized_ris_positions = optimize_ris_distribution(obstacles, elements)
    
    channel_characteristics = calculate_channel_characteristics(optimized_ris_positions)
    
    flattened_obstacles = obstacles.reshape(-1, 1)
    x_train = np.concatenate((flattened_obstacles, optimized_ris_positions, channel_characteristics), axis=1)
    y_train = np.random.random((x_train.shape[0], 2))  # Replace with your target values
    
    num_samples = x_train.shape[0]
    num_features = x_train.shape[1]
    
    split_ratio = 0.8
    split_index = int(split_ratio * num_samples)
    x_train, x_val = x_train[:split_index], x_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]
    
    model = create_model(input_shape=(num_features,))
    model.summary()
    
    history = train_model(model, x_train, y_train, x_val, y_val)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_save_path = os.path.join(models_dir, "trained_model")
    model.save(model_save_path)
    print("Trained model saved to:", model_save_path)
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
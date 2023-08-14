    # Transmit and receive positions (for example)
    tx_position = np.array([0, 0, 0])  # Transmitting device position (x, y, z)
    rx_position = np.array([10, 10, 2])  # Receiving device position (x, y, z)

    # Ensure that ris_positions has the correct dimensions (num_elements, 3)
    ris_positions = np.array(ris_positions).reshape(-1, 3)

    # Calculate distances from RIS elements to the transmitter and receiver
    tx_distances = np.linalg.norm(ris_positions - tx_position, axis=1)
    rx_distances = np.linalg.norm(ris_positions - rx_position, axis=1)
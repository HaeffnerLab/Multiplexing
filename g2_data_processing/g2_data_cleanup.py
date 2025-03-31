import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Time windows for the 9 ion positions - based on the identified peak centers
# Each window is now twice as large as before
time_windows = [
    (-99.5, -95.5),   # Position 1 (furthest left) - doubled width
    (-86.5, -82.5),   # Position 2 - doubled width
    (-74.0, -70.0),   # Position 3 - doubled width
    (-62.5, -58.5),   # Position 4 - doubled width
    (-51.0, -47.0),   # Position 5 - doubled width
    (-39.0, -35.0),   # Position 6 - doubled width
    (-27.0, -23.0),   # Position 7 - doubled width
    (-15.0, -11.0),   # Position 8 - doubled width
    (-3.0, 1.0)       # Position 9 (closest to center) - doubled width
]

# Define corresponding positive time windows - also doubled in width
pos_time_windows = [
    (11.0, 15.0),     # Position 1 - doubled width
    (23.0, 27.0),     # Position 2 - doubled width
    (35.0, 39.0),     # Position 3 - doubled width
    (47.0, 51.0),     # Position 4 - doubled width
    (59.0, 63.0),     # Position 5 - doubled width
    (71.0, 75.0),     # Position 6 - doubled width
    (83.0, 87.0),     # Position 7 - doubled width
    (95.0, 99.0),     # Position 8 - doubled width
    (107.0, 111.0)    # Position 9 - doubled width
]

# Load the index file from parent directory
index_file_path = os.path.join('..', 'g2_data', '1_index_center_ion_data.npy')
time_bins = np.load(index_file_path)

def process_left_ion_data(file_path, position_index):
    """
    Process left ion data files
    Keep left (9-i) peaks, right (i-1) peaks, and the center peak
    """
    data = np.load(file_path)
    
    # Create a mask of all zeros
    mask = np.zeros_like(data, dtype=bool)
    
    # Identify where the time bins fall within the windows we want to keep
    
    # Center peak - always keep - now using wider window
    center_condition = (time_bins >= -3.0) & (time_bins <= 1.0)
    mask = mask | center_condition
    
    # Keep left (9-position_index) peaks
    for j in range(9-position_index):
        left_condition = (time_bins >= time_windows[j][0]) & (time_bins <= time_windows[j][1])
        mask = mask | left_condition
    
    # Keep right (position_index-1) peaks
    for j in range(position_index-1):
        right_condition = (time_bins >= pos_time_windows[j][0]) & (time_bins <= pos_time_windows[j][1])
        mask = mask | right_condition
    
    # Apply the mask
    cleaned_data = np.zeros_like(data)
    cleaned_data[mask] = data[mask]
    
    return cleaned_data, data

def process_right_ion_data(file_path, position_index):
    """
    Process right ion data files
    Keep left (i-1) peaks, right (9-i) peaks, and the center peak
    """
    data = np.load(file_path)
    
    # Create a mask of all zeros
    mask = np.zeros_like(data, dtype=bool)
    
    # Identify where the time bins fall within the windows we want to keep
    
    # Center peak - always keep - now using wider window
    center_condition = (time_bins >= -3.0) & (time_bins <= 1.0)
    mask = mask | center_condition
    
    # Keep left (position_index-1) peaks
    for j in range(position_index-1):
        left_condition = (time_bins >= time_windows[j][0]) & (time_bins <= time_windows[j][1])
        mask = mask | left_condition
    
    # Keep right (9-position_index) peaks
    for j in range(9-position_index):
        right_condition = (time_bins >= pos_time_windows[j][0]) & (time_bins <= pos_time_windows[j][1])
        mask = mask | right_condition
    
    # Apply the mask
    cleaned_data = np.zeros_like(data)
    cleaned_data[mask] = data[mask]
    
    return cleaned_data, data

def plot_data(time_bins, cleaned_data, original_data, title, file_path, position_index, is_left=True):
    """Plot and save the data visualization with original data, cleaned data, and time window indicators"""
    plt.figure(figsize=(14, 8))
    
    # Plot original data
    plt.plot(time_bins, original_data, 'b-', alpha=0.5, label='Original Data')
    
    # Plot cleaned data
    plt.plot(time_bins, cleaned_data, 'r-', label='Cleaned Data')
    
    # Plot vertical dashed lines to indicate time windows
    
    # Center window - always included - now using wider window
    plt.axvline(x=-3.0, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=0.0, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=1.0, color='g', linestyle='--', alpha=0.7)
    
    # Left time windows
    if is_left:
        # For left ion data: keep left (9-position_index) peaks
        for j in range(9-position_index):
            plt.axvline(x=time_windows[j][0], color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=time_windows[j][1], color='g', linestyle='--', alpha=0.7)
        
        # For left ion data: keep right (position_index-1) peaks
        for j in range(position_index-1):
            plt.axvline(x=pos_time_windows[j][0], color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=pos_time_windows[j][1], color='g', linestyle='--', alpha=0.7)
    else:
        # For right ion data: keep left (position_index-1) peaks
        for j in range(position_index-1):
            plt.axvline(x=time_windows[j][0], color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=time_windows[j][1], color='g', linestyle='--', alpha=0.7)
        
        # For right ion data: keep right (9-position_index) peaks
        for j in range(9-position_index):
            plt.axvline(x=pos_time_windows[j][0], color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=pos_time_windows[j][1], color='g', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time (ns)')
    plt.ylabel('Counts')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(file_path, dpi=150)
    plt.close()

# Process all left ion data files
for i in range(1, 10):
    left_file_path = os.path.join('..', 'g2_data', f'{i}_final_result_left_ion_data.npy')
    
    # Check if file exists
    if os.path.exists(left_file_path):
        # Process the file
        cleaned_data, original_data = process_left_ion_data(left_file_path, i)
        
        # Save the cleaned data
        output_file = f'f{i}_cleaned_left_ion_data.npy'
        np.save(output_file, cleaned_data)
        
        # Plot the cleaned data
        plot_file = f'f{i}_cleaned_left_ion_data.png'
        plot_data(time_bins, cleaned_data, original_data, 
                  f'Cleaned Left Ion Data - Position {i}', plot_file, i, is_left=True)
        
        print(f"Processed and saved {output_file}")

# Process all right ion data files
for i in range(1, 10):
    right_file_path = os.path.join('..', 'g2_data', f'{i}_final_result_right_ion_data.npy')
    
    # Check if file exists
    if os.path.exists(right_file_path):
        # Process the file
        cleaned_data, original_data = process_right_ion_data(right_file_path, i)
        
        # Save the cleaned data
        output_file = f'f{i}_cleaned_right_ion_data.npy'
        np.save(output_file, cleaned_data)
        
        # Plot the cleaned data
        plot_file = f'f{i}_cleaned_right_ion_data.png'
        plot_data(time_bins, cleaned_data, original_data,
                  f'Cleaned Right Ion Data - Position {i}', plot_file, i, is_left=False)
        
        print(f"Processed and saved {output_file}")

print("Data cleanup complete!")

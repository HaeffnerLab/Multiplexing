import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Scaling parameter for time window widths (1.0 = default width)
time_window_scale = 1.0  # Adjust this value to scale all time window widths

# Time windows for the 9 ion positions - based on the identified peak centers
# Reordered from closest to center to furthest away (negative side)
# Define window centers and base half-widths
window_centers = -np.array([-1.0, -13.0, -25.0, -37.0, -49.0, -60.5, -72.0, -84.5, -97.5])
window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 1.5

# Generate time windows with scaled widths
time_windows = []
for center, half_width in zip(window_centers, window_half_widths):
    scaled_half_width = half_width * time_window_scale
    time_windows.append((center - scaled_half_width, center + scaled_half_width))

# Define corresponding positive time window centers and half-widths
# Reordered from closest to center to furthest away (positive side)
pos_window_centers = -np.array([3.0, 13.0, 25.0, 37.0, 49.0, 61.0, 73.0, 85.0, 97.0])
pos_window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 1.5

# Generate positive time windows with scaled widths
pos_time_windows = []
for center, half_width in zip(pos_window_centers, pos_window_half_widths):
    scaled_half_width = half_width * time_window_scale
    pos_time_windows.append((center - scaled_half_width, center + scaled_half_width))

# Load the index file from parent directory
index_file_path = os.path.join('..', 'g2_data', '1_index_center_ion_data.npy')
time_bins = np.load(index_file_path)

def process_left_ion_data(file_path, position_index):
    """
    Process left ion data files
    Keep peaks starting from the center and extending outward
    """
    data = np.load(file_path)
    
    # Create a mask of all zeros
    mask = np.zeros_like(data, dtype=bool)
    
    # Always keep the center peak
    center_condition = (time_bins >= -3.0) & (time_bins <= 1.0)
    mask = mask | center_condition
    
    # For left ion at position i, we want to keep:
    # 1. Center peak (always)
    # 2. i peaks on the negative side (closest to center first)
    # 3. (9-i) peaks on the positive side (closest to center first)
    
    # Keep i peaks on the negative side (starting from center)
    for j in range(min(position_index, 9)):
        if j < len(time_windows):
            left_condition = (time_bins >= time_windows[j][0]) & (time_bins <= time_windows[j][1])
            mask = mask | left_condition
    
    # Keep (9-i) peaks on the positive side (starting from center)
    for j in range(min(9-position_index+1, 9)):
        if j < len(pos_time_windows):
            right_condition = (time_bins >= pos_time_windows[j][0]) & (time_bins <= pos_time_windows[j][1])
            mask = mask | right_condition
    
    # Apply the mask
    cleaned_data = np.zeros_like(data)
    cleaned_data[mask] = data[mask]
    
    return cleaned_data, data

def process_right_ion_data(file_path, position_index):
    """
    Process right ion data files
    Keep peaks starting from the center and extending outward
    """
    data = np.load(file_path)
    
    # Create a mask of all zeros
    mask = np.zeros_like(data, dtype=bool)
    
    # Always keep the center peak
    center_condition = (time_bins >= -3.0) & (time_bins <= 1.0)
    mask = mask | center_condition
    
    # For right ion at position i, we want to keep:
    # 1. Center peak (always)
    # 2. (9-i) peaks on the negative side (closest to center first)
    # 3. i peaks on the positive side (closest to center first)
    
    # Keep (9-i) peaks on the negative side (starting from center)
    for j in range(min(9-position_index+1, 9)):
        if j < len(time_windows):
            left_condition = (time_bins >= time_windows[j][0]) & (time_bins <= time_windows[j][1])
            mask = mask | left_condition
    
    # Keep i peaks on the positive side (starting from center)
    for j in range(min(position_index, 9)):
        if j < len(pos_time_windows):
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
    
    # Always show the center window
    center_window = time_windows[0]
    plt.axvline(x=center_window[0], color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=center_window[1], color='g', linestyle='--', alpha=0.7)
    
    # Show windows for the peaks we're keeping
    if is_left:
        # For left ion data, show i peaks on negative side and (9-i) peaks on positive side
        for j in range(min(position_index, 9)):
            if j < len(time_windows):
                plt.axvline(x=time_windows[j][0], color='g', linestyle='--', alpha=0.7)
                plt.axvline(x=time_windows[j][1], color='g', linestyle='--', alpha=0.7)
        
        for j in range(min(9-position_index+1, 9)):
            if j < len(pos_time_windows):
                plt.axvline(x=pos_time_windows[j][0], color='g', linestyle='--', alpha=0.7)
                plt.axvline(x=pos_time_windows[j][1], color='g', linestyle='--', alpha=0.7)
    else:
        # For right ion data, show (9-i) peaks on negative side and i peaks on positive side
        for j in range(min(9-position_index+1, 9)):
            if j < len(time_windows):
                plt.axvline(x=time_windows[j][0], color='g', linestyle='--', alpha=0.7)
                plt.axvline(x=time_windows[j][1], color='g', linestyle='--', alpha=0.7)
        
        for j in range(min(position_index, 9)):
            if j < len(pos_time_windows):
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

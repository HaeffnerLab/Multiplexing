import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import find_peaks, peak_widths
import re
from scipy.interpolate import interp1d

# Scaling parameter for time window widths (1.0 = default width)
time_window_scale = 1.0  # Adjust this value to scale all time window widths

# Time windows for the 8 ion positions - based on the identified peak centers
# Reordered from closest to center to furthest away (negative side)
# Define window centers and base half-widths
window_centers = -np.array([-13.0, -25.0, -37.0, -49.0, -60.5, -72.0, -84.5, -97.5])
window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 1.5

# Generate time windows with scaled widths
time_windows = []
for center, half_width in zip(window_centers, window_half_widths):
    scaled_half_width = half_width * time_window_scale
    time_windows.append((center - scaled_half_width, center + scaled_half_width))

# Define corresponding positive time window centers and half-widths
# Reordered from closest to center to furthest away (positive side)
pos_window_centers = np.array([-13.0, -25.0, -37.0, -49.0, -60.5, -72.0, -84.5, -97.5])
pos_window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 1.5

# Generate positive time windows with scaled widths
pos_time_windows = []
for center, half_width in zip(pos_window_centers, pos_window_half_widths):
    scaled_half_width = half_width * time_window_scale
    pos_time_windows.append((center - scaled_half_width, center + scaled_half_width))

def extract_peaks_in_windows(data, time_bins):
    """
    Extract all peaks from data within defined time windows
    
    Args:
        data: Array of y values (counts)
        time_bins: Array of x values (time)
        
    Returns:
        peaks_info: Dictionary with peak information for each window
    """
    peaks_info = {}
    
    # Process center peak first (assuming it exists around 0)
    center_window = (-3.0, 3.0)
    center_mask = (time_bins >= center_window[0]) & (time_bins <= center_window[1])
    if np.any(center_mask):
        window_data = data[center_mask]
        window_time = time_bins[center_mask]
        
        # Find peaks in the window data
        if len(window_data) > 0 and np.max(window_data) > 0:
            min_height = 0.2 * np.max(window_data)
            peak_indices, _ = find_peaks(window_data, height=min_height)
            
            if len(peak_indices) > 0:
                # Find the highest peak
                highest_peak_idx = np.argmax(window_data[peak_indices])
                peak_idx = peak_indices[highest_peak_idx]
                
                # Store peak information
                peaks_info['center'] = {
                    'time': window_time[peak_idx],
                    'value': window_data[peak_idx],
                    'target_center': 0,
                    'indices': np.where(center_mask)[0],
                    'window_data': window_data,
                    'window_time': window_time,
                    'shift': 0 - window_time[peak_idx]  # Shift to move peak to center
                }
    
    # Process negative windows
    for i, (window, center) in enumerate(zip(time_windows, window_centers)):
        window_mask = (time_bins >= window[0]) & (time_bins <= window[1])
        if np.any(window_mask):
            window_data = data[window_mask]
            window_time = time_bins[window_mask]
            
            # Find peaks in the window data
            if len(window_data) > 0 and np.max(window_data) > 0:
                min_height = 0.2 * np.max(window_data)
                peak_indices, _ = find_peaks(window_data, height=min_height)
                
                if len(peak_indices) > 0:
                    # Find the highest peak
                    highest_peak_idx = np.argmax(window_data[peak_indices])
                    peak_idx = peak_indices[highest_peak_idx]
                    
                    # Store peak information
                    peaks_info[f'neg_{i}'] = {
                        'time': window_time[peak_idx],
                        'value': window_data[peak_idx],
                        'target_center': center,
                        'indices': np.where(window_mask)[0],
                        'window_data': window_data,
                        'window_time': window_time,
                        'shift': center - window_time[peak_idx]  # Shift to move peak to center
                    }
    
    # Process positive windows
    for i, (window, center) in enumerate(zip(pos_time_windows, pos_window_centers)):
        window_mask = (time_bins >= window[0]) & (time_bins <= window[1])
        if np.any(window_mask):
            window_data = data[window_mask]
            window_time = time_bins[window_mask]
            
            # Find peaks in the window data
            if len(window_data) > 0 and np.max(window_data) > 0:
                min_height = 0.2 * np.max(window_data)
                peak_indices, _ = find_peaks(window_data, height=min_height)
                
                if len(peak_indices) > 0:
                    # Find the highest peak
                    highest_peak_idx = np.argmax(window_data[peak_indices])
                    peak_idx = peak_indices[highest_peak_idx]
                    
                    # Store peak information
                    peaks_info[f'pos_{i}'] = {
                        'time': window_time[peak_idx],
                        'value': window_data[peak_idx],
                        'target_center': center,
                        'indices': np.where(window_mask)[0],
                        'window_data': window_data,
                        'window_time': window_time,
                        'shift': center - window_time[peak_idx]  # Shift to move peak to center
                    }
    
    return peaks_info

def align_all_peaks(data, time_bins, is_left=True):
    """
    Create a new dataset with all peaks aligned to their respective window centers
    
    Args:
        data: Array of y values (counts)
        time_bins: Array of x values (time)
        is_left: Boolean indicating if this is left ion data
        
    Returns:
        aligned_data: Data with all peaks aligned
        peaks_info: Dictionary with peak information for alignment
    """
    # Extract all peaks and their information
    peaks_info = extract_peaks_in_windows(data, time_bins)
    
    # Create a new array to hold the aligned data
    aligned_data = np.zeros_like(data)
    
    # Instead of trying to modify the original data, we'll create a new array
    # with the shifted peaks
    for window_key, peak_info in peaks_info.items():
        # Calculate the amount to shift this peak
        shift = peak_info['shift']
        
        # Create shifted x-values for interpolation
        shifted_times = peak_info['window_time'] + shift
        
        # Find the closest indices in the original time_bins array
        # This is important to correctly place the shifted peak in the output array
        new_indices = np.searchsorted(time_bins, shifted_times)
        
        # Make sure we don't go out of bounds
        valid_mask = (new_indices >= 0) & (new_indices < len(time_bins))
        new_indices = new_indices[valid_mask]
        
        if len(new_indices) > 0:
            # Place the peak data at the new shifted positions
            # Use min to avoid index errors if we're near the array boundary
            max_idx = min(len(peak_info['window_data']), len(valid_mask))
            aligned_data[new_indices] = peak_info['window_data'][valid_mask][:max_idx]
    
    return aligned_data, peaks_info

def improved_align_all_peaks(data, time_bins, is_left=True):
    """
    Improved version of peak alignment that uses interpolation to properly shift each peak
    
    Args:
        data: Array of y values (counts)
        time_bins: Array of x values (time)
        is_left: Boolean indicating if this is left ion data
        
    Returns:
        aligned_data: Data with all peaks aligned
        peaks_info: Dictionary with peak information for alignment
    """
    # Extract all peaks and their information
    peaks_info = extract_peaks_in_windows(data, time_bins)
    
    # Create a new array to hold the aligned data
    aligned_data = np.zeros_like(data)
    
    # Define a broader interpolation of the original data
    # This handles regions outside the original range by setting to 0
    f_data = interp1d(time_bins, data, bounds_error=False, fill_value=0)
    
    # For each window, create a shifted version of the data
    for window_key, peak_info in peaks_info.items():
        # Get the shift amount for this peak
        shift = peak_info['shift']
        
        # Get the window boundaries
        window_mask = (time_bins >= peak_info['window_time'][0] - 1) & (time_bins <= peak_info['window_time'][-1] + 1)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) > 0:
            # Calculate the shifted time points for this window
            shifted_time_points = time_bins[window_indices] - shift
            
            # Use the interpolation function to get values at the shifted points
            shifted_values = f_data(shifted_time_points)
            
            # Add the shifted values to the aligned data
            aligned_data[window_indices] = shifted_values
    
    return aligned_data, peaks_info

def plot_alignment_result(time_bins, original_data, aligned_data, peaks_info, title, save_path):
    """
    Plot the original and aligned data, with markers for window centers and peaks
    
    Args:
        time_bins: Array of x values (time)
        original_data: Original y values
        aligned_data: Aligned y values
        peaks_info: Dictionary with peak information
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the original and aligned data
    ax.plot(time_bins, original_data, 'b-', alpha=0.5, label='Original Data')
    ax.plot(time_bins, aligned_data, 'r-', label='Aligned Data')
    
    # Plot window centers and borders, and annotate with shift amount
    for window_key, peak_info in peaks_info.items():
        # Mark window center (target position)
        ax.axvline(x=peak_info['target_center'], color='g', linestyle='--', alpha=0.7)
        
        # Mark window borders
        window_time = peak_info['window_time']
        ax.axvline(x=window_time[0], color='g', linestyle=':', alpha=0.4)
        ax.axvline(x=window_time[-1], color='g', linestyle=':', alpha=0.4)
        
        # Add annotation about the shift
        ax.annotate(f"{peak_info['shift']:.2f}",
                   xy=(peak_info['target_center'], 0.9 * np.max(aligned_data)),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8)
        
        # Mark the original peak position with a vertical line
        ax.axvline(x=peak_info['time'], color='b', linestyle=':', alpha=0.4)
    
    ax.set_title(title)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_data_files():
    """
    Process all the *_final_result_left_ion_data.npy and *_final_result_right_ion_data.npy files
    """
    # Get the pattern for the files with path relative to the current directory
    left_pattern = os.path.join('..', 'g2_data', '*_final_result_left_ion_data.npy')
    right_pattern = os.path.join('..', 'g2_data', '*_final_result_right_ion_data.npy')
    
    # Get all matching files
    left_files = glob.glob(left_pattern)
    right_files = glob.glob(right_pattern)
    
    print(f"Found {len(left_files)} left ion files and {len(right_files)} right ion files")
    
    # Processing all files
    for file_path in left_files + right_files:
        print(f"Processing {file_path}...")
        
        # Extract position index from filename (e.g., ../g2_data/1_final_result_left_ion_data.npy -> 1)
        index_match = re.match(r'\.\.\/g2_data\/(\d+)_final_result_(\w+)_ion_data\.npy', file_path)
        if index_match:
            position_index = int(index_match.group(1))
            ion_type = index_match.group(2)  # left or right
        else:
            print(f"  Could not extract position index from {file_path}")
            continue
        
        # Get corresponding index file
        index_file = os.path.join('..', 'g2_data', f'{position_index}_index_{ion_type}_ion_data.npy')
        if not os.path.exists(index_file):
            print(f"  Index file {index_file} not found")
            continue
        
        # Load the data
        try:
            time_bins = np.load(index_file)
            data = np.load(file_path)
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue
        
        # Align peaks
        is_left = ion_type == 'left'
        aligned_data, peaks_info = improved_align_all_peaks(data, time_bins, is_left)
        
        # Create file names for saving
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        aligned_file = f"{base_name}_aligned.npy"
        plot_file = f"{base_name}_aligned.png"
        
        # Save aligned data
        np.save(aligned_file, aligned_data)
        
        # Create and save plot
        title = f"Aligned Peaks for {ion_type.capitalize()} Ion at Position {position_index}"
        plot_alignment_result(time_bins, data, aligned_data, peaks_info, title, plot_file)
        
        print(f"  Saved aligned data to {aligned_file}")
        print(f"  Saved plot to {plot_file}")

if __name__ == "__main__":
    process_data_files()

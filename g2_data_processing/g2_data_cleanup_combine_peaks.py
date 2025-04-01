import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import find_peaks
import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Time windows for the 9 ion positions - based on the identified peak centers
# Reordered from closest to center to furthest away (negative side)
window_centers = -np.array([-1.0, -13.0, -25.0, -37.0, -49.0, -60.5, -72.0, -84.5, -97.5])
window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 2

# Generate time windows
time_windows = []
for center, half_width in zip(window_centers, window_half_widths):
    time_windows.append((center - half_width, center + half_width))

# Define corresponding positive time window centers and half-widths
# Reordered from closest to center to furthest away (positive side)
pos_window_centers = -np.array([3.0, 13.0, 25.0, 37.0, 49.0, 61.0, 73.0, 85.0, 97.0])
pos_window_half_widths = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 2

# Generate positive time windows
pos_time_windows = []
for center, half_width in zip(pos_window_centers, pos_window_half_widths):
    pos_time_windows.append((center - half_width, center + half_width))

def detect_and_align_peaks(time_bins, data, windows, window_centers, half_width_for_shift):
    """
    Detect peaks in each window and align them to the center of the window
    
    Args:
        time_bins: Array of time values
        data: Array of count data
        windows: List of time window tuples (start, end)
        window_centers: Array of window centers
        half_width_for_shift: Half width of region around peak to shift
        
    Returns:
        aligned_data: Data with peaks aligned to window centers
    """
    aligned_data = np.zeros_like(data)
    
    # For debugging
    peak_locations = []
    
    # Process each window
    for i, ((window_start, window_end), center) in enumerate(zip(windows, window_centers)):
        # Create mask for the current window
        window_mask = (time_bins >= window_start) & (time_bins <= window_end)
        
        # Skip if no data in this window
        if not np.any(window_mask):
            continue
        
        # Extract window data
        window_times = time_bins[window_mask]
        window_data = data[window_mask]
        
        # Skip if all data is zero
        if np.all(window_data == 0):
            continue
        
        # Find the weighted center of mass of the window data
        # This will be more reliable than trying to detect individual peaks
        if np.sum(window_data) > 0:  # Prevent division by zero
            weighted_center = np.sum(window_times * window_data) / np.sum(window_data)
        else:
            # If no meaningful data, use the center of the window's time range
            weighted_center = (window_start + window_end) / 2
            
        # Calculate shift amount to move the center of mass to the target center
        shift_amount = center - weighted_center
        
        # Store peak location for debugging
        peak_locations.append((weighted_center, center))
        
        # Shift ALL data points in the window
        # Calculate new time positions for all points in the window
        new_time_positions = window_times + shift_amount
        
        # Find closest indices in the original time_bins array
        original_indices = np.where(window_mask)[0]
        
        # For each original data point in the window, find where it should go
        for idx, orig_idx in enumerate(original_indices):
            new_pos = new_time_positions[idx]
            new_idx = np.abs(time_bins - new_pos).argmin()
            
            # Only assign if within array bounds
            if 0 <= new_idx < len(aligned_data):
                aligned_data[new_idx] += window_data[idx]
    
    return aligned_data, peak_locations

def process_file(file_index, half_width_for_shift=3.0, plot=True):
    """
    Process a single data file, detecting and aligning peaks
    
    Args:
        file_index: Index of the file to process
        half_width_for_shift: Half width of region around peak to shift
        plot: Whether to create plots
        
    Returns:
        Peak-aligned data
    """
    # Load the data files
    file_path = f'{file_index}_final_result_caled_ion_data.npy'
    index_file_path = os.path.join('..', 'g2_data', f'{file_index}_index_center_ion_data.npy')
    
    data = np.load(file_path)
    time_bins = np.load(index_file_path)
    
    # Process negative and positive time windows separately
    neg_aligned_data, neg_peaks = detect_and_align_peaks(
        time_bins, data, time_windows, window_centers, half_width_for_shift
    )
    
    pos_aligned_data, pos_peaks = detect_and_align_peaks(
        time_bins, data, pos_time_windows, pos_window_centers, half_width_for_shift
    )
    
    # Combine the aligned data
    combined_aligned_data = neg_aligned_data + pos_aligned_data
    
    # Save the aligned data
    output_file = f'{file_index}_aligned_peaks_ion_data.npy'
    np.save(output_file, combined_aligned_data)
    
    if plot:
        # Plot original and aligned data
        plt.figure(figsize=(14, 8))
        
        # Plot original data
        plt.plot(time_bins, data, 'b-', alpha=0.5, label='Original Data')
        
        # Plot aligned data
        plt.plot(time_bins, combined_aligned_data, 'r-', label='Aligned Peaks Data')
        
        # Define distinct colors for negative windows (using tab10, Set1, and Set2 colormaps)
        distinct_colors_neg = []
        
        # Add colors from tab10 (10 colors)
        distinct_colors_neg.extend(plt.cm.tab10(np.arange(10)))
        
        # Add colors from Set1 (9 colors) that are not too similar to tab10
        distinct_colors_neg.extend(plt.cm.Set1(np.arange(9)))
        
        # Add colors from Set2 (8 colors)
        distinct_colors_neg.extend(plt.cm.Set2(np.arange(8)))
        
        # Ensure we have enough distinct colors
        n_neg_windows = len(window_centers)
        neg_colors = [distinct_colors_neg[i % len(distinct_colors_neg)] for i in range(n_neg_windows)]
        
        # For positive windows, use a different set of colors to further distinguish them
        distinct_colors_pos = []
        
        # Add colors from tab20 (20 colors)
        distinct_colors_pos.extend(plt.cm.tab20b(np.arange(20)))
        
        # Add colors from Set3 (12 colors)
        distinct_colors_pos.extend(plt.cm.Set3(np.arange(12)))
        
        # Ensure we have enough distinct colors
        n_pos_windows = len(pos_window_centers)
        pos_colors = [distinct_colors_pos[i % len(distinct_colors_pos)] for i in range(n_pos_windows)]
        
        # Plot vertical lines for window centers and boundaries with different colors
        for i, (center, (window_start, window_end)) in enumerate(zip(window_centers, time_windows)):
            color = neg_colors[i]
            # Center line (solid)
            plt.axvline(x=center, color=color, linestyle='-', alpha=0.7)
            # Boundary lines (dashed)
            plt.axvline(x=window_start, color=color, linestyle='--', alpha=0.5)
            plt.axvline(x=window_end, color=color, linestyle='--', alpha=0.5)
        
        for i, (center, (window_start, window_end)) in enumerate(zip(pos_window_centers, pos_time_windows)):
            color = pos_colors[i]
            # Center line (solid)
            plt.axvline(x=center, color=color, linestyle='-', alpha=0.7)
            # Boundary lines (dashed)
            plt.axvline(x=window_start, color=color, linestyle='--', alpha=0.5)
            plt.axvline(x=window_end, color=color, linestyle='--', alpha=0.5)
        
        plt.title(f'Original vs Aligned Peaks - Index {file_index}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'{file_index}_aligned_peaks.png', dpi=150)
        plt.close()
    
    return combined_aligned_data

def main():
    """
    Main function to process all files and combine peaks
    """
    parser = argparse.ArgumentParser(description='Process and align peaks in G2 data files')
    parser.add_argument('--half-width', type=float, default=3.0,
                      help='Half width of region around peak to shift (default: 3.0)')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable plotting of results')
    
    args = parser.parse_args()
    
    half_width_for_shift = args.half_width
    plot = not args.no_plot
    
    print(f"Processing files with half width for peak shift: {half_width_for_shift}")
    
    # Process all 9 files
    combined_data = None
    for i in range(1, 10):
        try:
            print(f"Processing file {i}...")
            aligned_data = process_file(i, half_width_for_shift, plot)
            
            # Combine all aligned data
            if combined_data is None:
                combined_data = aligned_data
            else:
                combined_data += aligned_data
                
            print(f"Successfully processed file {i}")
            
        except Exception as e:
            print(f"Error processing file {i}: {e}")
    
    # Save combined aligned data
    if combined_data is not None:
        np.save('combined_aligned_peaks_data.npy', combined_data)
        
        # Plot combined data
        if plot:
            # Load time bins
            index_file_path = os.path.join('..', 'g2_data', f'1_index_center_ion_data.npy')
            time_bins = np.load(index_file_path)
            
            plt.figure(figsize=(14, 8))
            plt.plot(time_bins, combined_data, 'b-')
            
            # Define distinct colors for negative windows (using tab10, Set1, and Set2 colormaps)
            distinct_colors_neg = []
            
            # Add colors from tab10 (10 colors)
            distinct_colors_neg.extend(plt.cm.tab10(np.arange(10)))
            
            # Add colors from Set1 (9 colors) that are not too similar to tab10
            distinct_colors_neg.extend(plt.cm.Set1(np.arange(9)))
            
            # Add colors from Set2 (8 colors)
            distinct_colors_neg.extend(plt.cm.Set2(np.arange(8)))
            
            # Ensure we have enough distinct colors
            n_neg_windows = len(window_centers)
            neg_colors = [distinct_colors_neg[i % len(distinct_colors_neg)] for i in range(n_neg_windows)]
            
            # For positive windows, use a different set of colors to further distinguish them
            distinct_colors_pos = []
            
            # Add colors from tab20 (20 colors)
            distinct_colors_pos.extend(plt.cm.tab20b(np.arange(20)))
            
            # Add colors from Set3 (12 colors)
            distinct_colors_pos.extend(plt.cm.Set3(np.arange(12)))
            
            # Ensure we have enough distinct colors
            n_pos_windows = len(pos_window_centers)
            pos_colors = [distinct_colors_pos[i % len(distinct_colors_pos)] for i in range(n_pos_windows)]
            
            # Plot vertical lines for window centers and boundaries with different colors
            for i, (center, (window_start, window_end)) in enumerate(zip(window_centers, time_windows)):
                color = neg_colors[i]
                # Center line (solid)
                plt.axvline(x=center, color=color, linestyle='-', alpha=0.7)
                # Boundary lines (dashed)
                plt.axvline(x=window_start, color=color, linestyle='--', alpha=0.5)
                plt.axvline(x=window_end, color=color, linestyle='--', alpha=0.5)
            
            for i, (center, (window_start, window_end)) in enumerate(zip(pos_window_centers, pos_time_windows)):
                color = pos_colors[i]
                # Center line (solid)
                plt.axvline(x=center, color=color, linestyle='-', alpha=0.7)
                # Boundary lines (dashed)
                plt.axvline(x=window_start, color=color, linestyle='--', alpha=0.5)
                plt.axvline(x=window_end, color=color, linestyle='--', alpha=0.5)
            
            plt.title('Combined Aligned Peaks Data')
            plt.xlabel('Time (ns)')
            plt.ylabel('Counts')
            plt.grid(True, alpha=0.3)
            plt.savefig('combined_aligned_peaks_data.png', dpi=150)
            plt.close()
            
        print("Combined aligned peaks data saved to combined_aligned_peaks_data.npy")
    else:
        print("No data was processed. Check for errors.")

if __name__ == "__main__":
    main()

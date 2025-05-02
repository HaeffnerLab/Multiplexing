import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator
import os

# File path setup
file_path = "/Users/QUBIT/Documents/GitHub/Untitled/Multiplexing/ion-shuttling/RigolDS4.csv"
if not os.path.exists(file_path):
    print(f"Warning: Could not find {file_path}")
    print("Please adjust the file path or place the CSV file in the correct location.")
    # Alternative: ask for file path input
    # file_path = input("Enter the full path to your CSV file: ")

try:
    # Load the CSV file
    data = pd.read_csv(file_path)
    print(f"Successfully loaded CSV with columns: {data.columns.tolist()}")
    
    # Applying a rolling average for smoothing
    window_size = 10
    smoothed_data = data.rolling(window=window_size).mean()

    # Extract column names based on your CSV structure
    # Adjust these column names to match your actual CSV file
    time_col = 'Time(s)' if 'Time(s)' in data.columns else 'Time (us)' if 'Time (us)' in data.columns else data.columns[0]
    value_col = 'CH1(V)' if 'CH1(V)' in data.columns else 'Value' if 'Value' in data.columns else data.columns[1]
    
    print(f"Using time column: {time_col} and value column: {value_col}")
    
    # Create binned data from the smoothed data
    # This is a flexible approach to handle different data formats
    binned_data = np.array(smoothed_data[value_col].dropna())
    
    # Determine time scaling based on column names
    time_scaling = 1e6 if 'Time(s)' in time_col else 1.0
    time_offset = 39.9 if 'Time(s)' in time_col else 0
    time_multiplier = 0.96 if 'Time(s)' in time_col else 1.0
    
    binned_hist_index = np.array(smoothed_data[time_col].dropna() * time_scaling * time_multiplier + time_offset)
    
    # Find peaks with appropriate parameters
    # Adjust these parameters based on your data characteristics
    peak_prominence = 1    # Minimum peak prominence
    peak_distance = 10     # Minimum distance between peaks
    
    peaks, properties = find_peaks(binned_data, prominence=peak_prominence, distance=peak_distance)
    
    # Select the top 9 peaks based on their prominences
    num_peaks = 9  # Number of peaks to analyze
    top_peaks_indices = np.argsort(properties['prominences'])[-num_peaks:]
    top_peaks = peaks[top_peaks_indices]
    top_peaks = np.sort(top_peaks)  # Sort peaks by position, not prominence
    
    # Set a threshold above background for peak region detection
    # Percentile-based approach for robust background estimation
    background_percentile = 50  # 50th percentile as baseline
    background_multiplier = 1.5  # Multiplier for threshold
    
    background_level = np.percentile(binned_data, background_percentile) * background_multiplier
    print(f"Background threshold: {background_level:.2f}")
    
    # Find regions for each peak
    peak_regions = []
    peak_counts = []
    
    for peak in top_peaks:
        # Start from the peak and go left until we hit the background level
        left_idx = peak
        while left_idx > 0 and binned_data[left_idx] > background_level:
            left_idx -= 1
        
        # Start from the peak and go right until we hit the background level
        right_idx = peak
        while right_idx < len(binned_data) - 1 and binned_data[right_idx] > background_level:
            right_idx += 1
        
        # Ensure minimum width for very narrow peaks
        min_width = 5  # Minimum width of peak region
        if right_idx - left_idx < min_width:
            pad = (min_width - (right_idx - left_idx)) // 2
            left_idx = max(0, left_idx - pad)
            right_idx = min(len(binned_data) - 1, right_idx + pad)
        
        # Store the region boundaries
        peak_regions.append((left_idx, right_idx))
        
        # Calculate the sum of counts in this region
        region_sum = np.sum(binned_data[left_idx:right_idx+1])
        peak_counts.append(region_sum)
        
        print(f"Peak {len(peak_regions)}: Region [{binned_hist_index[left_idx]:.2f}, {binned_hist_index[right_idx]:.2f}] μs, Sum: {region_sum:.0f} counts")
    
    # Calculate total counts across all peaks
    total_counts = sum(peak_counts)
    print(f"Total counts across all {num_peaks} peaks: {total_counts:.0f}")
    
    # Create a plot with the original data and highlighted peak regions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot the original data on the first subplot
    ax1.plot(binned_hist_index, binned_data, 'black', linewidth=1)
    ax1.set_ylabel('Counts', fontdict={'family': 'STIXGeneral', 'size': 12})
    ax1.grid(False)
    ax1.set_title('Original Data with Peak Markers', fontdict={'family': 'STIXGeneral', 'size': 12})
    
    # Add vertical lines at peak positions in the upper plot
    for peak in top_peaks:
        ax1.axvline(x=binned_hist_index[peak], color='red', linestyle='--', linewidth=1)
    
    # Plot the histogram with peak regions on the second subplot
    ax2.bar(binned_hist_index, binned_data, width=0.75, alpha=0.5, color='gray', label='All data')
    
    # Use different colors for each peak region
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_peaks)))
    
    # Plot colored regions and dashed lines for each peak
    for i, ((left_idx, right_idx), peak, color) in enumerate(zip(peak_regions, top_peaks, colors)):
        # Plot the region with a distinct color
        ax2.bar(binned_hist_index[left_idx:right_idx+1], 
                binned_data[left_idx:right_idx+1], 
                width=0.75, 
                color=color, 
                label=f'Peak {i+1}: {peak_counts[i]:.0f} counts')
        
        # Add vertical dashed lines at region boundaries
        ax2.axvline(x=binned_hist_index[left_idx], color='black', linestyle='--', linewidth=1)
        ax2.axvline(x=binned_hist_index[right_idx], color='black', linestyle='--', linewidth=1)
        
        # Annotate each peak with its count
        ax2.annotate(f'{peak_counts[i]:.0f}', 
                    xy=(binned_hist_index[peak], binned_data[peak]), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)
    
    ax2.set_xlabel('Time (μs)', fontdict={'family': 'STIXGeneral', 'size': 12})
    ax2.set_ylabel('Counts', fontdict={'family': 'STIXGeneral', 'size': 12})
    ax2.set_yscale('log')
    ax2.set_ylim(70, 2500)  # Y-limits from original plot
    ax2.grid(False)
    
    # Add a horizontal line showing the background threshold
    ax2.axhline(y=background_level, color='blue', linestyle=':', linewidth=1, 
               label=f'Background threshold ({background_level:.2f})')
    
    # Set x limits for both plots
    plt.xlim(0, 89)
    
    # Add legend at the bottom
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=3, frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for the legend
    plt.savefig('peak_regions.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    # Now create a summary table of all peak data
    print("\nPeak Summary Table:")
    print("=" * 70)
    print(f"{'Peak #':<6}{'Position (μs)':<15}{'Region Start (μs)':<20}{'Region End (μs)':<20}{'Total Counts':<15}")
    print("-" * 70)
    for i, (peak, (left_idx, right_idx), count) in enumerate(zip(top_peaks, peak_regions, peak_counts)):
        print(f"{i+1:<6}{binned_hist_index[peak]:<15.2f}{binned_hist_index[left_idx]:<20.2f}{binned_hist_index[right_idx]:<20.2f}{count:<15.0f}")
    print("=" * 70)
    print(f"Total counts from all {num_peaks} peaks: {total_counts:.0f}")
    
    # Save the peak data to a CSV file
    peak_data = pd.DataFrame({
        'Peak Number': range(1, len(top_peaks) + 1),
        'Position (μs)': binned_hist_index[top_peaks],
        'Region Start (μs)': [binned_hist_index[left_idx] for left_idx, _ in peak_regions],
        'Region End (μs)': [binned_hist_index[right_idx] for _, right_idx in peak_regions],
        'Total Counts': peak_counts
    })
    
    peak_data.to_csv('peak_summary.csv', index=False)
    print("Peak data saved to 'peak_summary.csv'")
    
except Exception as e:
    print(f"Error processing data: {e}")
    import traceback
    traceback.print_exc() 
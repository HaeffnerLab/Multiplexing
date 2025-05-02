import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator

# Load the CSV file
file_path = "/Users/QUBIT/Documents/GitHub/Untitled/Multiplexing/ion-shuttling/RigolDS4.csv"
data = pd.read_csv(file_path)

# Applying a rolling average for smoothing
window_size = 10
smoothed_data = data.rolling(window=window_size).mean()

# For this example, we'll create the binned data from the smoothed data
# Ensure these match your original data processing
binned_data = np.array(smoothed_data['CH1(V)'].dropna())
binned_hist_index = np.array(smoothed_data['Time(s)'].dropna() * 1e6 * 0.96 + 39.9)

# Find peaks with appropriate parameters
peaks, properties = find_peaks(binned_data, prominence=1, distance=10)  # Adjust as needed

# Select the top 9 peaks based on their prominences
top_peaks_indices = np.argsort(properties['prominences'])[-9:]
top_peaks = peaks[top_peaks_indices]
top_peaks = np.sort(top_peaks)  # Sort peaks by position, not prominence

# Set a threshold above background for peak region detection
# Use a percentile-based approach for robust background estimation
background_level = np.percentile(binned_data, 50) * 1.5  # 50th percentile × 1.5
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
    if right_idx - left_idx < 5:
        pad = (5 - (right_idx - left_idx)) // 2
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
print(f"Total counts across all 9 peaks: {total_counts:.0f}")

# Create a plot with the original data and highlighted peak regions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# Plot the original data on the first subplot (similar to your original plot)
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
ax2.set_ylim(70, 2500)  # Same y-limits as in your original plot
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
print(f"Total counts from all 9 peaks: {total_counts:.0f}") 
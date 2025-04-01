import numpy as np
import matplotlib.pyplot as plt
import os

def process_g2_data(i):
    """
    Process g2 data for a given index i by:
    1. Loading left and right ion data
    2. Summing them
    3. Subtracting center ion data
    4. Saving the result
    """
    # Load the data files
    left_data = np.load(f'f{i}_final_result_left_ion_data_aligned.npy')
    right_data = np.load(f'f{i}_final_result_right_ion_data_aligned.npy')
    # center_data = np.load(f'g2_data/{i}_final_result_center_ion_data.npy')
    center_file_path = os.path.join('..', 'g2_data', f'{i}_final_result_center_ion_data.npy')
    center_data = np.load(center_file_path)
    # index_data = np.load(f'g2_data/{i}_index_center_ion_data.npy')
    index_file_path = os.path.join('..', 'g2_data', f'{i}_index_center_ion_data.npy')
    index_data = np.load(index_file_path)
    
    # Sum left and right data
    combined_data = left_data + right_data
    
    # Subtract center data
    final_data = combined_data - center_data
    
    # Save the processed data
    np.save(f'{i}_final_result_caled_ion_data.npy', final_data)
    print(f'Processed and saved data for index {i}')

def load_and_plot_caled_data():
    """
    Load the 7th calibrated data file and plot it.
    """
    try:
        # Load calibrated data for index 7
        i = 7
        caled_data = np.load(f'{i}_final_result_caled_ion_data.npy')
        # Load corresponding index data
        index_file_path = os.path.join('..', 'g2_data', f'{i}_index_center_ion_data.npy')
        index_data = np.load(index_file_path)
        
        print(f"Loaded data for index {i}, shape: {caled_data.shape}")
        
        # Plot 7th dataset
        plt.figure(figsize=(10, 6))
        plt.plot(index_data, caled_data)
        plt.title(f'Calibrated Data for Index {i}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.savefig(f'caled_data_index_{i}.png')
        plt.show()
        
    except Exception as e:
        print(f"Error processing index {i}: {e}")

def main():
    """
    Main function to process and plot only the 7th g2 data.
    """
    # Process only the 7th dataset
    i = 7
    print(f"Processing g2 data for index {i}...")
    try:
        process_g2_data(i)
    except Exception as e:
        print(f'Error processing index {i}: {str(e)}')
    
    # Then load and plot the calibrated data
    print(f"\nLoading and plotting calibrated data for index {i}...")
    load_and_plot_caled_data()

if __name__ == '__main__':
    main() 
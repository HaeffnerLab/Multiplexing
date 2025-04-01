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
    left_data = np.load(f'{i}_final_result_left_ion_data_aligned.npy')
    right_data = np.load(f'{i}_final_result_right_ion_data_aligned.npy')
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
    np.save(f'{i}_final_result_caled_ion_data_aligned.npy', final_data)
    print(f'Processed and saved data for index {i}')

def load_and_plot_caled_data():
    """
    Load all calibrated data files, plot individual datasets,
    sum all datasets, and plot the summed data.
    """
    # Initialize an array to store summed data
    summed_data = None
    
    # Store all datasets and indices for combined plot
    all_datasets = []
    index_data_ref = None
    
    # Load all 9 files and sum them
    for i in range(1, 10):
        try:
            # Load calibrated data
            caled_data = np.load(f'{i}_final_result_caled_ion_data_aligned.npy')
            # Load corresponding index data
            # index_data = np.load(f'{i}_index_center_ion_data.npy')
            index_file_path = os.path.join('..', 'g2_data', f'{i}_index_center_ion_data.npy')
            index_data = np.load(index_file_path)
            
            print(f"Loaded data for index {i}, shape: {caled_data.shape}")
            
            # Store dataset for combined plot
            all_datasets.append((i, caled_data))
            if index_data_ref is None:
                index_data_ref = index_data
            
            # Initialize summed_data with the shape of the first loaded data
            if summed_data is None:
                summed_data = caled_data
            else:
                # Sum the data
                summed_data += caled_data
                
            # Plot individual data if needed
            plt.figure(figsize=(10, 6))
            plt.plot(index_data, caled_data)
            plt.title(f'Calibrated Data for Index {i}')
            plt.xlabel('Time (ns)')
            plt.ylabel('Counts')
            plt.grid(True)
            plt.savefig(f'caled_data_index_{i}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error processing index {i}: {e}")
    
    # Plot all datasets in the same figure with different colors
    if all_datasets:
        plt.figure(figsize=(12, 8))
        for i, data in all_datasets:
            plt.plot(index_data_ref, data, label=f'Index {i}')
        plt.title('All Calibrated Datasets Comparison')
        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.legend()
        plt.savefig('all_caled_datasets.png')
        plt.close()
        print("All datasets plotted together and saved successfully.")
    
    # Plot summed data
    if summed_data is not None:
        plt.figure(figsize=(12, 8))
        plt.plot(index_data, summed_data)
        plt.title('Summed Calibrated Data (All Indices)')
        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.savefig('summed_caled_data.png')
        plt.show()
        print("Summed data plotted and saved successfully.")
    else:
        print("No data was loaded.")

def main():
    """
    Main function to process and plot the g2 data.
    """
    # First process all 9 datasets
    print("Step 1: Processing g2 data for all indices...")
    for i in range(1, 10):
        try:
            process_g2_data(i)
        except Exception as e:
            print(f'Error processing index {i}: {str(e)}')
    
    # Then load and plot the calibrated data
    print("\nStep 2: Loading and plotting calibrated data...")
    load_and_plot_caled_data()

if __name__ == '__main__':
    main() 
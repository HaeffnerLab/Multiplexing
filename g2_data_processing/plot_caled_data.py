import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_caled_data():
    # Initialize an array to store summed data
    summed_data = None
    
    # Load all 9 files and sum them
    for i in range(1, 10):
        try:
            # Load calibrated data
            caled_data = np.load(f'{i}_final_result_caled_ion_data.npy')
            # Load corresponding index data
            index_data = np.load(f'{i}_index_center_ion_data.npy')
            
            print(f"Loaded data for index {i}, shape: {caled_data.shape}")
            
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

if __name__ == "__main__":
    load_and_plot_caled_data() 
import numpy as np
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
    left_data = np.load(f'{i}_final_result_left_ion_data.npy')
    right_data = np.load(f'{i}_final_result_right_ion_data.npy')
    center_data = np.load(f'{i}_final_result_center_ion_data.npy')
    index_data = np.load(f'{i}_index_center_ion_data.npy')
    
    # Sum left and right data
    combined_data = left_data + right_data
    
    # Subtract center data
    final_data = combined_data - center_data
    
    # Save the processed data
    np.save(f'{i}_final_result_caled_ion_data.npy', final_data)
    print(f'Processed and saved data for index {i}')

def main():
    # Process all 9 datasets
    for i in range(1, 10):
        try:
            process_g2_data(i)
        except Exception as e:
            print(f'Error processing index {i}: {str(e)}')

if __name__ == '__main__':
    main()

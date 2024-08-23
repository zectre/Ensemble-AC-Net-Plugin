import numpy as np

def split_and_save_data(input_path, output_paths):
    # Load the .npy file
    data = np.load(input_path)

    # Split the data into four parts
    data1 = data[:, 0:8, :, :]
    data2 = data[:, 8:11, :, :]
    data3 = data[:, 11, :, :]
    data4 = data[:, 12, :, :]

    # Reshape the data
    data1 = data1.reshape(data1.shape[0], 3, 3, data1.shape[1], 1)
    data2 = data2.reshape(data2.shape[0], -1).astype('float32')
    data3 = data3.reshape(data3.shape[0], -1).astype('float32')
    data4 = data4.reshape(data4.shape[0], -1).astype('float32')
    
    # Save the split data to .npy files
    np.save(output_paths[0], data1)
    np.save(output_paths[1], data2)
    np.save(output_paths[2], data3)
    np.save(output_paths[3], data4)

#wew
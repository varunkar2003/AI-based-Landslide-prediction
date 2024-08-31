import h5py

# Open the HDF5 file
with h5py.File('C:/Users/Varun/Desktop/AI-based-Landslide-prediction/dataset/h5 file/model.h5', 'r') as file:
    # Access the 'dense_4' group
    group = file['dense_4']['dense_4']
    
    # Access and print the 'bias:0' dataset
    bias = group['bias:0'][:]
    print("Bias:")
    print(bias)
    
    # Access and print the 'kernel:0' dataset
    kernel = group['kernel:0'][:]
    print("\nKernel:")
    print(kernel)

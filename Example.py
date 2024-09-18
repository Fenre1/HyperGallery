import numpy as np
from hdf5_saver import HDF5Saver


# Initialize the saver with your HDF5 file name
saver = HDF5Saver('your_data.h5')

# Assume you have the following data
file_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
allcat_bin = np.array([0, 1, 0])
clustering_results = np.array([1, 2, 1])
features = np.random.rand(3, 128)  # Example feature vectors

# Save data to HDF5
saver.save_to_hdf5(file_list, allcat_bin, clustering_results, features)

# Create and save thumbnails without saving JPEG files to disk
saver.create_and_save_thumbnails(file_list)

# If you want to save the JPEG files for debugging
saver.create_and_save_thumbnails(file_list, save_jpg_files=True)

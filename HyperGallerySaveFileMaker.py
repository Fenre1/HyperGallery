import numpy as np
import h5py
from PIL import Image, ImageOps
import os

class HDF5Saver:
    def __init__(self, file_name):
        self.file_name = file_name

    def save_to_hdf5(self, file_list, allcat_bin, clustering_results, features):
        with h5py.File(self.file_name, 'a') as f:
            # Store file_list as variable-length strings
            dt = h5py.string_dtype(encoding='utf-8')
            if 'file_list' in f:
                del f['file_list']
            f.create_dataset('file_list', data=file_list, dtype=dt)

            # Store allcat_bin as an integer array
            if 'allcat_bin' in f:
                del f['allcat_bin']
            f.create_dataset('allcat_bin', data=allcat_bin, dtype='i8')

            if 'clustering_results' in f:
                del f['clustering_results']
            f.create_dataset('clustering_results', data=clustering_results, dtype='i8')

            # Store features with chunking for efficient access
            if 'features' in f:
                del f['features']
            chunk_size = (min(1000, features.shape[0]), features.shape[1])  # example chunk size
            f.create_dataset('features', data=features, dtype='f4', chunks=chunk_size)

    def create_and_save_thumbnails(self, file_list, thumbnail_size=(100, 100), output_dir='assets/thumbnails', save_jpg_files=False):
        if save_jpg_files:
            os.makedirs(output_dir, exist_ok=True)

        with h5py.File(self.file_name, 'a') as hdf:
            # Create or extend the 'thumbnails' dataset for paths
            dt = h5py.string_dtype(encoding='utf-8')
            if 'thumbnails' in hdf:
                del hdf['thumbnails']
            hdf.create_dataset('thumbnails', (len(file_list),), dtype=dt)

            # Create the 'thumbnail_images' dataset for images
            if 'thumbnail_images' in hdf:
                del hdf['thumbnail_images']
            hdf.create_dataset(
                'thumbnail_images',
                shape=(len(file_list),) + thumbnail_size + (3,),
                dtype='uint8',
                chunks=(1,) + thumbnail_size + (3,),
                maxshape=(None,) + thumbnail_size + (3,)
            )

            for i, file_path in enumerate(file_list):
                output_path = os.path.join(output_dir, f"{i}.jpg")
                new_image = self.process_image(file_path, thumbnail_size)

                if save_jpg_files:
                    # Save the processed image as JPEG
                    new_image.save(output_path, 'JPEG')
                    relative_path = 'thumbnails/' + output_path.split('thumbnails/')[1].replace('\\', '/')
                    hdf['thumbnails'][i] = relative_path  # Save the relative path
                else:
                    hdf['thumbnails'][i] = ''  # Save an empty string or a placeholder

                # Convert new_image to numpy array and store in 'thumbnail_images'
                img_array = np.array(new_image)
                hdf['thumbnail_images'][i] = img_array

    def process_image(self, input_path, target_size=(100, 100)):
        # Open the image
        image = Image.open(input_path).convert('RGB')
        # Resize while maintaining aspect ratio
        image.thumbnail(target_size, Image.ANTIALIAS)

        # Calculate padding to make the image square
        delta_w = target_size[0] - image.size[0]
        delta_h = target_size[1] - image.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

        # Add padding
        new_image = ImageOps.expand(image, padding, fill='black')

        return new_image  # Return the processed image for storage

loading_data.py
- Takes a csv file and loads all fields into a PostGRES database, and saves the raw images into the correct label directory.

standardize_size.py
- Resizes and standardizes the size of all the images (i.e. size = (100, 100)) and saves them in the same directory layout as the
raw images.

image_loader.py
    1. Creates a master list with all the image files names
    2. Reads in standardized images and converts them into numpy arrays
    3. Segments images and saves image segments

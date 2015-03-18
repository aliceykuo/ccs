#My Wedding Style: A  Wedding Image Classifier


##Part I: Preprocessing

#####loading_data.py
* This takes a csv file and loads all fields into a PostGRES database, and saves the raw images into the correct label directory.


#####standardize_size.py
* Resizes and standardizes the size of all the images (i.e. size = (100, 100)) and saves them in the same directory layout as the
raw images.

#####image_loader.py
* The ImageLoader class does the following:
    1. Creates a master list with all the image files names
    2. Reads in standardized images and converts them into numpy arrays
    3. Segments images and saves image segments


##Part II: Exploratory Data Analysis

#####dom_colors.py
* The class DominantColor uses kmeans to extract the most dominant colors
    from an image. There are two variations of how this file can run:
        1.) One image dominant colors extraction
        2.) Segmented images dominant colors extraction

#####extract_patches.py
* This class ExtractPatches breaks up an image into smaller patches determined by
    each patch's size and the number of patches desired.

#####filters.py
* This contains variety of image transformation filters used during the exploration phase.
    1. sift
    2. canny edge detector
    3. sobel edge detector
    4. roberts edge detector
    5. background extractor
    6. equalize historgram
    7. whiten
    8. dropout

#####helper.py
* This file contains general functions used across the board.

##Part III: Model

#####run_model.py
* This file contains all the models that were tested to determine most effective classification method.


##Part IV: Web App

#####webapp.py
* The folder webapp conatins the whole Flask webapp that allows users to upload an image and obtain a "read" on the wedding style.

#####Files in the folder:
    1. tempaltes: all the HTML files
    2. static: all CSS and JS libraries
    3. images: all the images used in the templates


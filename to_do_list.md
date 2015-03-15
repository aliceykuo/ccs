##OVERVIEW

small images 
- beach 
- rustic 
- ballroom/glam
- vintage 

larger images 
- beach 
- rustic 
- ballroom/glam
- vintage 

DATA PIPELINE 
- code to grab image url and drop into data directory   
- code to preprocess image 
	- standardize size 
	- convert to matrix 
	- flatten image 

FEATURE ENGINEERING 
- rotate image (90, 180, 270)
- dominant colors (kmeans )
- dropout method - multiply matrix 
- gridsplit - split image up into minichunks 


WEB APP 
- scrape pinterest board in real-time
https://github.com/Nateliason/pin-scrape/blob/master/main.py


binomial sampling of a vector equal to the lengh of my feature matrix 
multiply the bino sample with feature matrix 
some values 0, some 1

image bag of words

dropout technique - 0.3

Web Scrape 
Beach:
	individual_board_beach_usethis
	pin_meta_full_beach
Rustic:
	individual_board_rustic_usethis
	pin_meta_full_rustic
Ballroom: 
	individual_board_ballroom_usethis
	pin_meta_full_ballroom
Vintage:
	individual_board_vintage_usethis
	pin_meta_full_vintage

Daily status:
3/2 - rescraped (only using pin_meta data) - both 100 boards and 500 board versions 
- wrote initial code for loading data in postgres 
3/3 - rework code - load into postgres & save to img directory




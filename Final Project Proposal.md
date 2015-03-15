

#### Summary:
One of the first places recently many engaged women go to start planning their wedding is Pinterest. It's easy to start gathering inspiration for dress, hairstyles, flower, decor, and themes. Pinterest makes it so effortless to discover pins, but it's not always obvious what trends are emerging from the pins. The goal of this project is to prototype a method to classify people's pinterest boards into a Wedding Theme/Style based on what they have pinned. Are you trending towards a beach wedding or a more rustic style? 

#### Motivation:
This idea was inspired by one of my closest friends who recently got engaged. It occured to me that many women spend a great deal of time getting wedding inspiration from Pinterest. While Pinterest is a great resource for gathering inspiration, the patterns that emerge from Pinterest Boards are not always immediately obvious. I thought that it would be useful to have an app that can help identify trends/patterns of pinterest boards. One of my objectives for this project was to explore classifying images. The project may have started out with a specific domain in mind, but the application of image classification is extremely broad and relevant. 


#### Deliverables:

Results will be presented on a webpage, but more ideal would be a demo. 

#### Data Sources:
Pinterest Boards (both images and tags), Google images

Examples: 

* [Pinterest Board - Beach Wedding ](https://www.pinterest.com/search/boards/?q=wedding%20beach")

* [Pinterest Board - Rustic Wedding ](https://www.pinterest.com/search/boards/?q=wedding%20rustic")

* [Google Images - Beach Wedding ](https://www.google.com/search?q=beach+wedding&espv=2&biw=1213&bih=747&source=lnms&tbm=isch&sa=X&ei=umHtVOOkLYbtoASepoGIBw&ved=0CAYQ_AUoAQ&dpr=0.9")
* [Google Images - Rustic Wedding ](https://www.google.com/search?q=beach+wedding&espv=2&biw=1213&bih=747&source=lnms&tbm=isch&sa=X&ei=umHtVOOkLYbtoASepoGIBw&ved=0CAYQ_AUoAQ&dpr=0.9#tbm=isch&q=rustic+wedding") 

Status:

 - Beach label: Scraped 4000+ images
 - Rustic label: Scraped 1000+ images 


#### Process:

###### Determine Labels

1. Beach (related keywords: beach, destination wedding)
2. Rustic (related keywords: rustic, farm, ranbarn)
3. Glam/Modern - determine if distinct enough
4. Classic? Vintage - determine if distinct enough

###### Data Procurement

* Scrape Pinterest boards for images and tags (keywords are labels: beach, rustic, etc) Goal is to get at least 1000 images for each label. These boards include all sorts of images (objects, locations, dresses, flowers, etc), so may need to consider filtering out irrelevant images. This may not be a huge issue with enough images to train on. 
* Do the same with Google images 
* Explore other popular wedding sites (The Knot)


###### Image Preprocessing

1. Downsize and standardize all image sizes (200 x 150). Determine what size is most desirable (and practical). 
2. Convert each image into a numpy matrix of RBG pixels. Explore converting to HSV space as well. 
3. Flatten image into 1D array and scale with StandardScaler
4. Extract the most dominant colors from each image (can use kmeans for n top colors). Will need to determine how this changes when filters are applied. I will be exploring opencv and skimage. I will be using these dominant colors as part of my features. 

###### Pinterest Tags

Tags have also be scraped along with the images, and can used as additional features. Use TFIDF to extract the most common words.  

###### Features & Classifier

Current feature set is very large (assuming I have the flattend image vector, dominant colors, most common words). Are there other features that I can extract from using filters? IMB's Visual Recognition app could potentially be used to determine what kind of signal I can get from these images. 
Consider PCA to reduce the number of dimensions. For the classifier, I'd start with multinomial logistic regression. Also explore other models (multiclass SVM, Naive Bayes). Train on 80% of data and make predictions on the holdout set. 
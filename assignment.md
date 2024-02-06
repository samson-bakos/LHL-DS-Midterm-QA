## Part 1: EDA

Welcome to the new midterm project, it's been put together just for you!  You'll find a data folder with quite a few JSON files.  In each of these files you'll find data for housing sales in the US.  To wrangle the data, try iterating over each file, loading it, and storing it in memory.

Your tasks are as follows:
1. Load the data from the provided files (in the `data/` directory)
2. 
3. For each bike station, use the API to call the latitude, longitude and number of bikes. 
4. Parse the JSON object into a Pandas dataframe. 

Complete the **1 - EDA.ipynb** notebook to demonstrate how you executed the tasks above. 

## Part 2: Connecting to Foursquare and Yelp APIs

Your tasks are as follows:
1. Connect to the  [Foursquare](https://developer.foursquare.com/places) API
2. Connect to the [Yelp](https://www.yelp.com/developers/documentation/v3/get_started) API. This API offers similar services as Foursquare.
3. For each of the bike stations in Part 1, query both APIs to retrieve information for the following in that location:
 - Restaurants or bars
 - Various POIs (points of interest) of your choice
4. Create a DataFrame for the Yelp results and Foursquare results. 
5. Compare the quality of the Yelp and Foursquare API. For your location, which API gives you the most complete information/better coverage? *NOTE:* Your definition of 'coverage' is up to you. It could be simple 'number of POIs in the area', but it could also be something more specific like 'number of reviews per POI', or 'number of different attributes of each POI'.


Complete the **2 - model_selection.ipynb** notebook to demonstrate how you executed the tasks above.

## Part 3: Joining Data

1. Join the data from Part 1 with the data from Part 2 to create a new dataframe. 
2. Use data visualization to explore the data. 
3. Create your own SQLite database and store the data you've collected on the POIs. **Put some thought into the structure of your database.** We've used and created sqlite3 databases before in the activity [**SQL in Python**](https://data.compass.lighthouselabs.ca/b9e08cd5-68c6-490c-a32b-a66f01bf53e1).
Validate your data.

Complete the **3 - tuning_pipeline.ipynb** notebook to demonstrate how you executed the tasks above.
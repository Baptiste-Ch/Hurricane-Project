Thanks to the ‘Numerical Project’ course of the Master SOAC (Lyon, France), our goal aimed to better understanding the utilization of statistical tools to predict extreme weathers.
In this particular case, we focused on Atlantic hurricanes which were identified by machine learning tools.

The structure of that project is decomposed into two distinct parts :
- Creation_df.py : Thanks to IBTrACS database, hurricane historical events were compiled. 
  Their information of latitude/longitude/time were matched with ERA5 database of physical features to get a complete dataframe of hurricane records since 1979.
 
- Detection : This folder contains some model optimizations which ended to the laste model to apply with the right hyperparameters

- Optimization : That script defines a function that run the modelisation for any dataframe.

- API.py : In a second part, the most robust machine learning model was found to get a relation between physical features and hurricane occurrences. 
In order to observe the results (beside the numerical error on the test sets), maps of different climatic conditions where built.
As a result, it has been shown that our model is able to easily identify massive hurricanes (e.g. Katrina, 2005) and medium hurricane/tempest storms (e.g. 2008 early September) without identifying any false occurrence.
 
Further improvements may still be made and this project could be extended in the next months. The next idea should be at first using this model on meteorological data in order to predict these extreme events. However, this needs an access to homogeneous features between meteorological data and ERA5 database.
Finally, this project  was a perfect opportunity to get solid abilities of python programming and machine learning utilization in climate sciences.

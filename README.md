# Zillow Clustering and Regression Project: What is driving the errors in Zestimates?
- By Jeff Akins

## Project Goal: To determine the drivers of error in the Zestimates
### Additional Goals:
- Data Acquisition: Data is collected from the codeup cloud database with an appropriate SQL query
- Data Prep: Column data types are appropriate for the data they contain
- Data Prep: Missing values are investigated and handled
- Data Prep: Outliers are investigated and handled
- Exploration: the interaction between independent variables and the target variable is explored using visualization and statistical testing
- Exploration: Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
- Modeling: At least 4 different models are created and their performance is compared. One model is the distinct - combination of algorithm, hyperparameters, and features.
- Best practices on data splitting are followed
- The final notebook has a good title and the documentation within is sufficiently explanatory and of high quality
- Decisions and judgment calls are made and explained/documented
- All python code is of high quality

## Executive Summary
Exploration of Zillow's data did not provide obvious drivers of Zestimate error. By clustering around the log error itself you can determine the top drivers for each cluster; however, creating linear regression models based on those features have not produced improved results thus far over a baseline prediction of logerror. I believe that there may be potential in clustering by location, but the fidelity would need to be at the neighborhood level to produce reasonable results; so far, I have not been able to zoom in to that level. Below is more details on some of the exploration and modeling results.

### Data dictionary
|Index | Column Name | Description | Row Count | Dtype|
|---|---|---|---|---| 
| 0 |  bathroomcnt                  | Number of Bathrooms       | 70910 | float64|
| 1 |  bedroomcnt                   | Number of Bedrooms        | 70910 | float64|
| 2 |  calculatedfinishedsquarefeet | Square feet               | 70910 | float64|
| 3 |  fips                         | County Identifier Code    | 70910 | float64|
| 4 |  latitude                     | Latitude Coordinate       | 70910 | float64|
| 5 |  longitude                    | Longitude Coordinate      | 70910 | float64|
| 6 |  yearbuilt                    | Year built                | 70910 | float64|
| 7 |  taxvaluedollarcnt            | Home Value                | 70910 | float64|
| 8 |  logerror                     | Zestimate Error           | 70910 | float64|
| 9 |  transactiondate              | Date Sold                 | 70910 | object |
| 10|  heatingorsystemdesc          | Type of AC/heat           | 70910 | object |
| 11|  county                       | County Name               | 70910 | object |
| 12|  abs_logerr                   | Absolute Log Error        | 70910 | float64|

## Project Specifications

### Plan:
**Trello Board:**
https://trello.com/b/NO8GyMgY/zillow-clustering-project

### Acquire:
- The data was acquired form Codeup's MySQL server
- Additional details on data acquisition can be found in the **wrangle_zillow.py** file included in this repo
### Data Prep:
- Only single unit / single family homes were included
- Homes with a value over $5M were excluded
- Homes over 8000 sqft were excluded
- Here is a list of columns that were dropped due to a high number of nulls or due to information that was redundant or irrelevant to determining drivers of logerror:
    - parcelid
    - calculatedbathnbr
    - finishedsquarefeet12
    - fullbathcnt
    - heatingorsystemtypeid
    - propertycountylandusecode
    - propertylandusetypeid
    - propertyzoningdesc
    - censustractandblock
    - propertylandusedesc
    - buildingqualitytypeid
    - rawcensustractandblock
    - regionidcity
    - regionidcounty
    - regionidzip
    - unitcnt
    - assessmentyear
    - lotsizesquarefeet
    - roomcnt
    - structuretaxvaluedollarcnt
    - landtaxvaluedollarcnt
    - taxamount
- The data was then split into train, validate, and test
### Explore:
- All exploration was conducted on train
- Visual exploration via histograms and pairplots were conducted
- Scatterplots and statistical tests were conducted using logerror vs the features used for modeling
### Model & Evaluate:
- Initial linear regression was conducted using various models using (Zestimate) logerror as the target vs the following features:
    - bathroomcnt
    - bedroomcnt 
    - calculatedfinishedsquarefeet                           
    - latitude                      
    - longitude                     
    - yearbuilt                     
    - taxvaluedollarcnt 
- Clustering was conducted around location, home value, home size, and logerror
- logerror clustering features were used to determine the top drivers of error for each cluster
## Conclusion:
- Using the clustered dataframe subsets of train, based on legerror bands, did not improve the logerror prediction of validate over simply passing all of the train dataframe to the regression models. More exploration is needed.

## How to Reproduce:
- Use your own env file for login information for the Codeup SQL server
- Git Pull the repo or clone the following files into a local folder: 
    - zillow_clustering_project.ipynb
    - explore.py
    - explore_evaluate.py
    - wrangle_zillow.py
    - zillow_models.py
- Open zillow_clustering_project.ipynb in a jupyter notebook and explore!
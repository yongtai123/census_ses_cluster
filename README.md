# Census Tract Level Socioecnomics Status (SES) Data Clustering

We introduce a clustering approach for census tract-level socioeconomic data de-identification. The basic premise of the method is to group census tracts with similar SES together, such that every census tract group covers at least 20,000 inhabitants. We achieved this by first clustering all the tracts into small groups, then iteratively merging groups that fail to satisfy the privacy requirements to their most similar group until all groups meet the requirements.


### Requirements
pandas 
numpy 
sklearn 
scipy

### Usage
Please see KMeans_SES_clustering.py for details.

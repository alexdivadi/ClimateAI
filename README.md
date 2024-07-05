# ClimateAI
### LiveAI HIVE 2020

This project is a Solar Panel Anomaly AI Classification app made for LiveAI's HIVE program.
The image data used to train the CNN in this model can be found here: https://github.com/RaptorMaps/InfraredSolarModules


The model is a multiclass DNN classifier. It achieved an AUC score of about 0.915.

## Set Up
This app originally was hosted on Heroku. You can run the app as a local Flask server.


After setting up a virtualenv, navigate to the `src` directory and run:
```python
flask run
```


## Schedule
| Task | Description | Date |
|:------:|:-------:|:------:|
|Create Data Directory|Acquire data and create directory|11/27/20|
|Develop Tentative Schedule|Plan work pipeline into schedule|11/30/20|
|Data Exploration|Understand what kind of data is provided|11/30/20|
|Data Preprocessing|Configure data to be used by model|12/01/20|
|Modeling|Begin designing and coding model|12/02/20|
|Training|Train model through dataset and evaluate results|12/04/20|
|Tuning|Fine-tune parameters of model and retrain it|12/05/20|
|Development|Begin constructing an app to display results|12/06/20|
|Deployment|Push final product to AWS|12/07/20|


<a href="https://www.kaggle.com/c/allstate-claims-severity" target="_blank"><img src="https://www.allstatenewsroom.com/wp-content/uploads/2015/12/Allstate_Logo4.jpeg" width="250"></a>

###These are my scripts used for the Allstate Challenge on "How severe is an insurance claim?"

This is my first (serious) attempt at Kaggle and i manage to get 46/3055 in the private leaderboard with [this submission](./allstate1117.71816974.csv). You can find my Kaggle profile [here](https://www.kaggle.com/datajanitor).

There is a seperate blog post avaliable [here](http://freedom89.github.io/Allstate_kaggle), which talks about what i have learnt and some pitfalls. 

The order of folders are as follows:  

* [data_prep](./data_prep)
* [xgboost](./xgboost)
* [keras](./keras)
* [ef\_rf](./ef_rf)
* [second\_level\_models](./second_level_models)

Hyperopt Scripts and Results are also included [here](./hyperopt_results). 

***

### Data_prep

* Fork this Repo and make a directory named input.
* Download the train & test set from [here](https://www.kaggle.com/c/allstate-claims-severity/data) and unzip them. 
* Run the scripts `power2.py` , `power3.py` and `fourm_1106_prep.py`
* You should have 3 additional files for each script. 

***
### Xgboost

* The 3 scripts to generate the xgboost out of bag and predictions are 

	* [power2.ipynb](./xgboost/power2.ipynb)
	* [power3\_xgb\_final.ipynb](./xgboost/power3_xgb_final.ipynb)
	* [1106\_fourm.ipynb](./xgboost/1106_fourm.ipynb) 

The outputs and params are available in the notebook. Note that not all outputs are used in the end for second level modelling. Refer to [Second level modelling](#stacking). 
	
***

### Keras (Neural Networks)

The motivation is from this [script](https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364/comments) which is in this github repo found [here](./keras/keras.ipynb).

I used 4 different sets of parameters.

* The one found in the forum.
* Same Model with a different seed.
* By changing the second layer from 200 to 250 nodes.
* Training the same model without using log transform. 

The outputs can be found in the `csv's` in the keras folder. 

*** 
### Extra Trees And Random Forest 

* Everything is [here](./ef_rf/rf_ef.ipynb)

***

### Weighted Average

* Everything is [here](./fmin_first_level_models.ipynb)
* Submission file `allstate1118.34514548.csv ` which would be ranked 102 in the private leaderboard (top 5%).

***
### <a name="stacking"></a> Second Level Modelling (Stacking) 

* Combine all first level models into one data frame with [combine_data.ipynb](./second_level_models/combine_data.ipynb) which outputs the [training set](./second_level_models/train_second_level_model.csv) and [test set](./second_level_models/test_second_level_model.csv).
* Run a single 5-fold with early stopping on the out of bag data set with [keras\_stacking\_single\_fold.ipynb](./second_level_models/keras_stacking_single_fold.ipynb). 
	* You may use the pre-trained models included in that folder to skip the training. 
* Bagged the above model 5 times with [keras\_stacking\_bagged.ipynb](./second_level_models/keras_stacking_bagged.ipynb).
* Take the second level model output and feed it back with the first level models to find the optimum weights with [fmin\_second\_level.ipynb](./fmin_second_level.ipynb).
* Final Submission `allstate1117.71816974.csv`





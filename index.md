#Getting into Top 2% on Kaggle!

This post records my experience in Kaggle's *[Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity)* competition. It has three parts—feel free to skip to whichever you're interested in!

* [Self-Introduction](#intro)
* [Candid Reflections on the Competition](#reflect)
* [Things I have learnt](#start)

***

<a name="intro"></a>

### Self-Introduction

Hello! I am Yi Xiang, currently employed as a Junior Data Scientist. Feel free to drop me a message if you think I can improve in any way! You can reach me at my [LinkedIn](https://www.linkedin.com/in/yi-xiang-low-b349137b).

Alternatively, you could also leave your comments on this github's issue page [here](https://github.com/Freedom89/Allstate_kaggle/issues).

In the many meet-ups I have attended, a common question that aspiring data scientists ask employers is:

	What do you look for in a potential hire? 

Typical answers would include:
	
	* Communication or storytelling skills
	* Coding proficiency
	* Learning ability
	* Passion
	* And the lists goes on...

Upon hearing this, the follow-up question usually most likely be:

	 How should I demonstrate this to my potential employer?

And the advice given is usually to: 

	* Increase online presence (e.g. through github, blogs)
	* Find interesting projects to work on

The truth is, I had procrastinated on fulfilling the above two points, and it was time I did something about it. Kaggle seemed like a good fit for the above two objectives. That was how I joined the Allstate Claims Severity competition on 2nd nov, which lasted from Oct to Dec 2016.  

***

<a name="reflect"></a>

### Candid Reflections on the Competition

Among those familiar with this field, [xgboost](http://xgboost.readthedocs.io/) (XGB) comes to mind as a popular approach to **ANY** (structured) machine learning problem. 

Hence, I started the competition with an initial goal of learning how to tune XGB, and this was how my goals evolved during the competition over a single month: 

1. Squeezed into the top 10% by tuning XGB, thus decided an attempt to achieve a bronze medal (top 10%). 
2. Within no time at all, I was kicked out of the top 10%. In order to climb back up, I had to ensemble different models. Most people in the forums had recommended ensembling neural nets with XGB. Unfortunately, apart from learning about it Coursera, I had no experience with neural nets!
3. Neural nets are **very** slow on CPU. To speed things up, I learnt to set up CUDA on an AWS GPU-compute series, install python and Theano, transfer data, and to configure a Juypter notebook! 
4. With my neural net and XGB, a simple average got me into the top 5% - great! Let's attempt to win a sliver medal instead (top 5%).
5. Unfortunately, within no time at all, I was almost kicked out (again) of the top 5%, and there was a risk I might drop further down the private leaderboard due to overfitting.
6. Time for stacking! I understood the concept, but I have never done it before. Sadly, I did not extract my out-of-bag predictions from previous models (a painful but important lesson). I tried XGB and ridge regression for a second level modelling, which yielded lousy results. This was when I nearly hit a roadblock and thought I might have to settle for being in the top 10%.
7. Just then, someone posted about using neural nets as a second level model close to the last day. In a last burst of fire, I decided to give it a shot—and it worked! I was ranked 78th on the public leaderboard and 46th on the private leaderboard, which was really a surprise! 

The point i am trying to make is that:

	Kaggle is really a good place to start with lots of helpful people sharing.


### Therefore, my objective for publishing this post is to:  

1. Share about my experience, learnings! 
2. Document my code! 
3. Encourage aspiring data scientists to start! 

***

<a name="start"></a>

<a href="https://www.kaggle.com/c/allstate-claims-severity" target="_blank"><img src="https://www.allstatenewsroom.com/wp-content/uploads/2015/12/Allstate_Logo4.jpeg" width="400"></a>

#Aim of Competition

The aim of this competition was to create an algorithm to predict the severity of insurance claims. Evaluation metric used was the *[Mean Absolute Error (MAE)](https://www.kaggle.com/wiki/MeanAbsoluteError)*. 

If you wish to reproduce my results, refer to the `README.md` [here](https://github.com/Freedom89/Allstate_kaggle). 

## Contents

1. [Custom Objectives](#custom)
2. [Finding Interactions, Encoding, Boxcox](#xgbfir)
3. [Tuning XGB](#Tune)
4. [Neural Networks](#NN)
5. [Ensemble Version 1](#ensemble1)
6. [Ensemble with Weighted Average](#ensemble2)
7. [Ensemble with NN](#ensemble3)
8. [Things I should/would have tried](#reflections)


####  <a name="custom"></a>Custom Objectives  

***
My (current) understanding about MSE is that it penalises error that are further away from the mean, while MAE penalises errors equally. The first thing I learnt about the [MAE](http://www.vanguardsw.com/business-forecasting-101/mean-absolute-deviation-mad-mean-absolute-error-mae/) metric was that it optimises in terms of the median value, while MSE optimises for the mean. More information  [here](http://stats.stackexchange.com/questions/147001/is-minimizing-squared-error-equivalent-to-minimizing-absolute-error-why-squared).

If you had taken undergraduate mathematics, you would know that `y = |x|` is non-differentiable at ` x = 0`. So when you configure Xgboost to use ` eval_metric = 'mae' `, the [algorithm would still descent by MSE](http://stackoverflow.com/questions/34178287/difference-between-objective-and-feval-in-xgboost), which poses a problem if you are optimising for MAE. To avoid over-penalising values further away from the mean, you could compress the range of values of your target variable, such as normalising, scaling or log-transform, but it still would not solve the problem. 


However, it turned out that numerical approximation is very useful (thank you taylor series!). This [link](http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html) (**worth reading!**) describes the intuition behind optimising for MAE. For those who did undergraduate mathematics/statistics, you would remember functions like Cauchy and huber, which happens to be solvers for MAE problems. More information [here](http://scipy-cookbook.readthedocs.io/items/robust_regression.html).

Basically,  This is done via the 'Fair' objective function. Essentially, you define an MAE objective function, but with a the gradient (first derivative) and hessian (second derivative) of the `Fair objective Function`. 

Below, you can observe how the 'Fair' objective function mimics the least-absolute function pretty accurately:


<img src="http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/img334.gif" width="400">

The objective, gradient, hessian of the above functions are defined as follows:

<img src="http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/img333.gif" width="400">

Majority of the scripts in the forums used the 'Fair' objective, coded as: 

```
def fair_obj(preds, dtrain):
    fair_constant = 2
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess
```
The smaller `fair_constant` is, the *slower* or *smoother* the loss is. 

This custom objective can then be used in `xgb.train`:

```
clf = xgb.train(params,
                d_train,
                100000,
                watchlist,
                early_stopping_rounds=early_stop,
                obj=fair_obj,
                verbose_eval = n_print,
                feval=xg_eval_mae)
```

Additional information can be found [here](https://www.kaggle.com/c/allstate-claims-severity/forums/t/24520/effect-of-mae).

#### [Back to contents](#start)
####  <a name="xgbfir"></a>Finding Interactions, Encoding, Boxcox

***

##### Finding Interactions
One of the limitations of linear regression is in identifying interactions between features. To solve this, an [XGBoost model dump parser](https://github.com/Far0n/xgbfi) was developed as a way to find N-way feature interactions that can be used to improve your XGB model, or to be used as features themselves. 

Fortunately, someone else posted [this script](https://www.kaggle.com/modkzs/allstate-claims-severity/lexical-encoding-feature-comb/discussion), which saved me a bit of time on finding N-way feature interactions. 

##### Encoding

In the raw data, features ran from `A, B, ... , Z,` to `AA, .. AZ `, which seemed to suggest some significance in how the data was ordered.

Therefore, instead of using label or one hot encoding, I experimented with an alternative function to encode these categorical features:

```
def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r
    
```
Essentially, this function recodes categories based on their rank order:

* `encode('A')   = 1 `
* `encode('Z')   = 26`
* `encode('AA')  = 27`
* `encode('AC')  = 29`

While this method could be used to complement other functions like `min/max/mean/counts/ti-idf` , I did not manage to test this. 

##### Boxcox

Some machine learning algorithms perform better when features are normally distributed. Unfortunately, figuring out how to transform each feature as such requires a huge effort.

Introducing *boxcox*, a (very) convenient way of transforming these features by measuring their [skew](https://en.wikipedia.org/wiki/Skewness).

Here are a couple of good articles explaining boxcox that I came across:

* [Fairly Layman](https://www.isixsigma.com/tools-templates/normality/making-data-normal-using-box-cox-power-transformation/)
* [Math and more Math](http://onlinestatbook.com/2/transformations/box-cox.html)—never thought year one calculus would be this useful!

Implementing boxcox in code:

```
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.25]
skewed_feats = skewed_feats.index

for feats in skewed_feats:
    train_test[feats] = train_test[feats] + 1
    train_test[feats], lam = boxcox(train_test[feats])

```

#### [Back to contents](#start)

####  <a name="Tune"></a>Tuning XGB 
***

Unless you are extremely experienced and have good intuition about which parameter values to use, it is likely that you need to learn from trial and error. 

For this, I recommend [hyperopt](https://github.com/hyperopt/hyperopt), a python library for serial and parallel optimisation over awkward search spaces. It even allows you to tweak the number of layers in a neural net! 

I have some examples in my git repo:

* [XGB](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/hyper_opt_xgb.ipynb)

  Change the data input to power3 if you want to run hyperopt for a 3-way interaction
* [Extra Trees](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/extratrees_hyper_opt.ipynb)
* [Random Forest](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/hyper_opt_random_forest.ipynb)

Results of the hyperopt can be found in [this repo](https://github.com/Freedom89/Allstate_kaggle/tree/master/hyperopt_results).

#### [Back to contents](#start)

####  <a name="NN"></a>Neural Networks
***

As mentioned, this is my first time coding a neural net outside of Coursera ([Andrew Ng's Machine Learning](https://www.coursera.org/learn/machine-learning) and [UOW's Machine Learning specialisation](https://www.coursera.org/specializations/machine-learning)).

##### Using AWS 

[This script](https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364/comments) helped me a lot in starting out with neural networks, but running it on an 8-core MacBook pro would have taken at least 2 days (55 runs * 30 seconds * 10 fold * 10 bags = 45 hours)!

With all the hype over cloud computing, I decided to give AWS GPU-compute series spot instances a go, which was about 0.35 cents per hour. While there are plenty of online resources about how to implement this, they are not without missing pieces. 

I will write a guide on installing Anaconda, Juypter, Cudas, Keras and Theano in a separate post soon. 

For those who are starting out with Keras like me, there are two things you must note: 

In your home directory, run the following:

```
cd .keras/
nano keras.json #use open/subl depending on your OS 
```
You should see:

```
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "Tensorflow"
}
```

If you want to use Theano, you have to edit `Tensorflow` to `Theano`

```
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
```
If you are using an Nvidia GPU, installing Theano would introduce a `.theanorc` file in your home directory. If it's not there, you need to create one and paste the following inside: 

```
[global]
floatX = float32
device = gpu
[mode] = FAST_RUN

[nvcc]
fastmath=TRUE

[cuda]
root = /usr/local/cuda

[lib]
cnmem = 0.95
```

The line `cnmem = 0.95` is very important—it halves the duration of each iteration from 12 to 6 seconds!

##### Print Validation Loss and Early Stopping

To see monitor whether your score is improving and to determine early stopping, you need to first specify the metric function:

```
def mae(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_true) - K.exp(y_pred)))
```

and put it inside the `keras.compile` function:

```
model.compile(loss = 'mae', optimizer = optimizer, metrics=[mae])

```
Use this code to determine early stopping and checkpoints for your model:

```
callsback_list = [EarlyStopping(patience=10),
                  ModelCheckpoint('keras-regressor-' + str(i+1) +'_'+ str(j+1) + '.check'\
                                  , monitor='val_loss', save_best_only=True, verbose=0)]
```

Then, specify the `validation_data` and `callsback`:

```
fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0,
                                  validation_data=(xte.todense(),yte),
                                  callbacks=callsback_list)
```

where `xtr, ytr` is the training set, and `xte, yte` the validation set in the specified `Kfold`. 

You can then call the best model with `val_loss` or `val_mae`, before making the best prediction with:

```
fit = load_model('keras-regressor-' + str(i+1) + '_'+ str(j+1) + '.check')
    
pred += np.exp(fit.predict_generator(generator = batch_generatorp(xte, 800, False), \
                                       val_samples = xte.shape[0])[:,0])-200
```

##### Spoiler alert!

To see how I used early stopping for my second level modelling, check out [this link](https://github.com/Freedom89/Allstate_kaggle/blob/master/second_level_models/keras_stacking_single_fold.ipynb). Turns out that saving the model is a good idea if you want to use it later on. You can try it by commenting out the following lines from the notebook:

```
#comment from here 
callsback_list = [EarlyStopping(patience=10),\
          ModelCheckpoint('keras-regressor-' + str(i+1) +'_'+ str(j+1) + '.check'\
                          , monitor='val_loss', save_best_only=True, verbose=0)]
    
model = nn_model(layer1=250,layer2=100,\
     dropout1 = 0.4,dropout2=0.2, \
     optimizer = 'adadelta')
    
fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                          nb_epoch = nepochs,
                          samples_per_epoch = xtr.shape[0],
                          verbose = 0,
                          validation_data=(xte.todense(),yte),
                          callbacks=callsback_list)
    
# to here if you just want to use the pre-trained models yourself.
```

#### [Back to contents](#start)

####  <a name="ensemble1"></a>Ensemble Version 1
***
After training my neural networks, I randomly assigned weights to my best XGB and Keras predictions to see which would fit the public leaderboard best. 

When I decided to do stacking, I started reading up:

* [Here](http://mlwave.com/kaggle-ensembling-guide/)
* [Here](https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867/code)
* [And here](https://www.kaggle.com/c/allstate-claims-severity/forums/t/25743/stacking-understanding-python-package-for-stacking)

I did not have any of my out-of-bag (OOB) training sets. This meant that I had to re-write my codes and retrain the models. **Lesson Learnt: You should always extract the OOB sets for models running on *k*-fold validation even if you are unsure stacking is required.**

##### XGB example (pseudo code) 

```
pred_oob = np.zeros(train_x.shape[0])

for iterations in kfold:
	#split training and test set 
    scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    pred_oob[test_index] = np.exp(scores_val) - shift

#export the oob training set
oob_df = pd.DataFrame(pred_oob, columns = ['loss'])
sub_file = 'oob_xgb_fairobj_' + str(score) + '_' + str(
    now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("Writing submission: %s" % sub_file)
oob_df.to_csv(sub_file, index = False)   

```
This concept would apply to other models as well. 

Additionally, if you are using AWS, disconnections to your juypter kernels might disrupt tracking of your code progress (e.g which fold or bags it was running). I overcame this problem by adding these lines in my code:

For XGB:

```
partial_evalutaion = open('temp_scores_power2.txt','a') 

partial_evalutaion.write('Fold '+ str(i) + '- MAE:'+ str(cv_score)+'\n')

partial_evalutaion.flush()
```

For hyperopt:

```
partial_evalutaion = open('extra_trees_bootstrap2.txt','a')   

partial_evalutaion.write('iteration ' + str(space) +str(iter_count) + 'with' + ' ' + str(score) + '\n')
partial_evalutaion.flush()
```

To store all the parameters that ran on hyperopt, you can specify a data frame and call it within the function to append the results:

```
Df_results = pd.DataFrame() 

def objective(space):
	...
	...
	global Df_results
    Df_results = Df_results.append(log_files_df)
   
   return(...)
   
Df_results.to_csv("results.csv",index = None) 
 
```

#### [Back to contents](#start)

####  <a name="ensemble2"></a>Ensemble with Weighted Average
***

The idea for using a weighted average (in my opinion) stems from linear programming. Credit should go to [this post](https://www.kaggle.com/tilii7/allstate-claims-severity/ensemble-weights-minimization-vs-mcmc/comments) for sharing the code on finding optimal weights. 

To implement this,

* Bind all your OOB-training set together, then
* Define the objective function as follows:

  ```
  def mae_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return mean_absolute_error(Y_values, final_prediction)
    
  ```
* After which, run the following code:

  ```
  for i in range(100):
    starting_values = np.random.uniform(size=len(predictions))
    cons = ({'type':'ineq','fun':lambda w: 1.2-sum(w)})
    bounds = [(0,1)]*len(predictions)

    res = minimize(mae_func, starting_values, method='L-BFGS-B',
                   bounds=bounds, options={'disp': False, 'maxiter': 10000})

    lls.append(res['fun'])
    wghts.append(res['x'])
    
  bestSC = np.min(lls)
  bestWght = wghts[np.argmin(lls)]

  ```



For those familiar with linear programming, you would understand that

`cons = ({'type':'ineq','fun':lambda w: 1.2-sum(w)}` 

implies that the sum of weights should not be greater than 1.2, while 

`bounds = [(0,1)]*len(predictions)` 

implies that each weight should be between `0` and `1`.

With 6 XGB models and 4 Keras model, I generated [this result](https://github.com/Freedom89/Allstate_kaggle/blob/master/allstate1117.71816974.csv), which would have ranked me at 102th on the private leaderboard. The local CV score was about `1118.34` .


#### [Back to contents](#start)

####  <a name="ensemble3"></a>Ensemble with NN 
***

[This post](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26359/is-stacking-working-better-than-weighted-average-for-you?forumMessageId=149495#post149495) inspired me to try neural networks as my second level model. 

I must also admit I was pretty lucky in my first guess of parameters of a two layer NN with 250-100 nodes, found [here](https://github.com/Freedom89/Allstate_kaggle/blob/master/second_level_models/keras_stacking_single_fold.ipynb). My 5-fold approach generated the following results:

```
('Fold ', 1, '- MAE:', 1117.5260825665521)
('Fold ', 2, '- MAE:', 1113.2272453103922)
('Fold ', 3, '- MAE:', 1117.1135764027533)
('Fold ', 4, '- MAE:', 1121.9982577768997)
('Fold ', 5, '- MAE:', 1119.6595219061744)
('Total - MAE:', 1117.9049057391971)
```

Interestingly, I tried 10 folds but obtained a worse CV result than `1118.34` (although it might not have meant a worse LB). 

Afterwards, I decided to bag my model 5 times :

```
('Fold ', 1, '- MAE:', 1117.6329549570657)
('Fold ', 2, '- MAE:', 1113.3701316951469)
('Fold ', 3, '- MAE:', 1117.1293409206548)
('Fold ', 4, '- MAE:', 1121.8204992333194)
('Fold ', 5, '- MAE:', 1119.4491190596229)
('Total - MAE:', 1117.880379920515)
```
You can see the entire output [here](https://github.com/Freedom89/Allstate_kaggle/blob/master/second_level_models/keras_stacking_bagged.ipynb).

However, the score was only 0.02 points better. Hence, I decided to weigh them with my best first level models, and tried the following submissions on the **last day**:


|            | Single 5 fold Keras | W. avg with single fold NN | Bagged 5 fold Keras | W.avg with lvl 1 models with both NN |
| :--------- | :------------------ | :------------------------- | :------------------ | :----------------------------------- |
| Local CV   | 1117.90490574       | 1117.77760897              | 1117.88037992       | 1117.7181697                         |
| Public LB  | 1100.90013          | 1100.87763                 | 1100.88155          | 1100.86946                           |
| Private LB | 1112.84611          | 1112.77244                 | 1112.93370          | 1112.73936                           |

Surprisingly, the single 5-fold model performed better in the private LB than the bagged model. 

The final weighted scores, codes and datasets I used can be found [here](https://github.com/Freedom89/Allstate_kaggle/blob/master/fmin_second_level.ipynb). 



#### [Back to contents](#start)

####  <a name="reflections"></a>Things I should/would have tried 
***

1. Shuffle the datasets by binning the target—my *k*-fold CV results were pretty inconsistent
2. Train specific models for high values of the dataset
3. Bagging with other datasets, as well as other forms of feature engineering 
4. Further explore second level modelling, e.g. using a weighted average with only second level models

Here are additional links to top solutions:

* [1st place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26416/1st-place-solution)
* [2nd place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26427/2nd-place-solution)
* [3rd place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26447/faron-s-3rd-place-solution)
* [7th place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26537/giba-7-place-solution)
* [8th place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26440/8-solution)
* [12th place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26414/12th-place-solution)
* [16th place](https://www.kaggle.com/c/allstate-claims-severity/forums/t/26430/16-place-solution-and-some-questions-about-it)


#### [Back to contents](#start)

####

##  Want to leave comments? Visit the github issue page [here](https://github.com/Freedom89/Allstate_kaggle/issues).

# Thanks For Reading! 


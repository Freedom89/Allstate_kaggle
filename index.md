#Hello Dear Reader,

###Feel free to skip directly to the [Content](#start) ,
***

### Introduction

I am Yi Xiang, currently employed at a Junior Data Scientist! Do drop me a message or feedback if you think I can improve in any way possible! You can reach me at my [linked-in](https://www.linkedin.com/in/yi-xiang-low-b349137b)! 

In the many Meet-ups i have attended, many fledging data scientists to-be have questioned employers in Data Science or expert data scientists:

	What do you look for in a potential hire? 
	
The answer is usually:
	
	* Ability to communicate or story-tell
	* Coding ability 
	* Passion / Learning ability 
	* And the lists goes on...

Which would be followed up by:

	 "How should i demonstrate this to my potential employer?"
	 
The advice then given is: 

	* Increase your online presence (through github, blogs)
	* Find interesting projects to work on at your free time.

*** 
 
**The truth is that I have procrastinated for quite a long time (with regard to the above two points), and it is about time i do something about it.** To those who are in this field, everyone is talking about [xgboost](http://xgboost.readthedocs.io/) when it comes to **ANY** (structured) machine learning problems. 

Hence, I started out the competition with an initial goal of learning how to tune Xgboost and this is how my goals got evolved during the competition over a single month: 

1. Manage to squeeze into top 10% with XGB's and wanted to attempt at achieving a bronze medal (top 10%). 
2. Within no time at all, i was kicked out of top 10%. In order to climb back up to 10%, I had to ensemble with different models. Most people in the forums had recommended Neural Nets to ensemble with XGB. Unfortunately i had no experience with Neural Nets (other than Coursera) before!  
3. Neural Nets are **Very Very** slow on CPU, thus to speed things up, i also had to learn how to set up CUDA on AWS GPU-compute series, install python, Theano, transfer the data and learning to configure juypter notebook! 
4. With my Neural Net and XGB, a simple average got me into top 5% - great! let's attempt for a sliver medal instead (top 5%).
5. Unfortunately, within no time at all (again), i was almost kicked out of top 5% and there was no guarantee that i might drop further on the private leaderboard due to overfitting.
6. Time for stacking! I understood the concept, but i have never done stacking before ; Sadly, i did not extract my out-of-bag predictions from my previous models (Painful but important lesson). I then tried Xgboost and ridge regression for my second level modelling, which yielded lousy results. I gave up and decided to settle for top 10% instead. 
7. Someone posted about using Neural Nets as a second level model close to the last day, which i decided to give it a last burst of fire - which worked; i was rank 78 on the public leaderboard and 46 on the private leaderboard which was really a surprise! 

***
**My point about saying all these is:** 

	Kaggle is really a good place to start with lots of helpful people sharing.

***
	
### Therefore, my objective for publishing this post is to:  
	
1. Share about my experience, learnings! 
2. Document my code! 
3. Encourage fledging data scientists to-be to start! 

***

<a name="start"></a>

<a href="https://www.kaggle.com/c/allstate-claims-severity" target="_blank"><img src="https://www.allstatenewsroom.com/wp-content/uploads/2015/12/Allstate_Logo4.jpeg" width="400"></a>

#"How severe is an insurance claim?"

The competition was to create an algorithm which accurately predicts claims severity, and the competition metric is *[Mean Absolute Error(MAE)](https://www.kaggle.com/wiki/MeanAbsoluteError)* . 

## Contents

1. [Custom Objectives](#custom)
2. [Finding Interactions, Encoding, boxcox](#xgbfir)
3. [Tuning Xgb](#Tune)
4. [Neural Networks](#NN)
5. [Ensemble version 1](#ensemble1)
6. [Ensemble with weighted average](#ensemble2)
7. [Ensemble with NN](#ensemble3)
8. [Things i should/would have tried](#reflections)


####  <a name="custom"></a>Custom Objectives  

***

The first thing i have learnt about [MAE](http://www.vanguardsw.com/business-forecasting-101/mean-absolute-deviation-mad-mean-absolute-error-mae/) objective is that it optimises for the median rather than the mean, as compared to MSE which penalises more for points far away from the mean. More information can be found [here](http://stats.stackexchange.com/questions/147001/is-minimizing-squared-error-equivalent-to-minimizing-absolute-error-why-squared).

If you had taken undergraduate mathematics, you would know that `y = |x|` is non differentiable at ` x = 0`. [When you are using Xgboost ` eval_metric = 'mae' `, the algorithm would still descent by MSE which poses a problem if you are optimising for MAE](http://stackoverflow.com/questions/34178287/difference-between-objective-and-feval-in-xgboost). One natural way to overcome this is to simply 'squeeze' the target variable, so the effect of values far away from the mean does not get over-penalised. 

It turns out that numerical approximation is very useful (thank you taylor series!). This [link](http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html) (**worth reading!**) describes the intuition behind optimising for MAE.

To summarise, you can observe that the 'Fair' objective function mimics the least-absolute function pretty accurately.  


<img src="http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/img334.gif" width="400">

The objective, gradient (first derivative), hessian (second derivative) of the above functions are defined as follows:

<img src="http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/img333.gif" width="400">

Majority of the scripts in the forums uses the 'Fair' objective, - the code is 

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
The smaller `fair_constant` is, the 'slower/smoother' the loss is. 

This custom objective can then be used in xgb.train:

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

Additional/ Majority  of these information can be found [here](https://www.kaggle.com/c/allstate-claims-severity/forums/t/24520/effect-of-mae).
 
#### [Back to contents](#start)
####  <a name="xgbfir"></a>Finding Interactions, Encoding, boxcox

***

##### Finding interactions
One of the problems of linear regression is finding feature interactions. [This](https://github.com/Far0n/xgbfi) solves the problem by finding N-way interactions you can use to improve your xgb or as features to input to your model. 

Fortunately, someone else posted [this](https://www.kaggle.com/modkzs/allstate-claims-severity/lexical-encoding-feature-comb/discussion) which saved me abit of time on finding N-way feature interactions. 

##### Encoding

Another interesting function i have learnt on encoding categorical features as compared to label encoding or one hot encoding:

```
def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r
    
```
In the raw data, the features are`A, B, ... , Z, then AA, .. AZ ..`, which seems to suggest that the order matters. 

What the function does is essentially:

* `encode('A')   = 1 `
* `encode('Z')   = 26`
* `encode('AA')  = 27`
* `encode('AC')  = 29`

This can be used in addition to `min/max/mean/counts/ti-idf` which i did not get to try in this competition. 

##### Boxcox

In Certain Machine Learning Algorithms, a normally distributed column can help the algorithm perform better. Unfortunately, figuring out the exact transformation to take on each individual column requires a huge effort.

Introducing boxcox, a (very) decent way of transforming these features by measuring their [skew](https://en.wikipedia.org/wiki/Skewness).

These are good articles on explaining boxcox that i came across:

* [Fairly Layman](https://www.isixsigma.com/tools-templates/normality/making-data-normal-using-box-cox-power-transformation/)
* [Math and more Math](http://onlinestatbook.com/2/transformations/box-cox.html) Never thought year one calculus would be this useful!

Some code:

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

Unless you are extremely experienced in Machine Learning and have a great intuition over parameters, it is likely you have to do some trial and error. 

I recommend [hyperopt](https://github.com/hyperopt/hyperopt) which is a python library for serial and parallel optimization over awkward search spaces. You can even adjust / change the number of layers in a Neural Net! 

I have some examples in my git repo:

* [Xgb](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/hyper_opt_xgb.ipynb)
	* Change the data input to power3 if you want to run hyperopt for 3-Way interaction	 
* [Extra Trees](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/extratrees_hyper_opt.ipynb)
* [Random Forest](https://github.com/Freedom89/Allstate_kaggle/blob/master/hyperopt_results/hyper_opt_random_forest.ipynb)

The results of the hyperopt can also be found in the [repo](https://github.com/Freedom89/Allstate_kaggle/tree/master/hyperopt_results).

#### [Back to contents](#start)

####  <a name="NN"></a>Neural Networks

As mentioned, this is my first time coding a neural net outside of Coursera ([Andrew Ng's Machine Learning](https://www.coursera.org/learn/machine-learning) and [UOW's Machine Learning specialisation](https://www.coursera.org/specializations/machine-learning)).

##### Using AWS 

[This](https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364/comments) script helped a lot in starting out with Neural Networks, but a 8 core Mac Book pro would have taken at least 2 days (55 runs * 30 seconds * 10 fold * 10 bag = 45 hours)!

With the hype about Cloud, I decided to give AWS GPU-compute series spot instances a go, which was about 0.35 cents an hour. There are plenty of resources online, but there are still a bit of missing pieces). 

I will write my own guide of installing Anaconda, Juypter, Cudas, Keras, Theano in a separate post soon. 

For those who are starting out with Keras like me, there are two things you must take note: 

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

If you need to use Theano, you have to edit `Tensorflow` to `Thenao`

```
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
```
After installing theano and if you are using nvidia GPU, you should have a `.theanorc` file in the home directory, if it is not available you need to create it and paste the following: 

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

The line `cnmem = 0.95` is very important, it speed up each iteration from 12 seconds to 6! 	

##### Print Validation loss and Early Stopping

Similar to XGB, to see how the score is improving and early stopping:

You need to first specify the metric function:

```
def mae(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_true) - K.exp(y_pred)))
```

and put it inside the `keras.compile` function:

```
model.compile(loss = 'mae', optimizer = optimizer, metrics=[mae])

```
And the early stopping + Model checkpoint, 

```
callsback_list = [EarlyStopping(patience=10),
                  ModelCheckpoint('keras-regressor-' + str(i+1) +'_'+ str(j+1) + '.check'\
                                  , monitor='val_loss', save_best_only=True, verbose=0)]
```

and in the model specify the validation data as well as callsback_list:

```
fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0,
                                  validation_data=(xte.todense(),yte),
                                  callbacks=callsback_list)
```

where `xtr, ytr` is your training set, and `xte, yte` is your validation set in specified from `Kfold`. 

You can then call the best model with `val_loss` or `val_mae` and make the best prediction with:

```
fit = load_model('keras-regressor-' + str(i+1) + '_'+ str(j+1) + '.check')
    
pred += np.exp(fit.predict_generator(generator = batch_generatorp(xte, 800, False), \
                                       val_samples = xte.shape[0])[:,0])-200
```

[Spoiler alert](https://github.com/Freedom89/Allstate_kaggle/blob/master/second_level_models/keras_stacking_single_fold.ipynb) to see how i used early stopping for my second level modelling. It also turns out that saving the model is a good idea if you want to use it later on. You can try it by commenting out these lines in the notebook:

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

***
#### [Back to contents](#start)

####  <a name="ensemble1"></a>Ensemble version 1
***
#### [Back to contents](#start)

####  <a name="ensemble2"></a>Ensemble with weighted average
***
#### [Back to contents](#start)

####  <a name="ensemble3"></a>Ensemble with NN 
***
#### [Back to contents](#start)

####  <a name="reflections"></a>Things i should/would have tried 
***
#### [Back to contents](#start)

####
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
 
The truth is that I have procrastinated for quite a long time (with regard to the above two points), and it is about time i do something about it. To those who are in this field, everyone is talking about [xgboost](http://xgboost.readthedocs.io/) when it comes to **ANY** (structured) machine learning problems. 

Hence, I started out the competition with an initial goal of learning how to tune Xgboost and this is how my goals got evolved during the competition over a single month: 

1. Manage to squeeze into top 10% with XGB's and wanted to attempt at achieving a bronze medal (top 10%). 
2. Within no time at all, i was kicked out of top 10%. In order to climb back up to 10%, I had to ensemble with different models. Most people in the forums had recommended Neural Nets to ensemble with XGB. Unfortunately i had no experience with Neural Nets (other than Coursera) before!  
3. Neural Nets are **Very Very** slow on CPU, thus to speed things up, i also had to learn how to set up CUDA on AWS GPU-compute series, install python, Theano, transfer the data and learning to configure juypter notebook! 
4. With my Neural Net and XGB, an simple average got me into top 5% - great! let's attempt for a sliver medal instead (top 5%).
5. However within no time at all, i was almost kicked out of top 5% and there was no guarantee that i might drop further on the private leaderboard due to overfitting.
6. Time for stacking! I understood the concept, but i have never done stacking before ; Sadly, i did not extract my out-of-bag predictions from my previous models (Painful but important lesson). I then tried Xgboost and ridge regression for my second level modelling, which yielded lousy results. I gave up and decided to settle for top 10% instead. 
7. Someone posted about using Neural Nets as a second level model close to the last day, which i decided to give it a last burst of fire - which worked; i was rank 78 on the public leaderboard and 46 on the private leaderboard which was really a surprise! 

***
**My point about saying all these is:** 

	Kaggle is really a good place to start with lots of helpful people sharing.
	
1. Share about my experience, learnings! 
2. Document my code! 
3. Encourage fledging data scientists (to be) to start! 

***

<a name="start"></a>

<a href="https://www.kaggle.com/c/allstate-claims-severity" target="_blank"><img src="https://www.allstatenewsroom.com/wp-content/uploads/2015/12/Allstate_Logo4.jpeg" width="400"></a>

#"How severe is an insurance claim?"

The competition was to create an algorithm which accurately predicts claims severity. The competition metric is *[Mean Absolute Error(MAE)](https://www.kaggle.com/wiki/MeanAbsoluteError)*  





3r2

32r

23r
32r2
3r23r

23r

2

23r

32r3r2  

  
  23r
23  
r  
23r  
23  
r  
23r  
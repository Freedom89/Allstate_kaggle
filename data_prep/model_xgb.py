import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

ids = pd.read_csv('./input/test.csv')['id']
train_x = pd.read_csv("./input/train_x3.csv")
train_y = pd.read_csv("./input/train_y3.csv",header = None)
test_x = pd.read_csv("./input/test_x3.csv")

def fair_obj(preds, dtrain):
    fair_constant = 2
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

shift = 200
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

def xgb_model(params , train_x, train_y, ids, test_x ,n_folds = 5,n_print = 500, early_stop = 50,shift=200):
    #n_folds = n_folds
    cv_sum = 0
    fpred = []
    xgb_rounds = []
    test_x = xgb.DMatrix(test_x)
    kf = KFold(train_x.shape[0], n_folds=n_folds,random_state = 2016)

    pred_oob = np.zeros(train_x.shape[0])

    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=early_stop,
                        obj=fair_obj,
                        verbose_eval = n_print,
                        feval=xg_eval_mae)



        xgb_rounds.append(clf.best_iteration)
        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
        pred_oob[test_index] = np.exp(scores_val) - shift
        cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
        print('eval-MAE: %.6f' % cv_score)
        y_pred = np.exp(clf.predict(test_x, ntree_limit=clf.best_ntree_limit)) - shift

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score

        partial_evalutaion = open('temp_scores1.txt','a') 
        partial_evalutaion.write('Fold '+ str(i) + '- MAE:'+ str(cv_score)+'\n')
        partial_evalutaion.flush()


    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average eval-MAE: %.6f' % score)
    n_rounds = int(np.mean(xgb_rounds))

    print("Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = ids
    result = result.set_index("id")
    print("%d-fold average prediction:" % n_folds)


    now = datetime.now()
    score = str(round((cv_sum / n_folds), 6))
    sub_file = 'test_xgb_fairobj_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id')

    print("writing out of bag results")
    oob_df = pd.DataFrame(pred_oob, columns = ['loss'])
    sub_file = 'oob_xgb_fairobj_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Writing submission: %s" % sub_file)
    oob_df.to_csv(sub_file, index = False)
    
    return (params,score,result,n_rounds,oob_df)
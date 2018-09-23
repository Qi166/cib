# -*- coding: utf-8 -*-


import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()\
                                        .sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    
def evaluate(X, y):
    start = time.time()
    # AUC
    lgb_cv = lgb.cv(l_params, lgb.Dataset(X, label=y), 10000, nfold=5,\
                    verbose_eval=100, early_stopping_rounds=100)
    print('highest cv is: ',lgb_cv['auc-mean'][-1], 'round is: ',len(lgb_cv['auc-mean']))
    
    # F1-阈值
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4242)
    model = lgb.train(l_params, lgb.Dataset(x_train, label=y_train), len(lgb_cv['auc-mean']))
    y_pred_valid = model.predict(x_valid)
    best = 0
    index = 0
    for threshold in range(0,100,1):
        t = threshold/100.0
        tmp = f1_score(y_valid,y_pred_valid>t)
        if tmp > best:
            best = tmp
            index = t
            cm = confusion_matrix(y_valid,y_pred_valid>t)
    print('threshold is: ',index)
    print('f1-score is: ',best)
    plot_confusion_matrix(cm, classes=['0','1'], title='Confusion matrix')
    
    # final score
    print('final score is: ', lgb_cv['auc-mean'][-1]*0.7 + best*0.3)
    
    # 特征重要性
    importance_df = pd.DataFrame()
    importance_df["feature"] = list(x_train.columns)
    importance_df["importance"] = model.feature_importance(importance_type='split', iteration=-1)
    
    display_importances(importance_df)
    
    return lgb_cv, index


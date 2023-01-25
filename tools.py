# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 20:14:13 2021

@author: dell
"""
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


# from aif360.algorithms.inprocessing import GerryFairClassifier,PrejudiceRemover
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from mlxtend.classifier import StackingClassifier

def 初始化分类器():    
   
    #初始化skl中的分类器
    xgb = XGBClassifier(random_state=1) #XGBoost
    gbdt = GradientBoostingClassifier(random_state=0) #GBDT
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_split=20, min_samples_leaf=5),
                          algorithm="SAMME", n_estimators=300, learning_rate=0.05,random_state=0) #AdaBoost
    # lda = LinearDiscriminantAnalysis() #LDA
    rf = RandomForestClassifier(random_state=2) #RF
    svm = SVC(probability=True,random_state=3) #SVM
    # nn = MLPClassifier() #NN
    # dt = DecisionTreeClassifier() #DT
    # knn = KNeighborsClassifier() #KNN
    lr = LogisticRegression() #LR
    lgboost = LGBMClassifier(random_state=0) #LGB
    # et = ExtraTreesClassifier() #ExtraTree
    vt = StackingClassifier(classifiers=[('xgb', xgb),('gbdt',gbdt),('ada',ada)],meta_classifier= LogisticRegression())
    
    # classifiers_skl = [xgb, gbdt, ada, lda, rf, svm, nn, dt, knn, lr, lgboost, et, vt]
    classifiers_skl = [xgb, gbdt, ada,rf,svm,lr,lgboost,vt]

    performance_skl = []
    for i in range(len(classifiers_skl)):
        performance_skl.append(pd.DataFrame())
    
    clf_all =classifiers_skl
    clf_names = [clf.__class__.__name__ for clf in clf_all]
    
    return classifiers_skl, performance_skl, clf_names

from aif360.algorithms.inprocessing import GerryFairClassifier,PrejudiceRemover,MetaFairClassifier
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing,EqOddsPostprocessing,RejectOptionClassification

def 初始化对比分类器(feature,protected_value):
    #初始化公平性分类器
    clf1 = GerryFairClassifier()
    clf2 = PrejudiceRemover(eta=0.5, sensitive_attr=feature, class_attr="")
    clf3 = MetaFairClassifier(tau= 0, sensitive_attr=feature)
    # clf2 = AdversarialDebiasing() #需要在tensorflow环境下跑
    
    clf4 = CalibratedEqOddsPostprocessing(unprivileged_groups =[{feature:protected_value}],
                                          privileged_groups =[{feature:abs(1-protected_value)}],
                                          cost_constraint='fnr', seed = 0)
    clf5 = EqOddsPostprocessing(unprivileged_groups =[{feature:protected_value}],
                                privileged_groups =[{feature:abs(1-protected_value)}],
                                seed = 0)
    clf6 = RejectOptionClassification(unprivileged_groups =[{feature:protected_value}],
                                      privileged_groups =[{feature:abs(1-protected_value)}],
                                      low_class_thresh=0.1, high_class_thresh=0.9, 
                                      num_class_thresh=100, num_ROC_margin=50, 
                                      metric_name="Average odds difference", 
                                      metric_ub=0.1, metric_lb=-0.1)
    
    classifiers_aif = [clf1, clf2, clf3, clf4, clf5, clf6]
    # classifiersNames = [clf.__class__.__name__ for clf in classifiers_aif]
    # class_index = len(classifiersNames)

    performance_aif = []
    for i in range(len(classifiers_aif)):
        performance_aif.append(pd.DataFrame())
    
    clf_all = classifiers_aif
    clf_names = [clf.__class__.__name__ for clf in clf_all]
    return classifiers_aif,performance_aif, clf_names

def 定制结构化数据(X_train,Y_train,cloums_name,protected_value):
    """
    Parameters
    ----------
    X_train : DataFrame
        train data.
    Y_train : DataFrame
        train label.
    cloums_name : TYPE
        DESCRIPTION.
    protected_value : int
        0 or 1.

    Returns
    -------
    x : BinaryLabelDataset

    """

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    Y_train = pd.DataFrame(Y_train).reset_index(drop=True)
    x=pd.concat([X_train,Y_train], axis=1)
    x=BinaryLabelDataset(df = x,label_names = list(Y_train.columns) ,
                        protected_attribute_names=[list(X_train.columns)[cloums_name]],
                        unprivileged_protected_attributes=protected_value,
                        privileged_protected_attributes=abs(1-protected_value))
    return x


def skl结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,df,protected_value):
    
    cloums_name = list(X_test.columns).index(protected_attribute_name)
    X_orin = 定制结构化数据(X_test,Y_test,cloums_name,protected_value)
    X_pred = 定制结构化数据(X_test,Y_prep,cloums_name,protected_value)
    
    metric1=ClassificationMetric(X_orin, X_pred, 
                                 unprivileged_groups =[{protected_attribute_name:protected_value}],
                                 privileged_groups =[{protected_attribute_name:abs(1-protected_value)}])
    
    TPR = metric1.true_positive_rate()
    TNR = metric1.true_negative_rate()
    ACC= metric1.accuracy()
    BACC = 0.5*(TPR+TNR)
    EOD=metric1.equal_opportunity_difference()
    AOD = metric1.average_odds_difference()
    DI = metric1.disparate_impact()
    CO = metric1.generalized_entropy_index() #越小越公平
    df = df.append({'ACC':ACC,'BACC':BACC,'EOD':abs(EOD),'AOD':abs(AOD)}, ignore_index = True)
    # print(df)
    return df

def nsga结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    
    cloums_name = list(X_test.columns).index(protected_attribute_name)
    X_orin = 定制结构化数据(X_test,Y_test,cloums_name,protected_value)
    X_pred = 定制结构化数据(X_test,Y_prep,cloums_name,protected_value)
    
    metric1=ClassificationMetric(X_orin, X_pred, 
                                 unprivileged_groups =[{protected_attribute_name:protected_value}],
                                 privileged_groups =[{protected_attribute_name:abs(1-protected_value)}])
    
    ACC= metric1.accuracy()
    AOD = metric1.average_odds_difference()

    return ACC,AOD

def 结果计算函数(X_orin, X_pred,protected_attribute_name,df, protected_value):
    
    metric1=ClassificationMetric(X_orin, X_pred, 
                                 unprivileged_groups =[{protected_attribute_name:protected_value}],
                                 privileged_groups =[{protected_attribute_name:abs(1-protected_value)}])
    TPR = metric1.true_positive_rate()
    TNR = metric1.true_negative_rate()
    ACC= metric1.accuracy()
    BACC = 0.5*(TPR+TNR)
    EOD = metric1.equal_opportunity_difference()
    AOD = metric1.average_odds_difference()
    DI = metric1.disparate_impact()
    CO = metric1.generalized_entropy_index(alpha=1) #越小越公平
    df = df.append({'ACC':ACC,'BACC':BACC,'EOD':EOD,'AOD':AOD,'DI':DI, 'CO':CO}, ignore_index = True)
    
    return df

def 计算平均函数(classifiersNames,result):

    result_mean = pd.DataFrame()
    
    for j, name in enumerate(classifiersNames):
        result_mean = result_mean.append(result[j].mean(),ignore_index=True)

    result_mean.insert(0, 'name', classifiersNames)
    print(result_mean)
    return result_mean
    
    
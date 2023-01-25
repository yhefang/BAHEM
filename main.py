# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:33:12 2021

@author: dell
"""
import pandas as pd
import numpy as np
import time
import random
# from sklearn.svm import LinearSVC
from aif360.datasets import *
# from aif360.algorithms.preprocessing import
from preprocessing import *
from tools import *
from aif360.algorithms.inprocessing import *
# from aif360.algorithms.preprocessing.reweighing import Reweighing
# from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

ChooseFile = 4  #1-Adult, 2-Bank, 3-Compas, 4-German
kNumber =10  # 实验次数 
rand_start = 0 #起始随机种子设定


# ======================================
def subset(alist, idxs):
    '''
        用法：根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, group_num=2, shuffle=True, retain_left=False):
    '''
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist))) # 保留下标

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = []
    
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists.append(subset(alist, index[start:end]))
    
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index): # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists.append(subset(alist, index[end:]))
    
    return sub_lists

from sklearn.cluster import KMeans
from imblearn.over_sampling import ADASYN

def new1(X_train, Y_train, feature, rand):
    col_names = [x for x in X_train.columns]
    Ycol_name = [x for x in Y_train.columns]
    feature_index = col_names.index(feature)
    df_comb = pd.concat([X_train,Y_train],axis=1)
    
    # X_train_wadd = pd.concat([X_train,X_train[feature]],axis=1)
    list1 = [i for i in range(len(df_comb))]
    
    sublists = split_list(list1)
    
    df = pd.DataFrame()
    y = pd.DataFrame()
    #分成五个子集
    # for i in range(len(sublists)):
    #     sublist = sublists[i]
    #     df_sub = df_comb.iloc[sublist, :]
        
    rus = ADASYN(n_neighbors=3, random_state=rand)
    #聚类(可以尝试分为两种聚类方法，一种用原始标签，另一种用敏感特征，分别进行后合并)
    kmeans = KMeans(n_clusters=2, random_state=rand).fit(df_comb)
    labels = kmeans.labels_
   
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    
    for i in range(len(labels)):
        if labels[i] == 1:
            df1 = df1.append(df_comb.iloc[[i]])
        else :
            df2 = df2.append(df_comb.iloc[[i]])
        
    #在每个聚类出来的子集中根据失衡比例对特征或标签进行过采样
    if df1[feature].sum() > df1.iloc[:,-1].sum():
        #对特征进行平衡
        df1_copy = df1.copy()
        y_temp = df1[feature]   
        y_name = [feature]
        x_temp = df1.drop(feature,axis=1)
        x_name = [x for x in x_temp.columns]
        
        x_res, y_res = rus.fit_resample(x_temp, y_temp)
        
        x_res = pd.DataFrame(data=x_res,columns=x_name)
        y_res = pd.DataFrame(data=y_res,columns=y_name)
        
        x_res.insert(feature_index, feature, y_res)
        df1_res = x_res
        
        for i in range(len(df1_res)):
            if df1_res.iloc[:,-1][i] > 0.5:
                df1_res.iloc[:,-1][i] = 1
            else:
                df1_res.iloc[:,-1][i] = 0
        
    else:
        #对标签进行平衡 df1.iloc[:,feature_index]
        df1_copy = df1.copy()
        y_temp = df1.iloc[:,-1]   
        y_name = Ycol_name

        x_temp = df1.iloc[:,0:-1]
        x_name = col_names
        
        x_res, y_res = rus.fit_resample(x_temp, y_temp)
        
        x_res = pd.DataFrame(data=x_res,columns=x_name)
        y_res = pd.DataFrame(data=y_res,columns=y_name)
        
        df1_res = pd.concat([x_res,y_res],axis=1)
        
        for i in range(len(df1_res)):
            if df1_res.iloc[:,feature_index][i] > 0.5:
                df1_res.iloc[:,feature_index][i] = 1
            else:
                df1_res.iloc[:,feature_index][i] = 0

    if df2[feature].sum() > df2.iloc[:,-1].sum():
        #对特征进行平衡
        df2_copy = df2.copy()
        y_temp = df2[feature]   
        y_name = [feature]
        x_temp = df2.drop(feature,axis=1)
        x_name = [x for x in x_temp.columns]
        
        x_res, y_res = rus.fit_resample(x_temp, y_temp)
        
        x_res = pd.DataFrame(data=x_res,columns=x_name)
        y_res = pd.DataFrame(data=y_res,columns=y_name)
        
        x_res.insert(feature_index, feature, y_res)
        df2_res = x_res
        
        for i in range(len(df2_res)):
            if df2_res.iloc[:,-1][i] > 0.5:
                df2_res.iloc[:,-1][i] = 1
            else:
                df2_res.iloc[:,-1][i] = 0
        
    else:
        #对标签进行平衡 df1.iloc[:,feature_index]
        df2_copy = df2.copy()
        y_temp = df2.iloc[:,-1]   
        y_name = Ycol_name

        x_temp = df2.iloc[:,0:-1]
        x_name = col_names
        
        x_res, y_res = rus.fit_resample(x_temp, y_temp)
        
        x_res = pd.DataFrame(data=x_res,columns=x_name)
        y_res = pd.DataFrame(data=y_res,columns=y_name)
        
        df2_res = pd.concat([x_res,y_res],axis=1)
        
        for i in range(len(df2_res)):
            if df2_res.iloc[:,feature_index][i] > 0.5:
                df2_res.iloc[:,feature_index][i] = 1
            else:
                df2_res.iloc[:,feature_index][i] = 0 

    df_sub_res = df1_res.append(df2_res,ignore_index=True)
    
    # df = df.append(df_sub_res,ignore_index=True)
    
    x = df_sub_res.iloc[:,0:-1]
    y= df_sub_res.iloc[:,-1]
    return x,y
# ======================================
# ======================================
import copy
import gc
#First function to optimize 换acc
def function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    acc,aod = nsga结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    value = 1-acc
    return value

#Second function to optimize 换aod
def function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    acc,aod = nsga结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    value = abs(aod)
    return value


def post_nsga(times,X_test, Y_test, Y_prep, y_pred, protected_attribute_name,protected_value):
    
    acc_value = function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    aod_value = function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
   
    #在敏感特征多数中 随机选择修改
    if sum(X_test[protected_attribute_name]) > len(X_test[protected_attribute_name])/2:
        protect_f_index = []
        for i in range(len(X_test[protected_attribute_name])):
            if X_test[protected_attribute_name][i] == 1:
                protect_f_index.append(i)
    else:
        protect_f_index = []
        for i in range(len(X_test[protected_attribute_name])):
            if X_test[protected_attribute_name][i] == 0:
                protect_f_index.append(i)
    
    num_classes = y_pred.shape[1]
    k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True) # 归一化
    y_pred_entropy = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k) # top-k熵
    
    threshold = 0.5
    indexs = []
    for i in range(len(y_pred_entropy)):
        if [y_pred_entropy >= threshold][0][i] == True :
            indexs.append(i)

    y_pred_confident = y_pred[y_pred_entropy < threshold] # top-k熵低于阈值的是高置信度样本
    y_pred_unconfident = y_pred[y_pred_entropy >= threshold] # top-k熵高于阈值的是低置信度样本
    y_true_confident = np.array(Y_test[y_pred_entropy < threshold])
    y_true_unconfident = np.array(Y_test[y_pred_entropy >= threshold])
    
    prior = np.array([0.5,0.5])
    
    right, alpha, iters = 0, 1, 1 # 正确的个数，alpha次方，iters迭代次数
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0) # Y is L_0
        for _ in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1
    #==================================================
    change_index = []
    for i in indexs:
        for j in range(len(protect_f_index)):
            if i == protect_f_index[j]:
                change_index.append(i)
    
    # change_index=indexs
    
    # Y_prep_temp = copy.deepcopy(Y_prep)
    for i in change_index:
        # print(Y_prep_temp.iloc[i,0])
        Y_prep.iloc[i,0] = 1-Y_prep.iloc[i,0]
    
    # acc_valuem = function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    # aod_valuem = function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)

    # if acc_valuem+aod_valuem  > acc_value+aod_value :
    #     for i in change_index:
    #         # print(Y_prep_temp.iloc[i,0])
    #         Y_prep.iloc[i,0] = 1-Y_prep.iloc[i,0]
    #==================================================
    
    # results_temp = []
    # index_temp = []
    # for j in range(0,times):
        
    #     change_index=[random.randint(0,len(protect_f_index)-1) for i in range(0,int(0.1*len(protect_f_index)))]
        
    #     # Y_prep_temp = copy.deepcopy(Y_prep)
    #     for i in change_index:
    #         # print(Y_prep_temp.iloc[i,0])
    #         Y_prep.iloc[i,0] = 1-Y_prep.iloc[i,0]
        
    #     #修改完后再计算value并对比，好保留，坏不变
        
    #     acc_valuem = function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    #     aod_valuem = function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)

    #     index_temp.append(change_index)
    #     results_temp.append(acc_valuem+aod_valuem)
        
    #     for i in change_index:
    #         # print(Y_prep_temp.iloc[i,0])
    #         Y_prep.iloc[i,0] = 1-Y_prep.iloc[i,0]
        
    
    # best_index = results_temp.index(min(results_temp))

    # for i in index_temp[best_index]:
    #     Y_prep.iloc[i,0] = 1-Y_prep.iloc[i,0]
        
    
    return Y_prep
#========================================

def run_baseline(ChooseFile,受保护变量的列名,kNumber,rand_start):
    protected_value = 0

    random_n = [i for i in range(rand_start, rand_start + kNumber)]
    
    classifiers_skl,  performance_skl, clf_names= 初始化分类器()
    #只对比基分类器
    # classifiers_skl = classifiers_skl[:-1]
    part1_start_time = time.time()

    for n in range(kNumber):
        #试验模块1总时间
        
        #单次实验模块1时间
        part1_each_starttime = time.time()
        random_state = random_n[n]
        print('==============这是第 %d 次试验,random_state = %d==============' %(n+1, random_state))
        #train_data, vali_data, train_label, vali_label, test_data, test_label 
        X_train, X_vali, Y_train, Y_vali, X_test, Y_test, X_orig, Y_orig = Preprocessing(ChooseFile, random_state, 0.2)
        cloums_name = list(X_train.columns).index(受保护变量的列名)


        X_trainorig = pd.DataFrame(X_train)
        Y_trainorig = pd.DataFrame(Y_train)
        X_trainorig.columns = list(X_test.columns)
        Y_trainorig.columns = list(Y_test.columns)
        
        print('处理完毕，开始计算结果')
        X_all_orig = 定制结构化数据(X_orig,Y_orig,cloums_name,protected_value)
        X_test_orin = 定制结构化数据(X_test,Y_test,cloums_name,protected_value)
        train_stcut = 定制结构化数据(X_train,Y_train,cloums_name,protected_value)        
        
        #计算结果
        
        
        #对各个分类器根据各评估指标确定权重，进行排序，选用靠前的三个分类器作为投票的
        #基分类器。再将这三个分类器的结果进行调整后进行投票（最终返回一个预测的list）
        result1 = []
        for compare in range(len(classifiers_skl)):
            if classifiers_skl[compare].__class__.__name__ != 'StackingClassifier':
                classifiers_skl[compare].fit(X_train,Y_train)
                result = classifiers_skl[compare].predict(X_test)
                result_skl = pd.DataFrame(result)
                result_skl.columns=list(Y_test.columns)
                result1.append(result_skl)
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
            else:
                
                clf_stk = classifiers_skl[:-1] #这里可以加入一个根据性能选择分类器
                
                dataset_blend_train = np.zeros((X_train.shape[0], len(clf_stk)))      # 训练集样本数 X 基模型个数
                dataset_blend_test = np.zeros((X_test.shape[0], len(clf_stk)))  # 测试集样本数 X 基模型个数
                  
                '''5折stacking'''
                n_folds = 5
                skf = StratifiedKFold(n_folds)   # 初始化
                X_trainstk = np.array(X_train)
                Y_trainstk = np.array(Y_train)
                X_teststk = np.array(X_test)
                Y_teststk = np.array(Y_test)
                
                for j, clf in enumerate(clf_stk):
                    '''依次训练各个单模型'''
                    # print(j, clf)
                    dataset_blend_test_j = np.zeros((X_test.shape[0], n_folds))    # 测试集样本数 X 交叉验证个数（5）
                    for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
                        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                        #print("Fold", i)
                        X_strain, X_stest= X_trainstk[train], X_trainstk[test]
                        y_strain, y_stest = Y_trainstk[train], Y_trainstk[test]
                        clf.fit(X_strain, y_strain)
                        y_submission = clf.predict_proba(X_stest)[:, 1]
                        dataset_blend_train[test, j] = y_submission        # 5折交叉验证后的预测值（作为新的训练集）
                        dataset_blend_test_j[:, i] = clf.predict_proba(X_teststk)[:, 1]   # 
                    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
                    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)     # 对第一级stack预测的5次结果做平均
                    # print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))   # 计算模型预测的结果和测试集的得分！！
                
                # dataset_blend_test = post_prc(X_trainorig, Y_trainorig, X_test ,dataset_blend_test,受保护变量的列名, random_state)
                
                clf_fin = LogisticRegression()
                #clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
                clf_fin.fit(dataset_blend_train, Y_trainstk)    # 第二层stack
                train_pred = clf_fin.predict(dataset_blend_test)
                
                #根据stest中的敏感特征的比例调整 
                #把Y_train中的比例取出来
                
                result_skl = pd.DataFrame(train_pred)
                result_skl.columns=list(Y_test.columns)
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
        
        performance = performance_skl
        part1_each_endtime = time.time()
        part1_each_workingtime = part1_each_endtime - part1_each_starttime
        print("part1试验总共所花时间为：%.3f s" % part1_each_workingtime)
    
    print ("  ....................执行完成开始做平均....................")
    print ("="*60)

    Alog_mean = 计算平均函数(clf_names,performance)
    part1_end_time = time.time()
    part1_workingtime = part1_end_time - part1_start_time
    print("part2试验总共所花时间为：%.6f s" % part1_workingtime)
    
    return Alog_mean


def run_new1(ChooseFile,受保护变量的列名,kNumber,rand_start):
    protected_value = 0

    random_n = [i for i in range(rand_start, rand_start + kNumber)]
    
    classifiers_skl,  performance_skl, clf_names= 初始化分类器()
    #只对比基分类器
    # classifiers_skl = classifiers_skl[:-1]
    part1_start_time = time.time()

    for n in range(kNumber):
        #试验模块1总时间
        
        #单次实验模块1时间
        part1_each_starttime = time.time()
        random_state = random_n[n]
        print('==============这是第 %d 次试验,random_state = %d==============' %(n+1, random_state))
        #train_data, vali_data, train_label, vali_label, test_data, test_label 
        X_train, X_vali, Y_train, Y_vali, X_test, Y_test, X_orig, Y_orig = Preprocessing(ChooseFile, random_state, 0.2)
        cloums_name = list(X_train.columns).index(受保护变量的列名)
        
        X_train = pd.DataFrame(X_train)
        Y_train = pd.DataFrame(Y_train)
        X_train.columns=list(X_test.columns)
        Y_train.columns=list(Y_test.columns)
        
        X_train, Y_train = new1(X_train, Y_train,受保护变量的列名, random_state)

        X_train = pd.DataFrame(X_train)
        Y_train = pd.DataFrame(Y_train)
        X_train.columns=list(X_test.columns)
        Y_train.columns=list(Y_test.columns)
        
        print('处理完毕，开始计算结果')
        # X_all_orig = 定制结构化数据(X_orig,Y_orig,cloums_name,protected_value)
        # X_test_orin = 定制结构化数据(X_test,Y_test,cloums_name,protected_value)
        # train_stcut = 定制结构化数据(X_train,Y_train,cloums_name,protected_value)        
        
        #计算结果
        
        
        #对各个分类器根据各评估指标确定权重，进行排序，选用靠前的三个分类器作为投票的
        #基分类器。再将这三个分类器的结果进行调整后进行投票（最终返回一个预测的list）
        result1 = []
        for compare in range(len(classifiers_skl)):
            if classifiers_skl[compare].__class__.__name__ != 'StackingClassifier':
                classifiers_skl[compare].fit(X_train,Y_train)
                result = classifiers_skl[compare].predict(X_test)
                result_skl = pd.DataFrame(result)
                result_skl.columns=list(Y_test.columns)
                result1.append(result_skl)
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
            else:
                
                clf_stk = classifiers_skl[:-1] #这里可以加入一个根据性能选择分类器
                
                dataset_blend_train = np.zeros((X_train.shape[0], len(clf_stk)))      # 训练集样本数 X 基模型个数
                dataset_blend_test = np.zeros((X_test.shape[0], len(clf_stk)))  # 测试集样本数 X 基模型个数
                  
                '''5折stacking'''
                n_folds = 5
                skf = StratifiedKFold(n_folds)   # 初始化
                X_trainstk = np.array(X_train)
                Y_trainstk = np.array(Y_train)
                X_teststk = np.array(X_test)
                Y_teststk = np.array(Y_test)
                
                for j, clf in enumerate(clf_stk):
                    '''依次训练各个单模型'''
                    # print(j, clf)
                    dataset_blend_test_j = np.zeros((X_test.shape[0], n_folds))    # 测试集样本数 X 交叉验证个数（5）
                    for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
                        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                        #print("Fold", i)
                        X_strain, X_stest= X_trainstk[train], X_trainstk[test]
                        y_strain, y_stest = Y_trainstk[train], Y_trainstk[test]
                        clf.fit(X_strain, y_strain)
                        y_submission = clf.predict_proba(X_stest)[:, 1]
                        dataset_blend_train[test, j] = y_submission        # 5折交叉验证后的预测值（作为新的训练集）
                        dataset_blend_test_j[:, i] = clf.predict_proba(X_teststk)[:, 1]   # 
                    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
                    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)     # 对第一级stack预测的5次结果做平均
                    # print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))   # 计算模型预测的结果和测试集的得分！！
                
                # dataset_blend_test = post_prc(X_trainorig, Y_trainorig, X_test ,dataset_blend_test,受保护变量的列名, random_state)
                
                clf_fin = LogisticRegression()
                #clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
                clf_fin.fit(dataset_blend_train, Y_trainstk)    # 第二层stack
                train_pred = clf_fin.predict(dataset_blend_test)
                
                #根据stest中的敏感特征的比例调整 
                #把Y_train中的比例取出来
                
                result_skl = pd.DataFrame(train_pred)
                result_skl.columns=list(Y_test.columns)
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
        
        performance = performance_skl
        part1_each_endtime = time.time()
        part1_each_workingtime = part1_each_endtime - part1_each_starttime
        print("part1试验总共所花时间为：%.3f s" % part1_each_workingtime)
    
    print ("  ....................执行完成开始做平均....................")
    print ("="*60)

    Alog_mean = 计算平均函数(clf_names,performance)
    part1_end_time = time.time()
    part1_workingtime = part1_end_time - part1_start_time
    print("part2试验总共所花时间为：%.6f s" % part1_workingtime)
    
    return Alog_mean

def run_new2(ChooseFile,受保护变量的列名,kNumber,rand_start):
    protected_value = 0

    random_n = [i for i in range(rand_start, rand_start + kNumber)]
    
    classifiers_skl,  performance_skl, clf_names= 初始化分类器()
    #只对比基分类器
    # classifiers_skl = classifiers_skl[:-1]
    part1_start_time = time.time()

    for n in range(kNumber):
        #试验模块1总时间
        
        #单次实验模块1时间
        part1_each_starttime = time.time()
        random_state = random_n[n]
        print('==============这是第 %d 次试验,random_state = %d==============' %(n+1, random_state))
        #train_data, vali_data, train_label, vali_label, test_data, test_label 
        X_train, X_vali, Y_train, Y_vali, X_test, Y_test, X_orig, Y_orig = Preprocessing(ChooseFile, random_state, 0.2)
        cloums_name = list(X_train.columns).index(受保护变量的列名)
        
        X_trainorig = pd.DataFrame(X_train)
        Y_trainorig = pd.DataFrame(Y_train)
        X_trainorig.columns = list(X_test.columns)
        Y_trainorig.columns = list(Y_test.columns)
        
        X_train, Y_train = new1(X_train, Y_train,受保护变量的列名, random_state)

        X_train = pd.DataFrame(X_train)
        Y_train = pd.DataFrame(Y_train)
        X_train.columns=list(X_test.columns)
        Y_train.columns=list(Y_test.columns)
        
        print('处理完毕，开始计算结果')
        # X_all_orig = 定制结构化数据(X_orig,Y_orig,cloums_name,protected_value)
        # X_test_orin = 定制结构化数据(X_test,Y_test,cloums_name,protected_value)
        # train_stcut = 定制结构化数据(X_train,Y_train,cloums_name,protected_value)        
        
        #计算结果
        
        
        #对各个分类器根据各评估指标确定权重，进行排序，选用靠前的三个分类器作为投票的
        #基分类器。再将这三个分类器的结果进行调整后进行投票（最终返回一个预测的list）
        result1 = []
        for compare in range(len(classifiers_skl)):
            if classifiers_skl[compare].__class__.__name__ != 'StackingClassifier':
                classifiers_skl[compare].fit(X_train,Y_train)
                result = classifiers_skl[compare].predict(X_test)
                result_skl = pd.DataFrame(result)
                result_skl.columns=list(Y_test.columns)
                result1.append(result_skl)
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
            else:
                if ChooseFile==1:
                    clf_stk = [classifiers_skl[1],
                               classifiers_skl[6],
                               classifiers_skl[2]]
                elif ChooseFile==2:
                    clf_stk = [classifiers_skl[6],
                               classifiers_skl[1],
                               classifiers_skl[3]]
                elif ChooseFile==4 and 受保护变量的列名 == 'age':
                    clf_stk = [classifiers_skl[0],
                               classifiers_skl[1],
                               classifiers_skl[3]]
                elif ChooseFile==4 and 受保护变量的列名 == 'sex':
                    clf_stk = [classifiers_skl[1],
                               classifiers_skl[3],
                               classifiers_skl[6]]
                
                dataset_blend_train = np.zeros((X_train.shape[0], len(clf_stk)))      # 训练集样本数 X 基模型个数
                dataset_blend_test = np.zeros((X_test.shape[0], len(clf_stk)))  # 测试集样本数 X 基模型个数
                  
                '''5折stacking'''
                n_folds = 5
                skf = StratifiedKFold(n_folds)   # 初始化
                X_trainstk = np.array(X_train)
                Y_trainstk = np.array(Y_train)
                X_teststk = np.array(X_test)
                Y_teststk = np.array(Y_test)
                
                for j, clf in enumerate(clf_stk):
                    '''依次训练各个单模型'''
                    # print(j, clf)
                    dataset_blend_test_j = np.zeros((X_test.shape[0], n_folds))    # 测试集样本数 X 交叉验证个数（5）
                    for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
                        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                        #print("Fold", i)
                        X_strain, X_stest= X_trainstk[train], X_trainstk[test]
                        y_strain, y_stest = Y_trainstk[train], Y_trainstk[test]
                        clf.fit(X_strain, y_strain)
                        y_submission = clf.predict_proba(X_stest)[:, 1]
                        dataset_blend_train[test, j] = y_submission        # 5折交叉验证后的预测值（作为新的训练集）
                        dataset_blend_test_j[:, i] = clf.predict_proba(X_teststk)[:, 1]   # 
                    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
                    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)     # 对第一级stack预测的5次结果做平均
                    # print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))   # 计算模型预测的结果和测试集的得分！！
                
                # dataset_blend_test = post_prc(X_trainorig, Y_trainorig, X_test ,dataset_blend_test,受保护变量的列名, random_state)
                
                clf_fin = LogisticRegression()
                #clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
                clf_fin.fit(dataset_blend_train, Y_trainstk)    # 第二层stack
                train_pred = clf_fin.predict(dataset_blend_test)
                train_prob = clf_fin.predict_proba(dataset_blend_test)
                #根据stest中的敏感特征的比例调整 
                #把Y_train中的比例取出来
                
                result_skl = pd.DataFrame(train_pred)
                result_skl.columns=list(Y_test.columns)
                
                #插入post_nsga修改result_skl
                
                result_skl = post_nsga(2,X_test,Y_test,result_skl,train_prob,受保护变量的列名,0)
                
                performance_skl[compare]=skl结果计算函数(X_test,Y_test,result_skl,受保护变量的列名,performance_skl[compare],protected_value = 0)
        
        performance = performance_skl
        part1_each_endtime = time.time()
        part1_each_workingtime = part1_each_endtime - part1_each_starttime
        print("part1试验总共所花时间为：%.3f s" % part1_each_workingtime)
    
    print ("  ....................执行完成开始做平均....................")
    print ("="*60)

    Alog_mean = 计算平均函数(clf_names,performance)
    part1_end_time = time.time()
    part1_workingtime = part1_end_time - part1_start_time
    print("part2试验总共所花时间为：%.6f s" % part1_workingtime)
    
    return Alog_mean

if __name__ == '__main__':  
    # #baseline   
    # #1-Adult, 2-Bank, 3-Compas, 4-German
    # Adult_race = run_baseline(ChooseFile = 1, 受保护变量的列名 = 'race', kNumber = 10,rand_start = 0)
    # Adult_sex = run_baseline(ChooseFile = 1, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # Bank_age = run_baseline(ChooseFile = 2, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    # German_age = run_baseline(ChooseFile = 4, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    # German_sex = run_baseline(ChooseFile = 4, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # result = pd.concat([Adult_race,Adult_sex,Bank_age,German_age,German_sex])
    
    #new1
    #1-Adult, 2-Bank, 3-Compas, 4-German
    # Adult_race = run_new1(ChooseFile = 1, 受保护变量的列名 = 'race', kNumber = 10,rand_start = 0)
    # Adult_sex = run_new1(ChooseFile = 1, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # Bank_age = run_new1(ChooseFile = 2, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    # German_age = run_new1(ChooseFile = 4, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    # German_sex = run_new1(ChooseFile = 4, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # result = pd.concat([Adult_race,Bank_age,German_age,German_sex])
    
    #new2
    # 1-Adult, 2-Bank, 3-Compas, 4-German
    # Adult_race = run_new2(ChooseFile = 1, 受保护变量的列名 = 'race', kNumber = 10,rand_start = 0)
    # Adult_sex = run_new2(ChooseFile = 1, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # Bank_age = run_new2(ChooseFile = 2, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    German_age = run_new2(ChooseFile = 4, 受保护变量的列名 = 'age', kNumber = 10,rand_start = 0)
    # German_sex = run_new2(ChooseFile = 4, 受保护变量的列名 = 'sex', kNumber = 10,rand_start = 0)
    # result = pd.concat([Adult_race,Bank_age,German_age,German_sex])
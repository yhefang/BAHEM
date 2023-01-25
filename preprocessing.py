#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:15:34 2019

@author: loubeibei
"""

import copy
import pandas as pd
import numpy as np
# from sklearn import preprocessing
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
# from split import split_new
dataset_dir = 'dataset/'
from sklearn.model_selection import train_test_split
from aif360.datasets import *
import warnings
warnings.filterwarnings("ignore")


#哑变量处理 特征提取   pd.get_dummies()
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df  
                                                   

                                
def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df
def rename_function(X_df_New,col_NumAndCate):
    X_df_New = pd.DataFrame(X_df_New, columns=col_NumAndCate)
    return X_df_New
#数据预处理（参数：设置选择第几个文件）（from YDQ）
def select(X):
    temp = list(X.columns)
    temp2 = pd.DataFrame()
    name = []

    for i in range(X.shape[1]):
        if len(X[temp[i]].unique())>=5:
            temp2 = pd.concat([temp2,X[temp[i]]],axis=1)
            name.append(temp[i])
    norm = StandardScaler()
    minmax = MinMaxScaler()

    temp2 = norm.fit_transform(temp2)
    temp2 = minmax.fit_transform(temp2)
    return temp2,name
    
    
def change(X):
    if X <25:
        X=0
    elif X>45:
        X=2
    else:
        X=1
    return X
        
def Preprocessing(ChooseFile, rand, DivisionRatio):
    ###====================数据读取部分开始====================###
    # print("正在读取数据，请耐心等候...")
    Choose = ChooseFile
    #选择相应的文件
    if(Choose ==1):
        X =AdultDataset().convert_to_dataframe()[0]
        label_name = AdultDataset().convert_to_dataframe()[1]['label_names'][0]
        # print('数据集%s已经读入'%('AdultDataset'))
    elif(Choose ==2):
        X =BankDataset().convert_to_dataframe()[0]
        label_name = BankDataset().convert_to_dataframe()[1]['label_names'][0]
        print(X.value_counts)

        # print('数据集%s已经读入'%('BankDataset'))
    elif(Choose ==3):
        attr = ["age_cat", "race", "sex", "priors_count", "c_charge_degree",'two_year_recid']
        X =CompasDataset(features_to_keep=attr ).convert_to_dataframe()[0]
        label_name = CompasDataset().convert_to_dataframe()[1]['label_names'][0]
        X[label_name] = X[label_name].apply(lambda x: abs(x-1))

        # X['age'] = X['age'].apply(lambda x: change(x))
        # print('数据集%s已经读入'%('CompasDataset'))
    elif(Choose ==4):

        X =GermanDataset().convert_to_dataframe()[0]
        label_name = GermanDataset().convert_to_dataframe()[1]['label_names'][0]
        X['credit'] = X['credit'].apply(lambda x: abs(x-2))
        # print('数据集%s已经读入'%('GermanDataset'))
    

    temp  = select(X)
    
    X[temp[1]]=temp[0]
    
    Y = pd.DataFrame(X.pop(label_name).astype('int32'))
    # a = Y.value_counts()
    # print(a)

    # print('正在划分数据集，分出验证集：...')
    #分出验证集：藏一部分data来验证强分类器

    X_train_0,X_test_0, Y_train_0, Y_test_0 =train_test_split(X, Y,test_size=DivisionRatio, random_state=rand)

    #把用来训练若分类器的数据集分训练集和测试集
    X_normal, X_test_vali, Y_pca, Y_test_vali = train_test_split(X_train_0, Y_train_0,test_size=DivisionRatio, random_state=rand)

    ###====================数据预处理部分结束====================###
    X_normal = X_normal.reset_index(drop=True)
    X_test_vali = X_test_vali.reset_index(drop=True)
    Y_pca = Y_pca.reset_index(drop=True)
    Y_test_vali = Y_test_vali.reset_index(drop=True)
    X_test_0 = X_test_0.reset_index(drop=True)
    Y_test_0 = Y_test_0.reset_index(drop=True)

    return X_normal, X_test_vali, Y_pca, Y_test_vali, X_test_0, Y_test_0, X, Y  #输出的是DataFrame

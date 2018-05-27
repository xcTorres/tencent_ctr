import pandas as pd
import os,gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
import numpy as np

def gen_features():

    data = load_pickle(raw_data_path + 'preprocess.pkl')

    train = data[data.label != -1]

    del data
    gc.collect()


    print("uid_adCount")
    temp = train.groupby('uid')['aid'].nunique().reset_index()
    temp.columns=['uid','uid_adCount']
    temp.to_csv(feature_data_path+ "/statis/" +'uid_adCount.csv',index=False)

    temp = train.groupby('ct')['aid'].nunique().reset_index()
    temp.columns = ['ct', 'ct_adCount']
    temp.to_csv(feature_data_path + "/statis/" +'ct_adCount.csv', index=False)


    ## active user
    print("aid_userCount")
    temp = train.groupby('aid')['uid'].nunique().reset_index()
    temp.columns=['aid','aid_userCount']
    temp.to_csv(feature_data_path+"/statis/" +'aid_userCount.csv',index=False)


    print("adCategoryId_userCount")
    temp = train.groupby('adCategoryId')['uid'].nunique().reset_index()
    temp.columns=['adCategoryId','adCategoryId_userCount']
    temp.to_csv(feature_data_path+"/statis/" +'adCategoryId_userCount.csv',index=False)

    print("creativeId_userCount")
    temp = train.groupby('creativeId')['uid'].nunique().reset_index()
    temp.columns=['creativeId','creativeId_userCount']
    temp.to_csv(feature_data_path+"/statis/" +'creativeId_userCount.csv',index=False)

    print("LBS_userCount")
    temp = train.groupby('LBS')['uid'].nunique().reset_index()
    temp.columns=['LBS','LBS_userCount']
    temp.to_csv(feature_data_path+"/statis/" +'LBS_userCount.csv',index=False)


    ## active LBS
    temp = train.groupby('aid')['LBS'].nunique().reset_index()
    temp.columns = ['aid', 'aid_LBSCount']
    temp.to_csv(feature_data_path + "/statis/" +'aid_LBSCount.csv', index=False)

    temp = train.groupby('adCategoryId')['LBS'].nunique().reset_index()
    temp.columns = ['adCategoryId', 'adCategoryId_LBSCount']
    temp.to_csv(feature_data_path + "/statis/" +'adCategoryId_LBSCount.csv', index=False)

    temp = train.groupby('creativeId')['LBS'].nunique().reset_index()
    temp.columns = ['creativeId', 'creativeId_LBSCount']
    temp.to_csv(feature_data_path + "/statis/" +'creativeId_LBSCount.csv', index=False)


    ###age mean
    temp = train.groupby('aid')['age'].mean().reset_index()
    temp.columns = ['aid', 'ad_mean_age']
    temp.to_csv(feature_data_path + "/statis/" +'ad_mean_age.csv', index=False)

    temp = train.groupby('creativeId')['age'].mean().reset_index()
    temp.columns = ['creativeId', 'creativeId_mean_age']
    temp.to_csv(feature_data_path  + "/statis/" + 'creativeId_mean_age.csv', index=False)

    temp = train.groupby('adCategoryId')['age'].mean().reset_index()
    temp.columns = ['adCategoryId', 'adCategoryId_mean_age']
    temp.to_csv(feature_data_path + "/statis/" +'adCategoryId_mean_age.csv', index=False)



if __name__ == '__main__':
    gen_features()
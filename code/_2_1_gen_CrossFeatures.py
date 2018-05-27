import pandas as pd
import os,gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
import numpy as np

def gen_features():

    data = load_pickle(raw_data_path + 'preprocess.pkl')

    del data
    gc.collect()

    temp = data['advertiserId']*data['campaignId']
    temp.columns=['advertiserId_campaignId']
    temp.to_csv(feature_data_path+ "/cross/" +'advertiserId_campaignId.csv',index=False)

    temp = data['aid']*(data['uid']/1000)
    temp.columns=['aid_uid']
    temp.to_csv(feature_data_path+ "/cross/" +'uid_aid.csv',index=False)




if __name__ == '__main__':
    gen_features()
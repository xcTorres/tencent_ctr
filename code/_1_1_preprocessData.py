# coding: utf-8
import os
import pandas as pd
import gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle

def addAdFeature(data):

    feature_path = raw_data_path+'adFeature.pkl'
    if os.path.exists(feature_path):
        ad_feature = load_pickle(feature_path)
    else:
        ad_feature = pd.read_csv(raw_data_path+'adFeature.csv')
        dump_pickle(ad_feature,feature_path)
    return pd.merge(data,ad_feature,on='aid',how='left')


def addUserFeature(data):

    feature_path = raw_data_path+'userFeature.pkl'
    if os.path.exists(feature_path):
        user_feature = load_pickle(feature_path)
    else:
        user_feature = pd.read_csv(raw_data_path+'userFeature.csv')
        dump_pickle(user_feature,feature_path)
    return pd.merge(data,user_feature,on='uid',how='left')


def csv_pkl(csv_name_without_suffix, protocol=None):
    pkl_path = raw_data_path + csv_name_without_suffix + '.pkl'
    if not os.path.exists(pkl_path):
        print('generating ' + pkl_path)
        data = pd.read_csv(raw_data_path + csv_name_without_suffix + '.csv')
        dump_pickle(data, pkl_path, protocol=protocol)
    else:
        print('found ' + pkl_path)

def run():

    print('-------------------- read data  -------------------------------------')

    train = pd.read_csv(raw_data_path + 'train.csv')
    test = pd.read_csv(raw_data_path + 'test2.csv')
    train.loc[train['label'] == -1, 'label'] = 0
    test['label'] = -1
    dump_pickle(train, raw_data_path + 'train.pkl')
    dump_pickle(test, raw_data_path + 'test2.pkl')


    csv_pkl('adFeature')
    csv_pkl('userFeature', protocol=4)


    if not os.path.exists(feature_data_path):
        os.mkdir(feature_data_path)
    if not os.path.exists(cache_pkl_path):
        os.mkdir(cache_pkl_path)

    train = load_pickle(raw_data_path + 'train.pkl')
    test = load_pickle(raw_data_path + 'test2.pkl')

    data = pd.concat([train, test])
    data = addAdFeature(data)
    data = addUserFeature(data)

    data = data.drop(['appIdAction','appIdInstall','interest3','interest4','kw3','topic3'],axis=1)
    data = data.fillna('-1')

    print('-------------------- save to pickle  -------------------------------------')
    dump_pickle(data, raw_data_path + 'preprocess.pkl', protocol=4)


    del data
    gc.collect()


if __name__ == '__main__':
    run()
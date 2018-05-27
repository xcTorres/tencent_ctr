import pandas as pd
import numpy as np
import gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle


def add_AllFeatures():

    print('-------------------- read data----------------------------')
    data = load_pickle(raw_data_path + 'preprocess.pkl')

    statis_feats = ['uid_adCount','ct_adCount','aid_userCount','adCategoryId_userCount',
                    'creativeId_userCount','LBS_userCount',
                    'aid_LBSCount','adCategoryId_LBSCount','creativeId_LBSCount','ad_mean_age',
                    'creativeId_mean_age','adCategoryId_mean_age']

    for feat in statis_feats:
        gc.collect()
        count = pd.read_csv(feature_data_path + "statis/" + '%s.csv' % (feat))
        feat_1,feat_2 = list(count.columns.values)
        data = data.merge(count, how='left', on=[feat_1])
        data[feat_2].fillna(data[feat_2].mean())


    for feat_1 in ['creativeId', 'aid', 'advertiserId','campaignId','adCategoryId',
                   'productId','productType','education','age']:
        count = pd.read_csv(feature_data_path + "ctr/" + '%s.csv' % (feat_1+"_ctr"))
        data = data.merge(count, how='left', on=feat_1)
        data[feat_1].fillna(data[feat_1].mean())

        print ("concat " + feat_1 + "  over")

    for feat_1, feat_2 in [ ('aid', 'age'),('creativeId', 'age'),('campaignId', 'age'),
                            ('campaignId', 'gender'),
                            ('advertiserId', 'age'),('aid', 'gender'),('creativeId', 'gender'),
                            ('adCategoryId', 'age'), ('productType', 'age'), ('productId', 'gender'),
                            ('adCategoryId', 'gender'), ('advertiserId', 'gender'),
                            ('productType', 'marriageStatus'),('productType', 'gender'),
                            ('adCategoryId', 'education'),('productType', 'education'),
                            ('productType', 'house')
                          ]:
        count = pd.read_csv(feature_data_path + "ctr/" + '%s.csv' % (feat_1 + '_' + feat_2+'ctr'))
        data = data.merge(count, how='left', on=[feat_1, feat_2])
        data[feat_1 + '_' + feat_2 + '_ctr'].fillna(data[feat_1 + '_' + feat_2 + '_ctr'].mean())

        print("concat " + feat_1 +'_' + feat_2+ " over")

    print ( data.describe() )
    dump_pickle(data,raw_data_path+"concat_smoth.pkl",protocol=4)

if __name__=='__main__':
    add_AllFeatures()



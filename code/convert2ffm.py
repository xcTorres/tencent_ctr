import pandas as pd
import numpy as np
import os
import gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle


class FFMFormat:
    def __init__(self,vector_feat,one_hot_feat,continus_feat):
        self.field_index_ = None
        self.feature_index_ = None
        self.vector_feat=vector_feat
        self.one_hot_feat=one_hot_feat
        self.continus_feat=continus_feat


    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            if col in self.one_hot_feat:
                print(col)
                df[col]=df[col].astype('int')
                vals = np.unique(df[col])
                for val in vals:
                    if val==-1: continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in self.vector_feat:
                print(col)
                vals=[]
                for data in df[col].apply(str):
                    if data!="-1":
                        for word in data.strip().split(' '):
                            vals.append(word)
                vals = np.unique(vals)
                for val in vals:
                    if val=="-1": continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []

        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = '{}_{}'.format(col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                # ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], 1))
            elif col in self.vector_feat:
                for word in str(val).split(' '):
                    name = '{}_{}'.format(col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.continus_feat:
                if val!=-1:
                    ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        # val=[]
        # for k,v in self.feature_index_.items():
        #     val.append(v)
        # val.sort()
        # print(val)
        # print(self.field_index_)
        # print(self.feature_index_)
        return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})

def run():
    one_hot_feature=['creativeSize','LBS','age','carrier','consumptionAbility','education','gender','aid',
                     'advertiserId','campaignId', 'creativeId',
                    'adCategoryId', 'productId', 'productType']

    vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2','os','ct','marriageStatus']
    single_ctr = ['creativeId', 'age', 'aid', 'productType', 'advertiserId', 'campaignId', 'adCategoryId', 'productId'
                 ]

    pair_ctr = [('aid', 'age'), ('creativeId', 'age'), ('campaignId', 'age'), ('campaignId', 'gender'),
                ('advertiserId', 'age'), ('aid', 'gender'), ('creativeId', 'gender'),
                ('adCategoryId', 'age'), ('productType', 'age'), ('productId', 'gender'),
                ('adCategoryId', 'gender'), ('advertiserId', 'gender'), ('productType', 'marriageStatus'),
                ('productType', 'gender'), ('adCategoryId', 'education'), ('productType', 'education'),
                ('productType', 'house')
                ]

    continues_Features = []
    for feat_1 in single_ctr:
        continues_Features.append(feat_1 + "_ctr")

    for feat_1, feat_2 in pair_ctr:
        continues_Features.append(feat_1 + '_' + feat_2 + '_ctr')


    data = load_pickle(raw_data_path + 'concat_smoth.pkl')
    data.reset_index()

    train_leaf = pd.read_csv(raw_data_path + 'train_leaf_feature.csv')
    test_leaf = pd.read_csv(raw_data_path + 'test_leaf_feature.csv')
    all_leaf = pd.concat([train_leaf, test_leaf], ignore_index=True)
    del train_leaf,test_leaf
    gc.collect()
    print("read finish")

    all_leaf.fillna('-1')
    all_leaf = all_leaf.drop(['aid','uid'],axis=1)
    leafFeature = list(all_leaf.columns.values)
    data = pd.concat([data, all_leaf], axis=1)

    one_hot_feature = one_hot_feature + leafFeature

    def binning(series, bin_num):
        bins = np.linspace(series.min(), series.max(), bin_num)
        labels = [i for i in range(bin_num - 1)]
        out = pd.cut(series, bins=bins, labels=labels).astype(int)
        return out

    for col in continues_Features:
        one_hot_feature.append(col)
        data[col] = binning(data[col], 51)
        # data[col].fillna(24)


    data = data[one_hot_feature+vector_feature]

    tr = FFMFormat(vector_feature,one_hot_feature,[])
    print("start to ffm")
    user_ffm=tr.fit_transform(data)

    del data
    gc.collect()

    print("start to covert to ffm")
    user_ffm.to_csv(raw_data_path+'ffm.csv',index=False)

    train = pd.read_csv(raw_data_path + 'train.csv')

    Y = np.array(train.pop('label'))
    len_train=len(train)

    with open(raw_data_path+'ffm.csv') as fin:
        f_train_out=open(raw_data_path+'train_ffm.csv','w')
        f_test_out = open(raw_data_path+'test_ffm.csv', 'w')
        for (i,line) in enumerate(fin):
            if i<len_train:
                f_train_out.write(str(Y[i])+' '+line)
            else:
                f_test_out.write(line)
        f_train_out.close()
        f_test_out.close()

if __name__=='__main__':
    run()
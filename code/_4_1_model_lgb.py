import numpy as np
import  pandas as pd
import gc
from scipy import sparse
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
import lightgbm as lgb

def base_process():

    data = load_pickle(raw_data_path + "concat_smoth.pkl")
    data.reset_index()
    train = data[data.label != -1]
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    train_y = train.pop('label')

    del data
    gc.collect()

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


    train_x = train[continues_Features].values
    test_x = test[continues_Features].values
    print("continus prepared")

    del train,test
    gc.collect()

    train_onehot  = sparse.load_npz(raw_data_path + 'onehot_train_with_leaf.npz')
    test_onehot = sparse.load_npz(raw_data_path + 'onehot_test_with_leaf.npz')
    print('one-hot prepared !')

    train_x = sparse.hstack((train_x, train_onehot))
    test_x = sparse.hstack((test_x, test_onehot))

    del train_onehot,test_onehot
    gc.collect()
    # sparse.save_npz(raw_data_path+'ctr_afterLeaf_train521.npz', train_x)
    # sparse.save_npz(raw_data_path+'ctr_afterLeaf_test521.npz', test_x)

    LGB_predict(train_x, train_y.values, test_x, res )


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)

    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(result_path+'submission_521_lgb.csv', index=False)

    fea_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
    fea_imp.to_csv(result_path+"importance.csv")


    return clf


if __name__=='__main__':
    base_process()
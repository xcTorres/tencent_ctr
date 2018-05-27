import pandas as pd
from scipy import sparse
import xgboost as xgb
import gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle


def gen_leafFeatures():

    data = load_pickle(raw_data_path + "concat_smoth.pkl")
    print("read finish")

    # statis_feats = ['uid_adCount', 'ct_adCount', 'aid_userCount', 'adCategoryId_userCount',
    #                 'creativeId_userCount', 'LBS_userCount',
    #                 'aid_LBSCount', 'adCategoryId_LBSCount', 'creativeId_LBSCount', 'ad_mean_age',
    #                 'creativeId_mean_age', 'creativeId_mean_age', 'adCategoryId_mean_age']

    single_ctr = ['creativeId', 'age', 'aid', 'productType', 'advertiserId', 'campaignId', 'adCategoryId', 'productId'
                  ]

    pair_ctr = [('aid', 'age'), ('creativeId', 'age'), ('campaignId', 'age'), ('campaignId', 'gender'),
                ('advertiserId', 'age'), ('aid', 'gender'), ('creativeId', 'gender'),
                ('adCategoryId', 'age'), ('productType', 'age'), ('productId', 'gender'),
                ('adCategoryId', 'gender'), ('advertiserId', 'gender'), ('productType', 'marriageStatus'),
                ('productType', 'gender'), ('adCategoryId', 'education'), ('productType', 'education'),
                ('productType', 'house')
                ]

    train = data[data.label != -1]
    print(len(train))
    test = data[data.label == -1]
    print(len(test))
    test = test.drop('label', axis=1)
    train_y = train.pop('label')

    train_id = train[['aid', 'uid']]
    predict_id = test[['aid', 'uid']]

    del data
    gc.collect()

    continues_Features = []
    for feat_1 in single_ctr:
        continues_Features.append(feat_1 + "_ctr")

    for feat_1, feat_2 in pair_ctr:
        continues_Features.append(feat_1 + '_' + feat_2 + '_ctr')

    # for feat in statis_feats:
    #     continues_Features.append(feat)

    train_x = train[continues_Features].values
    test_x = test[continues_Features].values
    print("continus prepared")

    del train, test
    gc.collect()

    train_onehot = sparse.load_npz(raw_data_path + 'onehot_train.npz')
    test_onehot = sparse.load_npz(raw_data_path + 'onehot_test.npz')

    train_x = sparse.hstack((train_x, train_onehot))
    test_x = sparse.hstack((test_x, test_onehot))

    train_x = train_x.tocsc()
    test_x = test_x.tocsc()

    # sparse.save_npz(raw_data_path + 'ctr_beforeLeaf_train520.npz', train_x)
    # sparse.save_npz(raw_data_path + 'ctr_beforeLeaf_test520.npz', test_x)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 7,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'lambda': 1,
        'seed': 2018,
        'eval_metric': 'auc'
    }

    clf = xgb.XGBClassifier(xgb_params)
    clf.fit(train_x, train_y)

    train_new_feature = clf.predict(train_x, pred_leaf=True)
    test_new_feature = clf.predict(test_x, pred_leaf=True)

    train_new_feature1 = pd.DataFrame(train_new_feature)
    test_new_feature1 = pd.DataFrame(test_new_feature)

    train_xgb_leaf = pd.concat([train_id, train_new_feature1], axis=1)
    print(len(train_xgb_leaf))
    test_xgb_leaf = pd.concat([predict_id, test_new_feature1], axis=1)
    print(len(test_xgb_leaf))

    dump_pickle(train_xgb_leaf, raw_data_path + 'train_leaf_feature.pkl', protocol=4)
    dump_pickle(test_xgb_leaf, raw_data_path + 'test_leaf_feature.pkl', protocol=4)


if __name__ == '__main__':
    gen_leafFeatures()
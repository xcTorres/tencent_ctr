import gc
import  pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
def gen_features():

    raw = load_pickle(raw_data_path+"preprocess.pkl")
    raw.reset_index()
    print(raw.shape)
    train_leaf = pd.read_csv(raw_data_path + 'train_leaf_feature.csv')
    test_leaf = pd.read_csv(raw_data_path + 'test_leaf_feature.csv')
    all_leaf = pd.concat([train_leaf, test_leaf], ignore_index=True)
    print(all_leaf.shape)

    del train_leaf,test_leaf
    gc.collect()

    print("read finish")

    all_leaf.fillna('-1')
    all_leaf = all_leaf.drop(['aid','uid'],axis=1)
    leafFeature = list(all_leaf.columns.values)

    data = pd.concat([raw, all_leaf], axis=1)

    del raw
    gc.collect()

    print (leafFeature)

    data = data[leafFeature+['label']]

    print('start!')
    for feature in (leafFeature):
        print (feature)
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    test = data[data.label == -1]
    test = test.drop('label', axis=1)
    train_y = train.pop('label')

    train_x  = sparse.load_npz(raw_data_path + 'onehot_train.npz')
    test_x = sparse.load_npz(raw_data_path + 'onehot_test.npz')
    print('one-hot prepared !')

    oc_encoder = OneHotEncoder()
    for feature in (leafFeature):
        print (feature)
        gc.collect()
        oc_encoder.fit(data[feature].values.reshape(-1, 1))
        train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
        test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))

        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('finished!')

    del data,train,test
    gc.collect()

    sparse.save_npz(raw_data_path+'onehot_train_with_leaf.npz', train_x)
    sparse.save_npz(raw_data_path+'onehot_test_with_leaf.npz', test_x)

if __name__== '__main__':
    gen_features()
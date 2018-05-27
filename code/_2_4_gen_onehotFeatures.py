import gc
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
def gen_features():

    data = load_pickle(raw_data_path+"preprocess.pkl")
    print ("read finish")

    one_hot_feature=[ 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
                     'adCategoryId', 'productId', 'productType']
    vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

    data = data[one_hot_feature + vector_feature + ['label','creativeSize']]

    print('start!')
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    test = data[data.label == -1]
    test = test.drop('label', axis=1)
    train_y = train.pop('label')


    train_x = train[['creativeSize']].values
    test_x = test[['creativeSize']].values

    oc_encoder = OneHotEncoder()
    for feature in one_hot_feature:
        print (feature)
        gc.collect()
        oc_encoder.fit(data[feature].values.reshape(-1, 1))
        train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
        test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))

        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    ct_encoder = CountVectorizer(min_df=0.0009)
    for feature in vector_feature:
        gc.collect()
        print(feature)
        ct_encoder.fit(data[feature])
        train_a = ct_encoder.transform(train[feature])
        test_a = ct_encoder.transform(test[feature])

        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')

    del data,train,test
    gc.collect()

    sparse.save_npz(raw_data_path+'onehot_train.npz', train_x)
    sparse.save_npz(raw_data_path+'onehot_test.npz', test_x)

if __name__== '__main__':
    gen_features()
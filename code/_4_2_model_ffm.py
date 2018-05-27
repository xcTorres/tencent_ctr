import gc,os
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
import xlearn as xl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def FFM_predict():

    train = pd.read_csv(raw_data_path + 'train_ffm.csv')
    train_ffm_1,val_ffm_1 = train_test_split(train,test_size=0.2,random_state=2018)
    del train
    gc.collect()
    train_ffm_1.to_csv(raw_data_path+'train_ffm_1.csv',index=False)
    val_ffm_1.to_csv(raw_data_path + 'val_ffm_1.csv', index=False)
    del train_ffm_1,val_ffm_1
    gc.collect()

    ffm_model = xl.create_ffm()
    ffm_model.setTrain(raw_data_path + 'train_ffm_1.csv')
    ffm_model.setValidate(raw_data_path+"val_ffm_1.csv")
    ffm_model.setTest(raw_data_path + 'test_ffm.csv')
    ffm_model.setSigmoid()
    param = {'task': 'binary', 'lr': 0.1, 'lambda': 0.00002, 'metric': 'auc', 'opt': 'ftrl', 'epoch': 25, 'k': 8}

    # ffm_model.cv(param)
    ffm_model.fit(param, "./model.out")
    ffm_model.predict("./model.out", "./output.txt")
    test = pd.read_csv(raw_data_path + 'test2.csv')
    sub = pd.DataFrame()
    sub['aid'] = test['aid']
    sub['uid'] = test['uid']
    sub['score'] = np.loadtxt("./output.txt")
    sub.to_csv(result_path+'submission_ffm.csv', index=False)



if __name__=='__main__':
    FFM_predict()
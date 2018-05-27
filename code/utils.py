import pickle

#file_path

raw_data_path = '../data/'
feature_data_path = '../feature/'
cache_pkl_path = '../cache_pkl/'
result_path = '../result/'

sample_feature_data_path = '../sample_feature/'

def load_pickle(path):
    return pickle.load(open(path,'rb'))
def dump_pickle(obj, path, protocol=None,):
    pickle.dump(obj,open(path,'wb'),protocol=protocol)
import pandas as pd
import os,gc
from utils import raw_data_path, feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
import numpy as np
import scipy.special as special
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)




def gen_features():

    data = load_pickle(raw_data_path + 'preprocess.pkl')

    train = data[data.label != -1]

    del data
    gc.collect()

    for feat_1 in ['creativeId', 'aid', 'advertiserId','campaignId','adCategoryId',
                   'productId','productType','education','age','marriageStatus','carrier']:
        gc.collect()
        print (feat_1)
        temp = train[[feat_1, 'label']]
        count = temp.groupby([feat_1]).apply(lambda x: x['label'].count()).reset_index(
            name=feat_1 + '_all')
        count1 = temp.groupby([feat_1]).apply(lambda x: x['label'].sum()).reset_index(
            name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        bs = BayesianSmoothing(1, 1)
        bs.update(count[feat_1 + '_all'].values, count[feat_1 + '_1'].values, 1000, 0.001)
        count[feat_1 + '_ctr'] = (count[feat_1 + '_1'] + bs.alpha) / (
                count[feat_1 + '_all'] + bs.alpha + bs.beta)
        count[feat_1 + '_ctr'] = count[feat_1 + '_ctr'].apply(lambda x: float('%.6f' % x))
        count.drop([feat_1 + '_1', feat_1 + '_all'], axis=1, inplace=True)

        count.to_csv(feature_data_path + "ctr/" +'%s.csv' % (feat_1+"_ctr"), index=False)
        print(feat_1, ' over')



    for feat_1,feat_2 in [('creativeId','LBS'),('creativeId','age'),('creativeId','gender'),
                          ('creativeId','education'),('creativeId','marriageStatus'),
                          ('creativeId','house'),('creativeId','consumptionAbility'),
                          ('aid', 'LBS'), ('aid', 'age'), ('aid', 'gender'),
                          ('aid', 'education'), ('aid', 'marriageStatus'),
                          ('aid', 'house'),('aid','consumptionAbility'),
                          ('advertiserId', 'LBS'), ('advertiserId', 'age'), ('advertiserId', 'gender'),
                          ('advertiserId', 'education'), ('advertiserId', 'marriageStatus'),
                          ('advertiserId', 'house'),('advertiserId','consumptionAbility'),
                          ('campaignId', 'LBS'), ('campaignId', 'age'), ('campaignId', 'gender'),
                          ('campaignId', 'education'), ('campaignId', 'marriageStatus'),
                          ('campaignId', 'house'),('campaignId','consumptionAbility'),
                          ('adCategoryId', 'LBS'), ('adCategoryId', 'age'), ('adCategoryId', 'gender'),
                          ('adCategoryId', 'education'), ('adCategoryId', 'marriageStatus'),
                          ('adCategoryId', 'house'),('adCategoryId','consumptionAbility'),
                          ('productId', 'LBS'), ('productId', 'age'), ('productId', 'gender'),
                          ('productId', 'education'), ('productId', 'marriageStatus'),
                          ('productId', 'house'), ('productId', 'consumptionAbility'),
                          ('productType', 'LBS'), ('productType', 'age'), ('adCategoryId', 'gender'),
                          ('productType', 'education'), ('productType', 'marriageStatus'),
                          ('productType', 'house'), ('productType', 'consumptionAbility')
                          ]:
        gc.collect()
        print (feat_1,feat_2)
        if os.path.exists(feature_data_path + "ctr/" +'%s.csv' % (feat_1+'_'+feat_2 + "ctr")):
            print('found  ' + feature_data_path + "ctr/"+'%s.csv' % (feat_1+'_'+feat_2 ) )
        else:
            print('generate ' + feature_data_path +  "ctr/" + '%s.csv' % (feat_1+'_'+feat_2 ) )

            temp = train[[feat_1, feat_2, 'label']]
            count = temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'].count()).reset_index(
                name=feat_1 + '_' + feat_2 + '_all')
            count1 = temp.groupby([feat_1, feat_2]).apply(lambda x: x['label'].sum()).reset_index(
                name=feat_1 + '_' + feat_2 + '_1')
            count[feat_1 + '_' + feat_2 + '_1'] = count1[feat_1 + '_' + feat_2 + '_1']

            bs = BayesianSmoothing(1, 1)
            bs.update(count[feat_1 + '_' + feat_2 + '_all'].values, count[feat_1 + '_' + feat_2 + '_1'].values, 1000,
                      0.001)
            count[feat_1 + '_' + feat_2 + '_ctr'] = (count[feat_1 + '_' + feat_2 + '_1'] + bs.alpha) / (
                    count[feat_1 + '_' + feat_2 + '_all'] + bs.alpha + bs.beta)
            count[feat_1 + '_' + feat_2 + '_ctr'] = count[feat_1 + '_' + feat_2 + '_ctr'].apply(
                lambda x: float('%.6f' % x))

            count.drop([feat_1 + '_' + feat_2 + '_1', feat_1 + '_' + feat_2 + '_all'], axis=1, inplace=True)
            count.to_csv(feature_data_path + "ctr/" + '%s.csv' % (feat_1+'_'+feat_2 + "ctr"), index=False)
            print(feat_1, feat_2, ' over')


if __name__ == '__main__':
    gen_features()
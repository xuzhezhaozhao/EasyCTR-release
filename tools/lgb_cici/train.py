# coding: utf-8
import argparse
import pandas as pd
import feature as ft
import lightgbm as lgb
import columns as cls
import numpy as np
import time


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_bin': 255,
    'learning_rate': 0.1,
    'num_leaves': 16,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_fraction': 0.8,
    'min_data_in_leaf': 21,
     #'min_sum_hessian_in_leaf': 3.0,
    'num_thread': 24,
    'header': True
}

class Gold:
    def __init__(self, param, args):
        self.param = param
        self.root_dir = args.root_dir
        self.version = args.version

    def build_model(self):
        print("haha")
        #print(type(cls.dtypes))
        #for key in cls.dtypes:
            #if cls.dtypes[key] == 'numpy.bytes_':
                #cls.dtypes[key] = 'numpy.string_'
        #print(cls.dtypes)
        source_data_train = pd.read_csv(self.root_dir + '/data/data/train_072508.csv.clean', header=None, sep='\t', dtype=cls.dtypes, names=cls.columns)
        source_data_test = pd.read_csv(self.root_dir + '/data/data/test_072508.csv.clean', header=None, sep='\t', dtype=cls.dtypes, names=cls.columns)

        #source_data_train = source_data_train.fillna(0)
        #source_data_test = source_data_test.fillna(0)
        
        print(source_data_train.shape[0])
        #print(source_data_train)
        X_train, y_train = ft.get_train_data(source_data_train)
        X_test, y_test = ft.get_train_data(source_data_test)
        print(X_test.loc[0])
        
        category = cls.category_columns

        lgb_train = lgb.Dataset(data=X_train, label=y_train, categorical_feature=category)

        lgb_eval = lgb.Dataset(data=X_test, label=y_test, reference=lgb_train, categorical_feature=category)

        booster = lgb.train(
            self.param,
            lgb_train,
            valid_sets=[lgb_eval],
            num_boost_round=1000,
            early_stopping_rounds=100,
            categorical_feature=category)

        self.model= booster
        self.save_feature_import()


        #booster.save_model(self.root_dir + '/data/model/'+self.version+'.LightGBM')

    def save_feature_import(self):
        booster = self.model
        # 放入训练的属性以及分类的特征值
        import_rate = booster.feature_importance(importance_type='gain')
        importance = np.array(import_rate) / sum(import_rate)
        feature_name = booster.feature_name()
        feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
        feature_importance.to_csv(self.root_dir + '/data/feature/feature_importance_072508.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        type=str,
        default='',
        help='root_dir'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='',
        help='version'
    )
   
    FLAGS, unparsed = parser.parse_known_args()
    #FLAGS.data_dir = 'E:/workspace/workspace_py/deep_recommender_system/newuser_lightgbm'
    gold = Gold(param=params, args=FLAGS)
    gold.build_model()

#!usr/bin/env/ python3
import pandas as pd
import numpy as np

class Parser:
    def __init__(self, filename_train, filename_out):
        self.filename_train = filename_train
        self.filename_out = filename_out
        self.nan_columns = []
        # raw data
        self.data_raw_set = None
        self.data_raw_up = None
        # drop nan
        self.data_set = None
        self.data_up = None
        self.mean, self.var = None, None
        self.train_X, self.valid_X, self.test_X = None, None, None
        self.train_y, self.valid_y, self.test_y = None, None, None
        self.train_w, self.valid_w, self.test_w = None, None, None
        self.out_X = None

        self.out_id = None

    def load_train_file(self):
        self.data_raw_set = pd.read_csv(self.filename_train)

    def load_out_file(self):
        self.data_raw_up = pd.read_csv(self.filename_out)
        self.out_id = self.data_raw_up.id

    def generate_norm_data(self, seed = 10, train_per = 0.8, test_era_amount = 1, era = 20, eps = 1e-8):
        # drop nan first
        self.nan_columns = []
        for i in np.where(np.isnan(self.data_raw_set.head(1)))[1]:
            self.nan_columns.append(self.data_raw_set.keys()[i])
        self.data_set = self.data_raw_set.drop(self.nan_columns, axis=1)
        self.data_up = self.data_raw_up.drop(self.nan_columns, axis=1)
        # split train_val test
        np.random.seed(seed)
        choice = np.random.permutation(20)
        test = self.data_set[self.data_set.era.isin(choice[0:test_era_amount])].copy()
        train_val = self.data_set[self.data_set.era.isin(choice[test_era_amount:])].copy()
        # random
        train_val = train_val.iloc[np.random.permutation(train_val.shape[0]),:]
        train_val_X = train_val.iloc[:, 1:-6]
        train_val_X['group1'] = train_val.group1
        train_val_X['group2'] = train_val.group2
        train_val_X['code_id'] = train_val.code_id
        train_val_X = train_val_X.as_matrix()
        self.mean = np.mean(train_val_X, axis=0)
        self.var = np.var(train_val_X, axis=0)
        train_val_X = (train_val_X-self.mean) / (np.sqrt(self.var)+eps)
        train_val_w = train_val.weight.as_matrix()
        train_val_y = train_val.label.as_matrix()
        # 
        choice = int(np.round(train_per*train_val.shape[0]))
        self.train_X, self.valid_X = train_val_X[0:choice,:], train_val_X[choice:,:]
        self.train_y, self.valid_y = train_val_y[0:choice], train_val_y[choice:]
        self.train_w, self.valid_w = train_val_w[0:choice], train_val_w[choice:]
        # 
        self.test_X = test.iloc[:, 1:-6]
        self.test_X['group1'] = test.group1
        self.test_X['group2'] = test.group2
        self.test_X['code_id'] = test.code_id
        self.test_X = self.test_X.as_matrix()
        self.test_y = test.label.as_matrix()
        self.test_w = test.weight.as_matrix()
        self.test_X = (self.test_X-self.mean) / (np.sqrt(self.var)+eps)
        # 
        self.out_X = (self.data_up.iloc[:,1:].as_matrix()-self.mean) / (np.sqrt(self.var)+eps)

    def get_train_var_test_data(self):
        return self.train_X, self.train_y, self.valid_X, self.valid_y, self.test_X, self.test_y
    
    def get_w(self):
        return self.train_w, self.valid_w, self.test.weight

    def get_out_data(self):
        return self.out_X

    def save_out_y(self, y, filename='output.csv', isPrint = True):
        df = pd.DataFrame()
        y = y.reshape((-1, 1))
        df['id'] = self.out_id
        df['proba'] = 0
        df.loc[:,'proba'] = y
        df.set_index('id', inplace = True)
        df.to_csv(filename)
        # print('max out:%0.3f, min out:%0.3f',np.max(outnumpy2), np.min(outnumpy2))

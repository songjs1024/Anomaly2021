import pandas as pd
from sklearn.preprocessing import StandardScaler

class data_loader(object):

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/data.csv')
        data = data.values[:, 1:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        


    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        


    def __get   (self):

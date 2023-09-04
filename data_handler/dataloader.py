import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence


class KrugerLoader(Sequence):

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/concat_data.csv')
        data = data.values[:, 1:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
    

    def __get   (self):
        '''
        인덱스 메인에서 불러오는거 확인
        
        '''

class PunkerLoader(Sequence):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/concat_data.csv')
        data = data.values[:, 1:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
    

    def __get   (self):
        '''
        인덱스 메인에서 불러오는거 확인
        
        '''
    
    
    
    
def get_loader_segment(data_path, batch_size, win_size=1000, step=100, mode='train', dataset='vibx'):
    if (dataset == 'kruger'):
        dataset = KrugerLoader(data_path, win_size, step, mode)
    elif (dataset == 'punker'):
        dataset = PunkerLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True
        
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


        

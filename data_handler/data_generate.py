import os
import pandas as pd

class generater(object):
  
  def __init__(self,raw_path):
    self.raw_path = raw_path
    self.combine_data = data
    

  def combine(self):
    appended_data = []
    for file in glob.glob(raw_path+'*.csv'):
      tmp_raw = pd.read_csv(file,skiprows=8,usecols=['12000'])
      appended_data.append(tmp_raw)
    appended_data = pd.concat(appended_data)
    if not os.path.exists(raw_path+'train_data'): 
      os.mkdir(raw_path+'train_data')
    appended_data.to_csv(raw_path+'train_data'+'/train_data.csv',sep=',')    

      
      

    

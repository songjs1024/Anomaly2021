import os
import argparse






if __name__ == '__main__':
    print('------------ Options -------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
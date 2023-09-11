import tensorflow as tf
from tensorflow import keras
from keras import losses 
from model.AnomalyTransformer import AnomalyTransformer
from keras import optimizers as opt

def my_kl_loss(p, q):
    res = p * (tf.log(p + 0.0001) - tf.log(q + 0.0001))
    return tf.reduce_mean(tf.reduce_sum(res,-1), 1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


'''
class  EarlyStopping:
    def __init__(self,patience = 7, verbose = False, min_delta = 0):
        tf.keras.callbacks.EarlyStopping(
        self.monitor='val_loss',
        self.min_delta= min_delta,
        self.patience= patience,
        self.verbose= verbose,
        self.mode= 'auto',
        self.modebaseline= None,
        self.restore_best_weights= True,
        self.start_from_epoch=0
        )

    def __call__(self,):
        score = -self.monitor
        if self.restore_best_weights is None:
            score = -val_loss

'''

class Solver(object):

  def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)


  self.build_model()
  self.device = torch.device("cuda:0" if tf.test.is_built_with_cuda() else "cpu")
  self.criterion = losses.MeanSquaredError()

  def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = opt.Adam(self.model.parameters(), learning_rate=self.lr)

  def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
              series_loss += (tf.reduce_mean(my_kl_loss(series[u], (
                prior[u] / tf.expand_dims(tf.reduce_sum(prior[u],-1),-1).repeat(1, 1, 1,self.win_size)).detach()))
                + tf.reduce_mean(my_kl_loss((prior[u] /tf.expand_dims(tf.reduce_sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach(),series[u])))
              prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / tf.expand_dims(tf.reduce_sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + tf.reduce_mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / tf.expand_dims(tf.reduce_sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())                              
                                                                                                                      
  return np.average(loss_1), np.average(loss_2)  





import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optimizers
import torchvision


class LSTM(nn.Module):
    def __init__(self, hidden_dim, data_dim, dropout_rate, n_class, **kwargs):
        super().__init__()
        self.l1 = nn.LSTM(data_dim, hidden_dim, 
                          batch_first=True)
        self.d1 = nn.Dropout(dropout_rate)
        self.l2 = nn.LSTM(hidden_dim, hidden_dim,  
                          batch_first=True)
        self.d2 = nn.Dropout(dropout_rate)
        self.l3 = nn.Linear(hidden_dim, n_class)
        self.l4 = nn.LogSoftmax(dim=1)

        torch.manual_seed(123)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)#シグモイド関数に使える初期値xavierの初期値 l0が１つ目の隠れ層の重みとして設定される　nn.RNNはnum_layersを引数として指定することができるのでそうするとl0,l1,l2と重みを設定できる
        torch.manual_seed(123)
        nn.init.orthogonal_(self.l1.weight_hh_l0)#直行行列を用いた初期値でオーバーフローしない様に
        torch.manual_seed(123)
        nn.init.xavier_normal_(self.l2.weight_ih_l0)
        torch.manual_seed(123)
        nn.init.orthogonal_(self.l2.weight_hh_l0)
        
        
    def forward(self, x, layer_num):
        h, _ = self.l1(x)
        x = self.d1(h)
        if layer_num == 2:
            h, _ = self.l2(x)
            x = self.d2(h)
        y1 = self.l3(x[:, -1])
        y2 = self.l4(y1)
        
        return x, y1, y2

def loss_opt(model,weights,lr, **kwargs):
    criterion = nn.NLLLoss(weight=weights)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=lr,
                                betas=(0.9, 0.999), amsgrad=True)
    return criterion, optimizer

def train_step(x, t, model, criterion, optimizer, phase, device, layer_num, **kwargs):
    x = x.float().to(device=device)
    t = t.to(device=device, dtype=torch.int64)

    optimizer.zero_grad()#勾配の初期化
    
    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(x, layer_num)
        loss = criterion(outputs[2], t)
        
        if phase == 'train':
            loss.backward()#勾配計算
            optimizer.step()#パラメータの更新
    
    return loss, outputs


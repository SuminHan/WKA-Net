import torch
import torch.nn as nn
import random
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class GFC(nn.Module):  # is_training: self.training
    def __init__(self, input_dims, units, activations, bn, bn_decay, device, use_bias=True):
        super(GFC, self).__init__()
        self.input_dims = input_dims
        self.units = units
        self.activations = activations
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.use_bias = use_bias
        self.layers = self._init_layers()

    def _init_layers(self):
        ret = nn.Sequential()
        units, activations = self.units, self.activations
        if isinstance(units, int):
            units, activations = [units], [activations]
        elif isinstance(self.units, tuple):
            units, activations = list(units), list(activations)
        assert type(units) == list
        index = 1
        input_dims = self.input_dims
        for num_unit, activation in zip(units, activations):
            if self.use_bias:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=True)
                nn.init.constant_(basic_conv2d.bias, 0)
            else:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=False)
            nn.init.xavier_normal_(basic_conv2d.weight)
            ret.add_module('conv2d' + str(index), basic_conv2d)
            if activation is not None:
                if self.bn:
                    decay = self.bn_decay if self.bn_decay is not None else 0.1
                    basic_batch_norm = nn.BatchNorm2d(num_unit, eps=1e-3, momentum=decay)
                    ret.add_module('batch_norm' + str(index), basic_batch_norm)
                ret.add_module('activation' + str(index), activation())
            input_dims = num_unit
            index += 1
        return ret

    def forward(self, x):
        # x: (N, H, W, C)
        x = x.transpose(1, 3).transpose(2, 3)  # x: (N, C, H, W)
        x = self.layers(x)
        x = x.transpose(2, 3).transpose(1, 3)  # x: (N, H, W, C)
        return x

class SGatedFusion(nn.Module):
    def __init__(self, D, bn, bn_decay, device):
        super(SGatedFusion, self).__init__()
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.output_fc = GFC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, XS, XT):
        '''
        gated fusion
        HS:     (batch_size, num_step, num_nodes, D)
        HT:     (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, XS), torch.multiply(1 - z, XT))
        H = self.output_fc(H)
        return H


class RNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = 1 # self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.zembedding_dim = 64

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.rnn_type = config.get('rnn_type', 'RNN')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.livepop = config.get('livepop')
        self.stemb = config.get('stemb')

        self.T = 24
        self.add_day_in_week = True
        self.D = int(config.get('rnn_units', 64))
        if self.stemb and self.livepop:
            # self.input_size = self.num_nodes * self.feature_dim + self.zembedding_dim + self.T + 7
            self.input_size = self.num_nodes * self.feature_dim + self.D
        elif self.livepop:
            self.input_size = self.num_nodes * self.feature_dim + self.D
            # self.input_size = self.num_nodes * self.feature_dim + self.zembedding_dim
        elif self.stemb:
            self.input_size = self.num_nodes * self.feature_dim + self.D
            # self.input_size = self.num_nodes * self.feature_dim + self.T + 7
        else:
            self.input_size = self.num_nodes * self.feature_dim

            
        self.input_dim = self.D #self.feature_dim
        # print(f"feature_dim = {self.input_dim}")
        
        self.livepop = config.get('livepop')
        self.stemb = config.get('stemb')
        self.rnn_units = int(config.get('rnn_units', 64))
        self.bn = True
        self.bn_decay = 0.1
        self.T = 24 #self.data_feature.get('points_per_hour', 12) * 24  # points_per_data
        self.add_day_in_week = True #self.data_feature.get('add_day_in_week', False)
        self.input_fc = GFC(input_dims=1, units=[self.D, self.D], activations=[nn.ReLU, None],
                           bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        # self.SE_fc = GFC(input_dims=self.num_nodes, units=[self.D, self.D],
        #                 activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.TE_fc = GFC(input_dims=7 + self.T if self.add_day_in_week else self.T, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.ZE_fc = GFC(input_dims=64, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)

        self._logger.info('You select rnn_type {} in RNN!'.format(self.rnn_type))
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.num_nodes * self.output_dim)

    def forward(self, batch):
        x_all = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y_all = batch['y']  # (batch_size, out_length, num_nodes, feature_dim)
        ZE = batch['Z']  # (batch_size, input_length+out_length, zemb_dim)
        index = -8 if self.add_day_in_week else -1
        TE = torch.cat((x_all[:, :, :, index:], y_all[:, :, :, index:]), dim=1)
        _timeofday = TE[:, :, :, 0:1]
        _timeofday = torch.round(_timeofday * self.T)
        _timeofday = _timeofday.to(int)  # (batch_size, input_length+output_length, num_nodes, 1)
        _timeofday = _timeofday[:, :, 0, :]  # (batch_size, input_length+output_length, 1)
        timeofday = torch.zeros((_timeofday.size(0), _timeofday.size(1), self.T), device=self.device).long()
        timeofday.scatter_(dim=2, index=_timeofday.long(), src=torch.ones(timeofday.shape, device=self.device).long())
        if self.add_day_in_week:
            _dayofweek = TE[:, :, :, 1:]
            _dayofweek = _dayofweek.to(int)  # (batch_size, input_length+output_length, num_nodes, 7)
            dayofweek = _dayofweek[:, :, 0, :]  # (batch_size, input_length+output_length, 7)
            TE = torch.cat((dayofweek, timeofday), dim=2).type(torch.FloatTensor)
        else:
            TE = timeofday.type(torch.FloatTensor)
        TE = TE.to(device=self.device)  # (batch_size, input_length+output_length, 7+T or T)
        # TE # (batch_size, input_length+output_length, 7+T or T)
        
        if self.livepop and self.stemb:
            TE = TE.unsqueeze(2)
            TE = self.TE_fc(TE)
            ZE = ZE.unsqueeze(2)
            ZE = self.ZE_fc(ZE)
            zinfo = (TE + ZE)  #self.sgated_fusion2(self.sgated_fusion(SE, TE), ZE)
            zinfo = zinfo.squeeze(2)
            zinfo = zinfo.permute(1, 0, 2, ) # # [input_window+output_window, batch_size, embedding_dim]
        elif self.livepop: 
            ZE = ZE.unsqueeze(2)
            zinfo = self.ZE_fc(ZE)
            zinfo = zinfo.squeeze(2)
            zinfo = zinfo.permute(1, 0, 2, ) # # [input_window+output_window, batch_size, embedding_dim]
        elif self.stemb:
            TE = TE.unsqueeze(2)
            zinfo = self.TE_fc(TE)
            zinfo = zinfo.squeeze(2)
            zinfo = zinfo.permute(1, 0, 2, ) # # [input_window+output_window, batch_size, embedding_dim]
            

        src = batch['X'][..., 0:1].clone()  # [batch_size, input_window, num_nodes, feature_dim]
        target = batch['y'][..., 0:1]  # [batch_size, output_window, num_nodes, feature_dim]
        
        # if self.stemb and self.livepop:
        #     zinfo = torch.cat((TE, batch['Z']), -1)
        # elif self.livepop:
        #     zinfo = batch['Z']  # [batch_size, input_window+output_window, embedding_dim]
        # elif self.stemb:
        #     zinfo = TE

        src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, output_dim]



        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        # src = [self.input_window, batch_size, self.num_nodes * self.feature_dim]
        outputs = []
        for i in range(self.output_window):
            # src: [input_window, batch_size, num_nodes * feature_dim]
            

            if self.livepop or self.stemb:
                out, _ = self.rnn(torch.cat((src, zinfo[i:i+self.input_window, ...]), -1))
            else:
                out, _ = self.rnn(src)
            # out: [input_window, batch_size, hidden_size * num_directions]
            out = self.fc(out[-1])
            # out: [batch_size, num_nodes * output_dim]
            out = out.reshape(batch_size, self.num_nodes, self.output_dim)
            # out: [batch_size, num_nodes, output_dim]
            outputs.append(out.clone())
            if self.output_dim < self.feature_dim:  # output_dim可能小于feature_dim
                out = torch.cat([out, target[i, :, :, self.output_dim:]], dim=-1)
            # out: [batch_size, num_nodes, feature_dim]
            if self.training and random.random() < self.teacher_forcing_ratio:
                src = torch.cat((src[1:, :, :], target[i].reshape(
                    batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
            else:
                src = torch.cat((src[1:, :, :], out.reshape(
                    batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
        outputs = torch.stack(outputs)
        # outputs = [output_window, batch_size, num_nodes, output_dim]
        return outputs.permute(1, 0, 2, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

from logging import getLogger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("bqthd,bkthd->bqkth", [queries, keys])

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=2)

        out = torch.einsum("bqkth,bkthd->bqthd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("bnqhd,bnkhd->bnqkh", [queries, keys])

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_mx):
        support = torch.einsum("bnd, dh->bnh", [x, self.weight])
        output = torch.einsum("mn,bnh->bmh", [adj_mx, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, device=device)
        self.gc2 = GraphConvolution(nhid, nclass, device=device)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj_mx):
        x = F.relu(self.gc1(x, adj_mx))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj_mx)
        return F.log_softmax(x, dim=2)


class STransformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=2,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.adj_mx = torch.FloatTensor(adj_mx).to(device)
        self.D_S = nn.Parameter(torch.FloatTensor(adj_mx).to(device))
        self.embed_linear = nn.Linear(adj_mx.shape[0], embed_dim)

        self.attention = SSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.gcn = GCN(embed_dim, embed_dim * 2, embed_dim, dropout_rate, device=device)
        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fs = nn.Linear(embed_dim, embed_dim)
        self.fg = nn.Linear(embed_dim, embed_dim)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape
        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(batch_size, input_windows, num_nodes, embed_dim)
        D_S = D_S.permute(0, 2, 1, 3)

        X_G = torch.Tensor(query.shape[0], query.shape[1], 0, query.shape[3]).to(self.device)
        self.adj_mx = self.adj_mx.unsqueeze(0).unsqueeze(0)
        self.adj_mx = self.norm_adj(self.adj_mx)
        self.adj_mx = self.adj_mx.squeeze(0).squeeze(0)

        for t in range(query.shape[2]):
            o = self.gcn(query[:, :, t, :], self.adj_mx)
            o = o.unsqueeze(2)
            X_G = torch.cat((X_G, o), dim=2)

        query = query + D_S
        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout_layer(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g * U_S + (1 - g) * X_G

        return out


class TTransformer(nn.Module):
    def __init__(self, TG_per_day=228, embed_dim=64, num_heads=2,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.temporal_embedding = nn.Embedding(TG_per_day, embed_dim)

        self.attention = TSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape

        D_T = self.temporal_embedding(torch.arange(0, input_windows).to(self.device))
        D_T = D_T.expand(batch_size, num_nodes, input_windows, embed_dim)

        query = query + D_T

        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout_layer(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.STransformer = STransformer(
            adj_mx, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.TTransformer = TTransformer(
            TG_per_day=TG_per_day, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        x1 = self.norm1(self.STransformer(value, key, query) + query)
        x2 = self.dropout_layer(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        return x2


class Encoder(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerBlock(
                adj_mx, embed_dim=embed_dim, num_heads=num_heads, TG_per_day=TG_per_day,
                forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device
            )
            for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dropout_layer(x)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Transformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.encoder = Encoder(
            adj_mx, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, TG_per_day=TG_per_day,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device,
        )

    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src



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

class WKSTTN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx', 1)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        # self.len_row = self.data_feature.get('len_row', 1)
        # self.len_column = self.data_feature.get('len_column', 1)

        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))

        self.embed_dim = config.get('embed_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 2)
        self.TG_per_day = config.get('TG_in_one_day', 288)  # number of time intevals per day
        self.forward_expansion = config.get('forward_expansion', 4)
        self.dropout_rate = config.get('dropout_rate', 0)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.conv1 = nn.Conv2d(1, self.embed_dim, 1) # nn.Conv2d(self.feature_dim, self.embed_dim, 1)
        self.transformer = Transformer(
            self.adj_mx, embed_dim=self.embed_dim, num_layers=self.num_layers, num_heads=self.num_heads,
            TG_per_day=self.TG_per_day, forward_expansion=self.forward_expansion, dropout_rate=self.dropout_rate,
            device=self.device,
        )
        self.conv2 = nn.Conv2d(self.input_window, self.output_window, 1)
        self.conv3 = nn.Conv2d(self.embed_dim, self.output_dim, 1)
        self.act_layer = nn.ReLU()

        
        self.livepop = config.get('livepop')
        self.stemb = config.get('stemb')
        self.bn = True
        self.bn_decay = 0.1
        self.D = int(config.get('embed_dim', 64))
        self.T = 24 #self.data_feature.get('points_per_hour', 12) * 24  # points_per_data
        self.add_day_in_week = True #self.data_feature.get('add_day_in_week', False)
        self.input_fc = GFC(input_dims=1, units=[self.D, self.D], activations=[nn.ReLU, None],
                           bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.SE_fc = GFC(input_dims=self.num_nodes, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.TE_fc = GFC(input_dims=7 + self.T if self.add_day_in_week else self.T, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.ZE_fc = GFC(input_dims=64, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.STE_fc = GFC(input_dims=64, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.SE = torch.eye(self.num_nodes) # self.SE = nn.Parameter(torch.randn(self.num_nodes, self.D), requires_grad=True)
        self.sgated_fusion = SGatedFusion(D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.sgated_fusion2 = SGatedFusion(D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, batch):
        x_all = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y_all = batch['y']  # (batch_size, out_length, num_nodes, feature_dim)
        ZE = batch['Z']  # (batch_size, input_length+out_length, zemb_dim)

        # print('x_all.shape, y_all.shape, ZE.shape', x_all.shape, y_all.shape, ZE.shape)

        batch_size, _, num_nodes, _ = x_all.shape
        index = -8 if self.add_day_in_week else -1
        SE = self.SE.to(device=self.device)
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.SE_fc(SE)
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
        TE = TE.unsqueeze(2).to(device=self.device)  # (batch_size, input_length+output_length, 1, 7+T or T)
        
        if self.livepop and self.stemb:
            TE = self.TE_fc(TE)
            ZE = ZE.unsqueeze(2)
            ZE = self.ZE_fc(ZE)
            ste = self.STE_fc(SE + TE + ZE)  #self.sgated_fusion2(self.sgated_fusion(SE, TE), ZE)
        elif self.livepop: 
            ZE = ZE.unsqueeze(2)
            ZE = self.ZE_fc(ZE)
            ste = self.STE_fc(SE + ZE) #self.sgated_fusion2(self.sgated_fusion(SE, TE), ZE)
        elif self.stemb:
            TE = self.TE_fc(TE)
            ste = self.STE_fc(SE + TE) #self.sgated_fusion(SE, TE)
        else:
            ste = torch.zeros((batch_size, self.input_window + self.output_window, self.num_nodes, self.D), device=self.device)

            
        # ste = ste.permute(1, 0, 2, 3)  # (total_window, batch_size, num_nodes, D)
        # ste = ste.view(self.input_window+self.output_window, batch_size, num_nodes * self.D).to(self.device)
        ste_p = ste[:, :self.input_window, ...]  # (input_window, batch_size, num_nodes * D)
        ste_q = ste[:, self.input_window:, ...]  # (output_window, batch_size, num_nodes * D)


        inputs = batch['X'][..., 0:1]
        inputs = inputs.permute(0, 3, 2, 1)
        ste_p = ste_p.permute(0, 3, 2, 1)
        # print('inputs.shape, ste_p.shape', inputs.shape, ste_p.shape)
        input_transformer = self.conv1(inputs) + ste_p
        input_transformer = input_transformer.permute(0, 2, 3, 1)

        ste_q = ste_q.permute(0, 3, 2, 1).permute(0, 2, 3, 1)
        output_transformer = self.transformer(input_transformer) + ste_q
        output_transformer = output_transformer.permute(0, 2, 1, 3)

        out = self.act_layer(self.conv2(output_transformer))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.permute(0, 3, 2, 1)
        return out

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)

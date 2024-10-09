import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, supports, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs_and_state.size(2)  # =total_arg_size

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])


class FC(nn.Module):
    def __init__(self, num_nodes, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        # Reshape input and state to (batch_size * self._num_nodes, input_dim/state_dim)
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        # (batch_size * self._num_nodes, input_size(input_dim+state_dim))
        value = torch.sigmoid(torch.matmul(inputs_and_state, self.weight))
        # (batch_size * self._num_nodes, self._output_dim)
        value += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        Args:
            input_dim:
            num_units:
            adj_mx:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        if self._use_gc_for_ru:
            self._fn = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                             input_dim=input_dim, hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        else:
            self._fn = FC(self._num_nodes, self._device, input_dim=input_dim,
                          hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        self._gconv = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                            input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units, bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: (B, num_nodes * input_dim)
            hx: (B, num_nodes * rnn_units)

        Returns:
            torch.tensor: shape (B, num_nodes * rnn_units)
        """
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx))  # (batch_size, num_nodes * output_size)
        value = torch.reshape(value, (-1, self._num_nodes, output_size))    # (batch_size, num_nodes, output_size)

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)

        c = self._gconv(inputs, r * hx)  # (batch_size, num_nodes * _num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state  # (batch_size, num_nodes * _num_units)


class Seq2SeqAttrs:
    def __init__(self, config, adj_mx):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 2))
        self.rnn_units = int(config.get('rnn_units', 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        # self.input_dim = config.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step, # input_dim -> rnn_units
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = nn.Linear(self.rnn_units, self.rnn_units)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step, # output_dim -> rnn_units
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.rnn_units) # output_dim -> rnn_units
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.rnn_units) # output_dim -> rnn_units
        return output, torch.stack(hidden_states)


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

class WKDCRNN(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        # self.SE = self.data_feature.get('SE') # self.SE = nn.Parameter(torch.randn(self.num_nodes, self.rnn_units), requires_grad=True)
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)
        Seq2SeqAttrs.__init__(self, config, self.adj_mx)
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model = DecoderModel(config, self.adj_mx)
        
        self.device = config.get('device', torch.device('cpu'))

        self.livepop = config.get('livepop')
        self.stemb = config.get('stemb')
        self.rnn_units = int(config.get('rnn_units', 64))
        self.bn = True
        self.bn_decay = 0.1
        self.D = int(config.get('rnn_units', 64))
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
        self.output_fc = GFC(input_dims=self.D, units=[self.D, self.output_dim], activations=[nn.ReLU, None],
                           bn=self.bn, bn_decay=self.bn_decay, device=self.device)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, ste_p):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)
            ste_p: shape (batch_size, input_window, zemb_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t] + ste_p[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)

        return encoder_hidden_state  # 最后一个隐状态

    def decoder(self, encoder_hidden_state, ste_q, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            ste_q: shape (batch_size, output_window, zemb_dim)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.rnn_units), device=self.device) # self.output_dim -> self.rnn_units
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input + ste_q[t], decoder_hidden_state)
            decoder_input = decoder_output     # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            # if self.training and self.use_curriculum_learning:
            #     c = np.random.uniform(0, 1)
            #     if c < self._compute_sampling_threshold(batches_seen):
            #         decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch, batches_seen=None):
        """
        seq2seq forward pass

        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
                batch['Z']: shape (batch_size, input_window+output_window, zembedding_dim) \n
            batches_seen: batches seen till now

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        x_all = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y_all = batch['y']  # (batch_size, out_length, num_nodes, feature_dim)
        ZE = batch['Z']  # (batch_size, input_length+out_length, zemb_dim)

        batch_size, _, num_nodes, _ = x_all.shape
        index = -8 if self.add_day_in_week else -1
        SE = self.SE.to(device=self.device)
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.SE_fc(SE)
        # SE = self.SE #torch.from_numpy(self.SE).to(device=self.device)
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
            ste = self.STE_fc(SE + ZE)  #self.sgated_fusion2(self.sgated_fusion(SE, TE), ZE)
        elif self.stemb:
            TE = self.TE_fc(TE)
            ste = self.STE_fc(SE + TE)   #self.sgated_fusion(SE, TE)
        else:
            ste = torch.zeros((batch_size, self.input_window + self.output_window, self.num_nodes, self.D), device=self.device)

        ste = ste.permute(1, 0, 2, 3)  # (total_window, batch_size, num_nodes, D)
        ste = ste.view(self.input_window+self.output_window, batch_size, num_nodes * self.D).to(self.device)
        ste_p = ste[:self.input_window, ...]  # (input_window, batch_size, num_nodes * D)
        ste_q = ste[self.input_window:, ...]  # (output_window, batch_size, num_nodes * D)


        inputs = self.input_fc(x_all[:, :, :, 0:1]) # batch['X'], D
        labels = self.input_fc(y_all[:, :, :, 0:1]) # batch['y'], D


        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * self.D).to(self.device)
        self._logger.debug("X: {}".format(inputs.size()))  # (input_window, batch_size, num_nodes * input_dim)

        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)  # (output_window, batch_size, num_nodes, rnn_units)
            # labels = labels[..., :self.output_dim].contiguous().view(
            #     self.output_window, batch_size, num_nodes * self.output_dim).to(self.device)
            labels = labels.contiguous().view(
                self.output_window, batch_size, num_nodes * self.D).to(self.device)
            self._logger.debug("y: {}".format(labels.size()))

        encoder_hidden_state = self.encoder(inputs, ste_p)
        # (num_layers, batch_size, self.hidden_state_size)
        self._logger.debug("Encoder complete")
        outputs = self.decoder(encoder_hidden_state, ste_q, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.rnn_units)
        outputs = self.output_fc(outputs).permute(1, 0, 2, 3)
        return outputs

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen)

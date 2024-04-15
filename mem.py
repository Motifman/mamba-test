import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import einops


def redfb_source_init(n, m, remainder_init=False, start=0.5, stop=1):
    if 2 * n > m:
        raise ValueError("2n should not be less than m")
    else:
        if remainder_init is False:
            weight_matrix = torch.zeros((m, n))
            for i in range(n):
                if 2 * i < m:
                    weight_matrix[2 * i][i] = 1
                if 2 * i + 1 < m:
                    weight_matrix[2 * i + 1][i] = 1
        else:
            n_loop = m // (2 * n)
            remainder_part = torch.zeros((m - 2 * n * n_loop, n))
            gain = np.linspace(start, stop, n_loop * n)  # 0.5->1
            source = torch.ones(2, 1)
            source_part = []
            for i in range(n_loop):
                diag = [source * gain[i * n + j] for j in range(n)]
                block_diag = torch.block_diag(*diag)
                source_part.append(block_diag)
            source_part.append(remainder_part)
            weight_matrix = torch.cat(source_part, dim=0)
        return weight_matrix


def redfb_feedback_init(n: int, type: str, ff: bool, start: float, stop: float):
    if n % 2 != 0:
        raise ValueError("n must be even number")
    if type == "pos" and type == "neg":
        raise ValueError("type must be neg or pos")

    weight_hh = torch.zeros((n, n))
    weight_fb_seq = np.linspace(start, stop, int(n / 2))
    for i in range(n):
        if i % 2 == 0:
            weight_hh[i, i + 1] = weight_fb_seq[i // 2]
        else:
            if type == "neg":
                weight_hh[i, i - 1] = weight_fb_seq[i // 2] * (-1)
            else:
                weight_hh[i, i - 1] = weight_fb_seq[i // 2]
    if ff:
        for i in range(n):
            if i % 2 == 0 and i + 2 < n:
                weight_hh[i, i + 2] = 0.1
    return weight_hh


class FeedForward(nn.Module):
    def __init__(self, hidden_size, nonlinear="relu"):
        super().__init__()
        if nonlinear == "gelu":
            layers = [
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ]
        elif nonlinear == "relu":
            layers = [
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size * 4, hidden_size),
            ]
        else:
            raise ValueError("invalid nonlinear!!")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class MultiScaleMemoryLayerParallel(nn.Module):
    """
    通常のRNNの重みをマスクによって擬似的に独立化し
    MSM Layerの順伝播を並列化するレイヤー
    """

    def __init__(self, input_size, hidden_size, num_rnn):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn = num_rnn
        self.rnns = nn.RNN(
            input_size, hidden_size * num_rnn, batch_first=True, nonlinearity="relu"
        )
        self.register_buffer("mask", self.rnns.weight_hh_l0.data.clone())
        self.mask_init()

    def forward(self, x, h_0=None):
        """
        :param x:
        :param h_0:
        :return hs_all, shape=(num_rnn, batch_size, T, hidden_size):
        :return hidden, 最終ステップの隠れ状態のリスト（解析用）:
        """
        self.rnns.weight_hh_l0.data *= self.mask  # mask except non-diagonal
        if h_0 is None:
            h_all, h_last = self.rnns(x)
            hidden = h_last.detach()

        else:
            h_all, h_last = self.rnns(x, h_0)
            hidden = h_last.detach()

        hs_all = torch.chunk(h_all, self.num_rnn, dim=2)  # dim=2 -> hidden_dim
        hs_all = torch.stack(hs_all, dim=2)  # (B, T, N, H/N)

        return hs_all, hidden

    def init_weights(self, arange_type="linear", t_a=1, t_b=700):
        # Linear RedFB Init or Half-life RedFB Init
        if arange_type == "linear":
            arange_hid = np.linspace(0.5, 0.999, self.num_rnn)
            print("linear arange FB-weight=", arange_hid)
        elif arange_type == "half":
            half_life = np.linspace(t_b, t_a, self.num_rnn)
            arange_hid = np.exp(np.log(0.5) / half_life)
            print("half-life arange FB-weight=", arange_hid)
        else:
            raise ValueError("arange_type must be chosen linear or exp")

        weight_ih_list = []
        weight_hh_list = []
        for i in range(self.num_rnn):
            init_weight_ih = redfb_source_init(
                self.input_size, self.hidden_size, remainder_init=True, start=1, stop=1
            )
            init_weight_hh = redfb_feedback_init(
                self.hidden_size,
                type="pos",
                ff=False,
                start=arange_hid[i],
                stop=arange_hid[i],
            )
            weight_ih_list.append(init_weight_ih)
            weight_hh_list.append(init_weight_hh)

        weight_ih = torch.cat(weight_ih_list, dim=0)  # hidden_dimで結合
        weight_hh = torch.block_diag(*weight_hh_list)

        for name, param in self.rnns.named_parameters():
            if "weight_ih" in name:
                with torch.no_grad():
                    param.copy_(weight_ih)
            elif "weight_hh" in name:
                with torch.no_grad():
                    param.copy_(weight_hh)
            elif "bias" in name:
                param.data.fill_(0)

    def mask_init(self):
        weight_list = [
            torch.ones([self.hidden_size, self.hidden_size])
            for _ in range(self.num_rnn)
        ]
        self.mask = torch.block_diag(*weight_list)
        show_weights(self.mask, "aon")


class AttentionReadoutLayer(nn.Module):
    """
    異なる時間スケールの記憶を動的に統合処理するモジュール
    """

    def __init__(self, hidden_size, num_heads, nonlinear):
        super().__init__()
        # attention + add
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )

        # feedforward + add
        self.ff = FeedForward(hidden_size, nonlinear)

    def forward(self, h):
        """
        :param h: hidden_states, shape=(B*T, num_rnn, H)
        :return h: outputs, shape=(B*T, num_rnn, H)
        """
        h_ = h  # skip connection
        h, attn_mat = self.attn(h, h, h)
        h = h + h_  # skip connection

        h_ = h  # skip connection
        h = self.ff(h)
        h = h + h_
        return h, attn_mat

    def init_weights(self):
        for m in self.modules():
            for name, param in m.named_parameters():
                if "bias" in name:
                    param.data.fill_(0)


class AttentionRedFB(nn.Module):
    """
    AttentionRedFB = RNN Split + RedFB Init + Attention Mechanism
    """

    def __init__(
        self,
        input_size,
        d_model,
        vocab_size,
        num_rnn=6,
        num_heads=4,
        nonlinear="gelu",
        parallel=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_rnn = num_rnn

        # Multi-Scale Memory Layer
        if parallel:
            self.msm_layer = MultiScaleMemoryLayerParallel(input_size, d_model, num_rnn)
        else:
            self.msm_layer = MultiScaleMemoryLayer(input_size, d_model, num_rnn)

        # Attention Readout Layer
        self.attn_readout = AttentionReadoutLayer(d_model, num_heads, nonlinear)

        # classification head
        self.linear_head = nn.Linear(d_model * num_rnn, vocab_size)

        self.hidden = None
        self.attn_weight = None

        # init
        self.init_weights("half", t_a=1, t_b=700)

    def get_hidden(self):
        """
        :return hidden: 最後にforwardしたときの最終ステップの隠れ状態
        """
        return self.hidden

    def get_weight(self):
        return self.msm_layer.rnns.weight_hh_l0

    def get_attn(self):
        return self.attn_weight

    def init_states(self):
        self.hidden = None

    def forward(self, x, h_0=None):
        """
        :param x: 入力系列
        :param h_0: 初期状態
        :return outputs: 出力
        :return attn_mat: 全ての時刻のAttention行列
        """
        # memorize input series
        h, hidden = self.msm_layer(x, h_0)
        self.hidden = hidden.detach()

        # reshape
        # _, B, T, _ = h.shape
        B, T, N, H = h.shape
        h = h.reshape(-1, N, H)

        # integrate memory
        h, attn_weight = self.attn_readout(h)
        self.attn_weight = attn_weight.detach()

        # reshape, classification
        h = h.reshape(B, T, -1)
        outputs = self.linear_head(h)

        return outputs

    def init_weights(self, arange_type="linear", t_a=1, t_b=700):
        # msm layer
        self.msm_layer.init_weights(arange_type, t_a, t_b)

        # attn readout layer
        self.attn_readout.init_weights()

        # linear classification head
        for name, param in self.linear_head.named_parameters():
            if "bias" in name:
                param.data.fill_(0)


class MultiScaleMemoryLayer(nn.Module):
    """
    異なるフィードバック重みのRedFB初期化を適用した複数のRNNにより
    異なる時間スケールの記憶を獲得するレイヤー
    """

    def __init__(self, input_size, hidden_size, num_rnn):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn = num_rnn
        rnn_list = [
            nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="relu")
            for _ in range(num_rnn - 1)
        ]
        rnn_list.append(
            nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
        )
        self.rnns = nn.ModuleList(rnn_list)

    def forward(self, x, h_0=None):
        """
        :param x:
        :param h_0:
        :return hs_all, shape=(num_rnn, batch_size, T, hidden_size):
        :return hidden, 最終ステップの隠れ状態のリスト（解析用）:
        """
        if h_0 is None:
            hidden = []
            hs_all = []
            for rnn in self.rnns:
                h_all, h_last = rnn(x)
                hidden.append(h_last.detach())
                hs_all.append(h_all)
        else:
            hidden = []
            hs_all = []
            for i, rnn in enumerate(self.rnns):
                h_all, h_last = rnn(x, h_0[i])
                hidden.append(h_last.detach())
                hs_all.append(h_all)

        hidden = torch.stack(hidden, dim=0)
        hs_all = torch.stack(hs_all, dim=2)  # (B, T, N, H)

        return hs_all, hidden

    def init_weights(self, arange_type="linear", t_a=1, t_b=700):
        # Linear RedFB Init or Half-life RedFB Init
        if arange_type == "linear":
            arange_hid = np.linspace(0.5, 0.999, self.num_rnn - 1)
        elif arange_type == "half":
            half_life = np.linspace(t_b, t_a, self.num_rnn - 1)
            arange_hid = np.exp(np.log(0.5) / half_life)
            print("half-life arange FB-weight=", arange_hid)
        else:
            raise ValueError("arange_type must be chosen linear or exp")

        # RedFB Init
        for i, m in enumerate(self.rnns):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    if i == self.num_rnn - 1:
                        torch.nn.init.xavier_uniform_(param.data)
                    else:
                        init_param = redfb_source_init(
                            self.input_size,
                            self.hidden_size,
                            remainder_init=True,
                            start=1,
                            stop=1,
                        )
                        with torch.no_grad():
                            param.copy_(init_param)
                elif "weight_hh" in name:
                    if i == self.num_rnn - 1:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        # init_param = redfb_init(self.hidden_size, chain=False, arange=True,
                        #                         start=arange_hid[i], stop=arange_hid[i])
                        init_param = redfb_feedback_init(
                            self.hidden_size,
                            type="pos",
                            ff=False,
                            start=arange_hid[i],
                            stop=arange_hid[i],
                        )
                        with torch.no_grad():
                            param.copy_(init_param)
                elif "bias" in name:
                    param.data.fill_(0)


class AttnNoNormLMSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_rnn,
        num_heads=4,
        nonlinear="gelu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_rnn = num_rnn

        rnn_list = [
            nn.RNN(
                self.input_size, self.hidden_size, batch_first=True, nonlinearity="relu"
            )
            for _ in range(num_rnn - 1)
        ]
        rnn_list.append(
            nn.RNN(
                self.input_size, self.hidden_size, batch_first=True, nonlinearity="tanh"
            )
        )
        self.rnns = nn.ModuleList(rnn_list)

        # attention + add
        self.attn = nn.MultiheadAttention(
            self.hidden_size, num_heads=num_heads, batch_first=True
        )

        # feedforward + add
        # self.norm = nn.LayerNorm([num_rnn, self.hidden_size])
        self.ff = FeedForward(self.hidden_size, nonlinear)

        # classification head
        self.linear_head = nn.Linear(self.hidden_size * num_rnn, self.output_size)

        self.hidden = None
        self.attn_mat = None

        self.init_weights("half", 1, 700)

    def get_hidden(self):
        return self.hidden

    def init_states(self):
        self.hidden = None

    def forward(self, x, h_0=None):
        """""" """""" """
        [input]
                --x (torch.Tensor)
                    --x.shape = torch.Size([B, T, I])
                --h_0 (list(torch.tensor))
                    --h_0.shape = list(torch.Size([1, B, H]))
        """ """""" """"""
        if h_0 is None:
            hidden = []
            hs_all = []
            for rnn in self.rnns:
                h_all, h_last = rnn(x)
                hidden.append(h_last.detach())
                hs_all.append(h_all)
            self.hidden = hidden
        else:
            hidden = []
            hs_all = []
            for i, rnn in enumerate(self.rnns):
                h_all, h_last = rnn(x, h_0[i])
                hidden.append(h_last.detach())
                hs_all.append(h_all)
            self.hidden = hidden

        hs_all = torch.stack(hs_all)
        N, B, T, H = hs_all.shape
        h = einops.rearrange(hs_all, "N B T H -> (B T) N H")

        h_ = h  # skip connection
        h, attn_mat = self.attn(h, h, h)
        h = h + h_  # skip connection
        self.attn_mat = attn_mat

        h_ = h  # skip connection
        h = self.ff(h)
        h = h + h_
        h = h.reshape(B, T, -1)
        outputs = self.linear_head(h)

        return outputs

    def init_weights(
        self,
        arange_type="linear",
        t_a=1,
        t_b=700,
    ):
        # rnn(1〜num_rnn-1)
        if arange_type == "linear":
            arange_hid = np.linspace(0.5, 0.999, self.num_rnn - 1)
        elif arange_type == "half":
            # half_life = np.linspace(700, 1, self.num_rnn - 1)
            half_life = np.linspace(t_b, t_a, self.num_rnn - 1)
            arange_hid = np.exp(np.log(0.5) / half_life)
            print("half-life arange FB-weight=", arange_hid)
        else:
            raise ValueError("arange_type must be choose linear or exp")

        for i, m in enumerate(self.rnns):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    if i == self.num_rnn - 1:
                        torch.nn.init.xavier_uniform_(param.data)
                    else:
                        init_param = redfb_source_init(
                            self.input_size,
                            self.hidden_size,
                            remainder_init=True,
                            start=1,
                            stop=1,
                        )
                        with torch.no_grad():
                            param.copy_(init_param)
                elif "weight_hh" in name:
                    if i == self.num_rnn - 1:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        init_param = redfb_feedback_init(
                            self.hidden_size,
                            type="pos",
                            ff=False,
                            start=arange_hid[i],
                            stop=arange_hid[i],
                        )
                        with torch.no_grad():
                            param.copy_(init_param)
                elif "bias" in name:
                    param.data.fill_(0)

        for name, param in self.attn.named_parameters():
            if "bias" in name:
                param.data.fill_(0)
        for name, param in self.ff.named_parameters():
            if "bias" in name:
                param.data.fill_(0)
        for name, param in self.linear_head.named_parameters():
            if "bias" in name:
                param.data.fill_(0)


# test
def show_weights(weights, name):
    plt.imshow(weights.detach().numpy(), cmap="inferno")
    plt.savefig(f"{name}.pdf")


# mod1 = AttentionRedFB(2, 6, 4, 2, 1, "relu", parallel=False)
# mod2 = AttnNoNormLMSTM(2, 6, 4, 2, 1, "relu")
# mod3 = AttnNoNormLMSTM2([2, 6, 4], 2, 1, "relu")
# mod3.init_weights("half", True, False, t_a=1, t_b=700)
# a = torch.randn(1, 5, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRecurrentPolicy(nn.Module):
    def __init__(self, in_channels, num_hidden, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=num_hidden,
            num_layers=num_layers,
            batch_first=False)

        self.num_layers = num_layers
        self.hidden_shape = (2, self.num_layers, num_hidden)

        self.eval_sequence = LSTMRecurrentPolicy.eval_sequence_slow

    def infer(self, in_features, cur_hidden):
        in_features = in_features.view(1, *in_features.shape)

        out, (new_h, new_c) = self.lstm(in_features,
                                        (cur_hidden[0], cur_hidden[1]))

        new_hidden = torch.stack([new_h, new_c], dim=0)

        return out.view(*out.shape[1:]), new_hidden

    def lstm_iter_slow(self, layer_idx, in_features, cur_hidden, breaks):
        ifgo = \
            torch.mm(self.lstm.weight_ih_l[layer_idx],
                     in_features) + \
            self.lstm.bias_ih_l[0] + \
            torch.mm(self.lstm.weight_hh_l[layer_idx],
                     cur_hidden[0, :, :]) + \
            self.lstm.bias_hh_l[0]

        c = F.sigmoid(ifgo[:, 4:8]) * cur_hidden[1, :, :] + \
                F.sigmoid(ifgo[:, 0:4]) * F.tanh(ifgo[:, 8:12])

        o = ifgo[:, 12:16]

        h = o * F.tanh(c)

        new_hidden = torch.stack([h, c], dim=0)

        return o, new_hidden

    # Slow naive LSTM implementation
    def eval_sequence_slow(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            for layer_idx in range(self.num_layers):
                cur_features, new_hidden = self.lstm_iter_slow(
                    layer_idx, cur_features, cur_hidden[:, layer_idx, :, :],
                    sequence_breaks[i])

                cur_hidden[:, layer_idx, :, :] = new_hidden

                out_sequences.append(cur_features)

            cur_hidden = torch.where(
                cur_breaks, zero_hidden,
                cur_hidden_states)

        return torch.stack(out_sequences, dim=0)

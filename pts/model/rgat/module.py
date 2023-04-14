from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder

from pts.feature.lag import get_fourier_lags_for_frequency
from pts.modules import MLP

from parse import parser
args = parser.parse_args()


class RGAT_Decoder(nn.Module):
    def __init__(
        self,
        target_dim,
        input_size,
        decoder_hidden,
        dropout_rate,
        distr_output
    ):
        super().__init__()

        self.target_dim = target_dim
        self.msg_out_shape = decoder_hidden
        self.dropout_rate = dropout_rate

        self.dropout_in = nn.Dropout(p=0.0)
        self.dropout = nn.Dropout(p=0.5)
        self.activation_att = nn.LeakyReLU()
        self.activation = nn.Tanh()

        self.adj = torch.ones(target_dim, target_dim) - torch.eye(target_dim)

        self.linear = nn.Parameter(torch.empty(4, input_size, decoder_hidden))
        nn.init.xavier_uniform_(self.linear.data)

        self.attention_vect_src = nn.Parameter(torch.empty(4, decoder_hidden, 1))
        nn.init.xavier_uniform_(self.attention_vect_src.data)

        self.attention_vect_tar = nn.Parameter(torch.empty(4, decoder_hidden, 1))
        nn.init.xavier_uniform_(self.attention_vect_tar.data)

        self.bias = nn.Parameter(torch.zeros(1, decoder_hidden))

        # for skip connection
        self.hidden_r = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.hidden_i = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.hidden_m = nn.Linear(decoder_hidden, decoder_hidden, bias=False)

        self.input_r = nn.Linear(input_size, decoder_hidden, bias=True)
        self.input_i = nn.Linear(input_size, decoder_hidden, bias=True)
        self.input_n = nn.Linear(input_size, decoder_hidden, bias=True)

        self.out_mlp = MLP(decoder_hidden, [decoder_hidden, decoder_hidden])
        self.proj_dist_args = distr_output.get_args_proj(
            decoder_hidden * self.target_dim
        )

    def get_initial_hidden(self, inputs_size, device):
        return torch.zeros(
            inputs_size[0], inputs_size[2], self.msg_out_shape, device=device
        )

    def forward(self, inputs, hidden):
        
        # Dropout step
        x = self.dropout_in(inputs)
        old_shape = x.shape
        x = x.contiguous().view(-1, x.size(-1))

        # Attention mechanism
        h = torch.matmul(x, self.linear)
        h_src = torch.matmul(h, self.attention_vect_src)
        h_tar = torch.matmul(h, self.attention_vect_tar)

        h = h.view(h.size(0), old_shape[0], old_shape[1], -1)
        h_src = h_src.view(h_src.size(0), old_shape[0], old_shape[1], -1)
        h_tar = h_tar.view(h_tar.size(0), old_shape[0], old_shape[1], -1)

        score_mat = h_tar + h_src.transpose(-1, -2)
        if self.activation_att is not None:
            score_mat = self.activation_att(score_mat)
        score_mat = torch.where(self.adj.cuda() > 0, score_mat, -9e15)
        att_mat = torch.softmax(score_mat, dim=-1)
        att_mat = self.dropout(att_mat)

        # Message passing step
        msgs = torch.matmul(att_mat, h)
        msgs = msgs.mean(0)
        msgs += self.bias
        msgs = self.activation(msgs)

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.target_dim, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.target_dim, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.target_dim, -1)
        m = torch.sigmoid(inp_r + self.hidden_r(msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(msgs))
        n = torch.tanh(inp_n + m * self.hidden_m(msgs))
        hidden = i * n + (1 - i) * hidden

        pred = self.out_mlp(hidden)
        distr_args = self.proj_dist_args(pred.flatten(1))

        return distr_args, hidden


class RGATModel(nn.Module):
    def __init__(
        self,
        freq,
        target_dim,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        # num_feat_static_real: int,
        num_feat_static_cat: int,
        decoder_hidden,
        dropout_rate,
        distr_output,
        embedding_dimension: int,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100
    ):
        super().__init__()

        self.target_dim = target_dim
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        # self.num_feat_static_real = num_feat_static_real

        self.lags_seq = lags_seq or get_fourier_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.embedding_dimension = embedding_dimension

        input_size = self._number_of_features + len(self.lags_seq)

        self.decoder = RGAT_Decoder(
            target_dim=target_dim,
            input_size=input_size,
            decoder_hidden=decoder_hidden,
            dropout_rate=dropout_rate,
            distr_output=distr_output
        )

        self.embedder = FeatureEmbedder(
            cardinalities=[target_dim],
            embedding_dims=[self.embedding_dimension],
        )

        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

    @property
    def _number_of_features(self) -> int:
        return (
            self.embedding_dimension
            + 1 # scale
            + self.num_feat_dynamic_real
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)
    
    def prepare_inputs(
        self,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target: Optional[torch.Tensor],
        feat_static_cat: torch.Tensor
    ):
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        prior_input = past_target[:, : -self.context_length] / scale
        input = (
            torch.cat((context, future_target[:, :-1]), dim=1) / scale
            if future_target is not None
            else context / scale
        ) # [B,T,N]

        static_feat = self.embedder(feat_static_cat) # [B,N,Fs]
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, input.shape[1], -1, -1) # [B,T,N,Fs]

        expanded_static_feat = torch.cat((expanded_static_feat, scale.log().expand(-1, input.shape[1], -1).unsqueeze(-1)), dim=-1)

        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, -self.context_length + 1 :, ...],
                    future_time_feat,
                ),
                dim=1,
            ) # [B,T,Ft]
            if future_time_feat is not None
            else past_time_feat[:, -self.context_length + 1 :, ...]
        )
        expanded_time_feat = time_feat.unsqueeze(2).expand(-1, -1, self.target_dim, -1) # [B,T,N,Ft]

        features = torch.cat((expanded_static_feat, expanded_time_feat), dim=-1)

        inputs = self.unroll(
            prior_input,
            input,
            features
        )

        return scale, static_feat, inputs

    def unroll(
        self,
        prior_input: torch.Tensor,
        input: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ):

        sequence = torch.cat((prior_input, input), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=sequence,
            subsequences_length=input.shape[1],
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], self.target_dim, -1
        )

        if features is None:
            inputs = reshaped_lagged_sequence
        else:
            inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return inputs

    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = self.lags_seq

        assert max(indices) + subsequences_length <= sequence_length, (
            "lags cannot go further than history length, found lag"
            f" {max(indices)} while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        scale, static_feat, inputs = self.prepare_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat[:, :1],
            future_target=None
        )        

        # decoder_hidden via prior_logits
        decoder_hidden = self.decoder.get_initial_hidden(
            inputs.size(), device=inputs.device
        )
        for k in range(self.context_length):
            params, decoder_hidden = self.decoder(
                inputs[:, k],
                hidden=decoder_hidden
            )

        # sampling decoder
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target = repeat(past_target)
        repeated_time_feat = repeat(future_time_feat)
        repeated_time_feat = repeated_time_feat.unsqueeze(2).expand(
            -1, -1, self.target_dim, -1
        )

        repeated_scale = repeat(scale)
        repeated_static_feat = repeat(static_feat).unsqueeze(1)

        repeated_decoder_hidden = repeat(decoder_hidden)

        repeated_params = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=0).unsqueeze(1)
            for s in params
        ]
        distr = self.output_distribution(repeated_params, scale=repeated_scale)
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_scale.unsqueeze(-1), repeated_time_feat[:, k : k + 1, ...]),
                dim=-1
            )
            inputs = self.unroll(
                repeated_past_target,
                scaled_next_sample,
                next_features
            )
            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            params, repeated_decoder_hidden = self.decoder(
                inputs.squeeze(1),
                hidden=repeated_decoder_hidden
            )
            params = tuple([param.unsqueeze(1) for param in params])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

        # (batch_size * num_samples, prediction_length, target_dim)
        future_samples_concat = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return future_samples_concat.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape
        )
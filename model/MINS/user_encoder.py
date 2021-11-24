import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.multihead_self import MultiHeadSelfAttention
from model.additive import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.layers)
        self.position_embedding = nn.Parameter(
            torch.empty(config.num_clicked_news_a_user,
                        config.word_embedding_dim).uniform_(-0.1, 0.1))
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)
        assert config.num_filters % config.layers == 0
        self.gru = nn.GRU(
            int(config.num_filters / config.layers),
            int(config.num_filters / config.layers))
        self.multi_channel_gru = nn.ModuleList([self.gru for i in range(self.config.layers)])

    def forward(self, clicked_news_length, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch
        clicked_news_length[clicked_news_length == 0] = 1

        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(
            user_vector)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        one_channel = torch.chunk(multihead_user_vector, self.config.layers, dim=2)
        channels = []
        # batch_size, num_clicked_news_a_user, word_embedding_dim/layers
        for n, g in zip(range(self.config.layers), self.multi_channel_gru):
            packed_clicked_news_vector = pack_padded_sequence(
                one_channel[n],
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            # _, last_hidden = g(packed_clicked_news_vector, user.unsqueeze(dim=0))
            _, last_hidden = g(packed_clicked_news_vector)
            # 1,batch,config.num_filters / config.layers
            channels.append(last_hidden)
        multi_channel_vector = torch.cat(channels, dim=2).transpose(0,1)
        # batch_size, 1, word_embedding_dim
        final_user_vector = self.additive_attention(multi_channel_vector)
        # batch_size, word_embedding_dim
        return final_user_vector

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.multihead_self import MultiHeadSelfAttention


class Attention(torch.nn.Module):
    """
    Attention Net.
    Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
    produce final user embedding vectors with respect to the candidate news.
    """

    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.dnn = nn.Sequential(
            nn.Linear(
                self.config.num_filters,
                int(self.config.num_filters / 2)),
            nn.ReLU(),
            nn.Linear(int(self.config.num_filters / 2), 1))
        self.multihead_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)

    def forward(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: batch_size,  num_clicked_news_a_user, num_filters
            clicked_news_vector: batch_size, num_filters
        Returns:
            user_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user
        clicked_news_weights = F.softmax(
            self.dnn(
                self.multihead_attention(
                    torch.cat(
                        (candidate_news_vector,
                         clicked_news_vector.unsqueeze(dim=1)),
                        dim=1)
                )

            ).squeeze(dim=2), dim=1).squeeze(dim=-1)

        return clicked_news_weights

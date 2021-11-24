import torch
from model.MINS.news_encoder import NewsEncoder
from model.MINS.user_encoder import UserEncoder
from model.dot_product import DotProductClickPredictor
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Exp1(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(Exp1, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        assert int(config.num_filters % config.layers) == 0
        self.user_embedding = nn.Embedding(
            config.num_users,
            int(config.num_filters / config.layers),
            padding_idx=0)

    def forward(self, clicked_news_length, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """

        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)

        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_length, clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_filters
        user = self.user_embedding(user.to(device))
        # batch_size, word_embedding_dim
        return self.user_encoder(user, clicked_news_length, clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)

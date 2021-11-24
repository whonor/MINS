import os

model_name = 'MINS'
# Currently included model


class BaseConfig():
    """
    General configurations appiled to all models
    """
    validation_proportion = 0.1
    num_epochs = 2
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 16
    learning_rate = 0.0001
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 2  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 101220
    num_categories = 1 + 295
    num_entities = 1 + 21842
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class MINSConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 15
    num_filters = 300
    layers = 6
    masking_probability=0.5



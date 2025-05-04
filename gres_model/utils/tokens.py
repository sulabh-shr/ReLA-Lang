import os
import logging

from transformers import BertTokenizer, RobertaTokenizer


def get_tokenizer(tokenizer_type):
    if tokenizer_type in ("bert-base-uncased", "bert-large-uncased"):
        logging.getLogger(__name__).info("Loading BERT tokenizer: {}...".format(tokenizer_type))
        tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
    elif tokenizer_type in ("roberta-base", "roberta-large"):
        logging.getLogger(__name__).info("Loading RoBERTa tokenizer: {}...".format(tokenizer_type))
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_type)
    elif os.path.exists(tokenizer_type):
        tokenizer = tokenizer_type
        logging.getLogger(__name__).info("Using SAVED tokens from {}...".format(tokenizer_type))
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")

    return tokenizer

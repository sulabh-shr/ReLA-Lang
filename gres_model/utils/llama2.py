import abc
from abc import ABC
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class BaseTokenEncoder(ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, text: List[str]):
        pass


class Llama2TokenEncoder(BaseTokenEncoder):

    def __init__(self, model_id, device="cuda", add_eos=True):
        super().__init__()

        self.model_id = model_id
        self.add_eos = add_eos
        self.device = device

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Get EOS token ID
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=self.device
        )

    def encode(self, text: List[str]):
        """
        Encode text into token embeddings and save them to disk.

        Args:
            text (List[str]): Input text to encode

        Returns:
            batch_decoded_tokens (List[List[str]]): decoded tokens
            batch_embeddings (torch.Tensor): embeddings of the decoded tokens
        """
        # Convert text to model inputs (token ids and attention mask)
        batch_inputs = self.get_inputs(text)

        # Decode token ids back to readable tokens for saving
        token_ids = batch_inputs.pop("token_ids")
        batch_decoded_tokens = self.decode_batch_tokens(token_ids)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_inputs, output_hidden_states=False)

        # Extract final layer embeddings (shape: batch x sequence_length x hidden_size)
        batch_embeddings = outputs.last_hidden_state

        return batch_decoded_tokens, batch_embeddings

    def get_inputs(self, text: List[str]):
        """
        Tokenize input text and get model inputs as lists

        Args:
            text (List[str]): Input text to tokenize

        Returns:
            batch_inputs (dict): Dictionary containing token IDs and attention masks
                `input_ids` (torch.Tensor): Padded list of token IDs for each sequence
                `attention_mask` (torch.Tensor): Padded list of attention masks for each sequence
                `token_ids` (List[List[int]]): List of token IDs for each sequence
        """

        batch_inputs: Dict[str, List] = {
            "input_ids": [],
            "attention_mask": [],
            "token_ids": [],
        }
        max_token_length = 0

        # Convert each sentence in the text list to tokens individually
        for sequence in text:
            inputs = self.tokenizer(
                sequence,
                return_tensors=None,  # pt for torch, None for list
                padding=False,  # Pads to the longest sequence in the batch
                truncation=False,  # Truncates if too long
                return_attention_mask=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Optionally append EOS token ID and corresponding attention mask
            if self.add_eos:
                input_ids.append(self.eos_token_id)
                attention_mask.append(1)

            batch_inputs["input_ids"].append(input_ids)
            batch_inputs["attention_mask"].append(attention_mask)
            batch_inputs["token_ids"].append(list(input_ids))

            max_token_length = max(max_token_length, len(input_ids))

        # Pad input_ids and attention_mask to the max token length
        for idx in range(len(text)):
            num_tokens = len(batch_inputs["input_ids"][idx])
            pad_length = max_token_length - num_tokens
            batch_inputs["input_ids"][idx].extend([self.eos_token_id] * pad_length)
            batch_inputs["attention_mask"][idx].extend([0] * pad_length)

        # Convert input lists to tensors
        batch_inputs["input_ids"] = torch.tensor(
            batch_inputs["input_ids"], dtype=torch.int64
        ).to(self.device)
        batch_inputs["attention_mask"] = torch.tensor(
            batch_inputs["attention_mask"], dtype=torch.int64
        ).to(self.device)

        return batch_inputs

    def decode_batch_tokens(self, batch_token_ids: List[List[int]]):
        """
        Decode token IDs back to readable tokens for saving

        Args:
            batch_token_ids (List[List[int]]): List of token IDs for each sequence

        Returns:
            list: List of token sequences
        """
        # Get tokenizer instance and extract token IDs from batch inputs
        tokenizer = self.tokenizer
        token_ids_per_sequence = batch_token_ids

        # Convert each sequence of token IDs to their corresponding tokens
        decoded_token_sequences = []
        for sequence_token_ids in token_ids_per_sequence:
            tokens = tokenizer.convert_ids_to_tokens(sequence_token_ids)
            decoded_token_sequences.append(tokens)

        return decoded_token_sequences

    @staticmethod
    def save(
            batch_decoded_tokens: List[List[str]],
            batch_embeddings: torch.Tensor,
            names: List[str],
    ):
        # Save tokens and their embeddings for each sequence in batch
        for idx in range(len(batch_embeddings)):
            decoded_sequence_tokens = batch_decoded_tokens[idx]
            valid_token_length = len(decoded_sequence_tokens)

            # Get embeddings for current sequence and move to CPU
            sequence_token_embeddings = (
                batch_embeddings[idx, :valid_token_length].detach().cpu()
            )

            torch.save(
                {
                    "tokens": decoded_sequence_tokens,  # Size: valid_token_length
                    "embeddings": sequence_token_embeddings,  # Size: valid_token_length x dim
                },
                names[idx],
            )


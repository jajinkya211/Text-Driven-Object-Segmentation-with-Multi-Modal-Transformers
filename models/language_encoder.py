"""
Language Encoder with BERT + BiLSTM for referring expression understanding
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from typing import Tuple, Optional


class LanguageEncoder(nn.Module):
    """
    BERT-based language encoder for referring expressions

    Architecture:
    - BERT-base (pretrained) for contextual word embeddings
    - BiLSTM for additional sequential processing
    - Word-level and sentence-level embeddings

    The encoder produces:
    - Word features: [B, L, output_dim] for each token
    - Sentence feature: [B, output_dim] for entire expression

    Args:
        bert_model: Name of pretrained BERT model
        hidden_dim: BERT hidden dimension (usually 768)
        output_dim: Output feature dimension
        lstm_hidden_dim: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        hidden_dim: int = 768,
        output_dim: int = 256,
        lstm_hidden_dim: int = 384,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()

        self.bert_model_name = bert_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Load pretrained BERT
        if 'roberta' in bert_model:
            self.bert = RobertaModel.from_pretrained(bert_model)
        else:
            self.bert = BertModel.from_pretrained(bert_model)

        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # BiLSTM for additional sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim // 2,  # Bidirectional, so // 2
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # Projection layers
        self.word_projection = nn.Sequential(
            nn.Linear(lstm_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        self.sentence_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        # Additional sentence-level processing
        self.sentence_refine = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode referring expression

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (1 for real tokens, 0 for padding)

        Returns:
            word_features: [B, L, output_dim] per-word features
            sent_feature: [B, output_dim] sentence-level feature
        """

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Word-level features (all tokens)
        word_embeddings = outputs.last_hidden_state  # [B, L, hidden_dim]

        # Sentence-level feature (CLS token or pooled output)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            sent_embedding = outputs.pooler_output  # [B, hidden_dim]
        else:
            # Use first token (CLS) if pooler output not available
            sent_embedding = word_embeddings[:, 0, :]  # [B, hidden_dim]

        # BiLSTM for sequential context
        # Pack padded sequence for efficiency
        lengths = attention_mask.sum(dim=1).cpu()
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            word_embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_lstm_out, _ = self.lstm(packed_embeddings)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out,
            batch_first=True,
            total_length=word_embeddings.size(1)
        )  # [B, L, lstm_hidden_dim]

        # Project to output dimension
        word_features = self.word_projection(lstm_out)  # [B, L, output_dim]

        # Project sentence embedding
        sent_feature = self.sentence_projection(sent_embedding)  # [B, output_dim]

        # Refine sentence feature
        sent_feature = sent_feature + self.sentence_refine(sent_feature)  # Residual

        return word_features, sent_feature

    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.output_dim


class SimpleLanguageEncoder(nn.Module):
    """
    Simplified language encoder without LSTM

    Uses only BERT with projection layers.
    Faster and uses less memory than full LanguageEncoder.

    Args:
        bert_model: Name of pretrained BERT model
        hidden_dim: BERT hidden dimension
        output_dim: Output feature dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        hidden_dim: int = 768,
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()

        self.bert_model_name = bert_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Load pretrained BERT
        if 'roberta' in bert_model:
            self.bert = RobertaModel.from_pretrained(bert_model)
        else:
            self.bert = BertModel.from_pretrained(bert_model)

        # Optionally freeze BERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layers
        self.word_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        self.sentence_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode referring expression

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask

        Returns:
            word_features: [B, L, output_dim] per-word features
            sent_feature: [B, output_dim] sentence-level feature
        """

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Word-level features
        word_embeddings = outputs.last_hidden_state  # [B, L, hidden_dim]
        word_features = self.word_projection(word_embeddings)  # [B, L, output_dim]

        # Sentence-level feature
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            sent_embedding = outputs.pooler_output
        else:
            sent_embedding = word_embeddings[:, 0, :]

        sent_feature = self.sentence_projection(sent_embedding)  # [B, output_dim]

        return word_features, sent_feature

    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.output_dim


def build_language_encoder(
    model_type: str = 'bert',
    bert_model: str = 'bert-base-uncased',
    output_dim: int = 256,
    use_lstm: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to build language encoder

    Args:
        model_type: Type of language model ('bert', 'roberta')
        bert_model: Specific model name
        output_dim: Output feature dimension
        use_lstm: Whether to use BiLSTM
        **kwargs: Additional arguments

    Returns:
        Language encoder module
    """

    # Construct full model name
    if model_type == 'bert' and not bert_model.startswith('bert'):
        bert_model = f'bert-{bert_model}'
    elif model_type == 'roberta' and not bert_model.startswith('roberta'):
        bert_model = f'roberta-{bert_model}'

    if use_lstm:
        return LanguageEncoder(
            bert_model=bert_model,
            output_dim=output_dim,
            **kwargs
        )
    else:
        return SimpleLanguageEncoder(
            bert_model=bert_model,
            output_dim=output_dim,
            **kwargs
        )


def get_tokenizer(bert_model: str = 'bert-base-uncased'):
    """
    Get appropriate tokenizer for the language model

    Args:
        bert_model: Name of BERT model

    Returns:
        Tokenizer
    """

    if 'roberta' in bert_model:
        return RobertaTokenizer.from_pretrained(bert_model)
    else:
        return BertTokenizer.from_pretrained(bert_model)

import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        token_ids = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')['input_ids']

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sentence in token_ids:
                sentence = sentence[sentence != tokenizer.pad_token_id]
                if len(sentence) < 2 * self.window_size + 1:
                    continue

                for center_idx in range(self.window_size, len(sentence) - self.window_size):
                    center_word = sentence[center_idx]
                    context_words = torch.cat([
                        sentence[center_idx - self.window_size:center_idx],
                        sentence[center_idx + 1:center_idx + 1 + self.window_size]
                    ])

                    if self.method == "cbow":
                        loss = self._train_cbow(center_word, context_words, criterion)
                    else:
                        loss = self._train_skipgram(center_word, context_words, criterion)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.4f}")

    def _train_cbow(
        self,
        center_word: LongTensor,
        context_words: LongTensor,
        criterion: nn.CrossEntropyLoss
    ) -> Tensor:
        embedded = self.embeddings(context_words)  # (2*window, d_model)
        context_vector = embedded.mean(dim=0)      # (d_model,)
        logits = self.linear(context_vector)       # (vocab_size,)
        loss = criterion(logits.unsqueeze(0), center_word.unsqueeze(0))
        return loss

    def _train_skipgram(
        self,
        center_word: LongTensor,
        context_words: LongTensor,
        criterion: nn.CrossEntropyLoss
    ) -> Tensor:
        center_vector = self.embeddings(center_word)  # (d_model,)
        logits = self.linear(center_vector)           # (vocab_size,)
        loss = criterion(logits.repeat(len(context_words), 1), context_words)
        return loss

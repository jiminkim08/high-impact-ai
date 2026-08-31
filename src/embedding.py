"""Qwen3-Embedding wrapper.

Qwen3-Embedding is a decoder (causal-attention) model, which has two
consequences that differ from the usual encoder-based sentence embedders:

* the final non-padding token carries the full context, so pooling takes that
  token rather than the mean of all tokens or a CLS position;
* last-token pooling requires **left** padding, so that the final position of
  every sequence in a batch is a real token.

Per the model's documented usage, the instruction prefix applies to queries
only; documents are encoded as-is. All embedding in this study is document-side.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Given a patent title and abstract, retrieve semantically related patents"
)


class Qwen3Embedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        instruction: str = DEFAULT_INSTRUCTION,
        normalize: bool = True,
        prefer_bf16: bool = True,
        allow_fp16_fallback: bool = True,
    ):
        self.instruction = instruction
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        logger.info("model loaded on %s", self.device)

        self._amp_dtype: Optional[torch.dtype] = None
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception as exc:  # pragma: no cover - depends on torch build
                logger.warning("could not set matmul precision: %s", exc)

            if prefer_bf16 and torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
            elif allow_fp16_fallback:
                self._amp_dtype = torch.float16

    @classmethod
    def from_config(cls, cfg: dict) -> "Qwen3Embedder":
        return cls(
            model_name=cfg.get("local_dir") or cfg["model_name"],
            normalize=cfg.get("l2_normalize", True),
            prefer_bf16=cfg.get("prefer_bf16", True),
            allow_fp16_fallback=cfg.get("allow_fp16_fallback", True),
        )

    # -- pooling ------------------------------------------------------------

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool the final non-padding token of each sequence.

        With left padding the last position is always a real token, so the
        tensor can be sliced directly; the right-padded branch is kept as a
        safeguard in case the tokenizer configuration changes.
        """
        left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padded:
            return last_hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[idx, lengths]

    def _apply_instruction(self, batch: Sequence[str], is_query: bool) -> list[str]:
        if not is_query:
            return list(batch)
        return [f"Instruct: {self.instruction}\nQuery:{s}" for s in batch]

    # -- encoding -----------------------------------------------------------

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        is_query: bool = False,
        batch_size: int = 64,
        max_length: int = 256,
        return_single_vector: bool = False,
    ) -> np.ndarray:
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            raise ValueError("no non-empty texts to encode")

        use_cuda = self.device.type == "cuda"
        embeddings: list[np.ndarray] = []

        try:
            for start in range(0, len(sentences), batch_size):
                batch = self._apply_instruction(
                    sentences[start : start + batch_size], is_query=is_query
                )

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    # pad to a multiple of 8 so tensor cores are used efficiently
                    pad_to_multiple_of=8 if use_cuda else None,
                    return_tensors="pt",
                ).to(self.device)

                if use_cuda and self._amp_dtype is not None:
                    amp_ctx = torch.autocast(device_type="cuda", dtype=self._amp_dtype)
                else:
                    amp_ctx = torch.autocast(device_type="cpu", enabled=False)

                with torch.inference_mode(), amp_ctx:
                    outputs = self.model(**inputs)
                    pooled = self._last_token_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    if self.normalize:
                        pooled = F.normalize(pooled, p=2, dim=1)

                embeddings.append(pooled.detach().float().cpu().numpy())

            result = np.vstack(embeddings)
            if is_single and return_single_vector:
                return result[0]
            return result

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "CUDA OOM: reduce embedding.batch_size (%d) or embedding.max_length (%d)",
                    batch_size,
                    max_length,
                )
            raise

    def clear_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

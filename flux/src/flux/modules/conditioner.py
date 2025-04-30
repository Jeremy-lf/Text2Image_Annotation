from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("/root/paddlejob/workspace/env_run/wchn/wepaint/openai/clip-vit-large-patch14", max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained("/root/paddlejob/workspace/env_run/wchn/wepaint/openai/clip-vit-large-patch14", **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("/root/.cache/huggingface/hub/models--xlabs-ai--xflux_text_encoders/snapshots/5ce032c6b9bfe31a4ffb220c8afa147e8de6acea", max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained("/root/.cache/huggingface/hub/models--xlabs-ai--xflux_text_encoders/snapshots/5ce032c6b9bfe31a4ffb220c8afa147e8de6acea", **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

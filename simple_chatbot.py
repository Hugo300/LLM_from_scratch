# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit


# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from classes.model import GPTModel
from classes.generation import generator, TextTokenConversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_generator():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """

    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True,         # Query-key-value bias
        "gpt_model_name": "gpt2-medium (355M)",
        "gpt_num_params": "355M"
    }

    model_path = Path(".") / "weights" / "gpt2-medium355M-instruction.pth"
    if not model_path.exists():
        print(f"Could not find the {model_path} file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    text_gen = generator(
        model=model,
        encoder=TextTokenConversion(tiktoken.get_encoding("gpt2"))
    )

    return text_gen, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
text_gen, model, model_config = get_model_and_generator()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)

    prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{message.content}"
    )

    token_ids = text_gen.generate_text_input_text(  # function uses `with torch.no_grad()` internally already
        text=prompt,
        max_new_tokens=35,
        eos_id=50256
    )

    text = text_gen.encoder.decode(token_ids)
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
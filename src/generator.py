import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from src.config import (
    LLM_MODEL,
    MAX_NEW_TOKENS
)


# PROMPT BUILDER

def build_prompt(
    query,
    retrieved_chunks
):

    context = ""

    for chunk in retrieved_chunks:

        context += (
            f"[{chunk.document} - Page {chunk.page}]\n"
            f"{chunk.text}\n\n"
        )

    prompt = f"""
You are answering questions ONLY from the provided context.

Context:
{context}

Question:
{query}

Answer clearly and concisely using ONLY the provided context.
"""

    return prompt


# QWEN GENERATOR

class QwenGenerator:

    def __init__(
        self,
        model_name=LLM_MODEL
    ):

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    # GENERATE

    def generate(
        self,
        query,
        retrieved_chunks,
        max_new_tokens=MAX_NEW_TOKENS
    ):

        prompt = build_prompt(
            query,
            retrieved_chunks
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()
import torch

from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


# MOONDREAM VLM

class MoondreamVLM:

    def __init__(
        self,
        model_name="vikhyatk/moondream2",
        revision="2025-06-21"
    ):

        # LOAD MODEL ON CPU

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to("cpu")

        self.model.eval()

        # OPTIONAL TOKENIZER

        try:

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision
                )
            )

        except Exception:

            self.tokenizer = None

    # VISUAL QUESTION ANSWERING

    def answer_visual_question(
        self,
        image_path,
        question
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        result = self.model.query(
            image,
            question
        )

        # HANDLE DIFFERENT OUTPUT FORMATS

        if isinstance(result, dict):

            return result.get(
                "answer",
                ""
            )

        return str(result)
"""
phi_prompter.py

Defines a PromptBuilder for building Phi-2 Input/Output Prompts --> recommended pattern used by HF / Microsoft.
Also handles Phi special case BOS token additions.

Reference: https://huggingface.co/microsoft/phi-2#qa-format
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


class PaliGemmaPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # Note =>> Phi Tokenizer is an instance of `CodeGenTokenizer(Fast)`
        #      =>> By default, does *not* append <BOS> / <EOS> tokens --> we handle that here (IMPORTANT)!
        self.bos, self.eos = "<bos>", "<eos>"
        self.image_token = "<image>"

        # # Get role-specific "wrap" functions
        # #   =>> Note that placement of <bos>/<eos> were based on experiments generating from Phi-2 in Input/Output mode
        # self.wrap_human = lambda msg: f"{msg}\n"
        # self.wrap_gpt   = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        return self.add_turn_for_multimodal(role, message, is_multimodal=True)

    def add_turn_for_multimodal(self, role: str, message: str, is_multimodal: bool = True) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace(f"{self.image_token}", "").strip()
        message = message.replace("\n", "\t")

        # Special Handling for "first" input --> prepend a <BOS> token (expected by Prismatic)
        if self.turn_count == 0:
            if is_multimodal:
                bos_human_message = f"{self.image_token}{self.bos}answer en {message}\n"
            else:
                bos_human_message = f"{self.bos}<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
            wrapped_message = bos_human_message
        elif (self.turn_count % 2) == 0:
            if is_multimodal:
                human_message = f"answer en {message}\n"
            else:
                human_message = f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
            wrapped_message = human_message
        else:
            if is_multimodal:
                gpt_message = f"{message if message != '' else ' '}{self.eos}"
            else:
                gpt_message = f"{message}<end_of_turn>\n"

            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def add_turn_for_caption(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace(f"{self.image_token}", "").strip()
        message = message.replace("\n", "\t")

        # Special Handling for "first" input --> prepend a <BOS> token (expected by Prismatic)
        if self.turn_count == 0:
            bos_human_message = f"{self.image_token}{self.bos}caption en\n"
            wrapped_message = bos_human_message
        else:
            gpt_message = f"{message if message != '' else ' '}{self.eos}"
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def add_turn_for_intention(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace(f"{self.image_token}", "").strip()
        message = message.replace("\n", "\t")

        # Special Handling for "first" input --> prepend a <BOS> token (expected by Prismatic)
        if self.turn_count == 0:
            bos_human_message = f"{self.image_token}{self.bos}intention en {message}\n"
            # bos_human_message = f"{self.image_token}{self.bos}intention en\n"
            wrapped_message = bos_human_message
        else:
            gpt_message = f"{message if message != '' else ' '}{self.eos}"
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # # Assumes that it's always the user's (human's) turn!
        # prompt_copy = str(self.prompt)

        # human_message = self.wrap_human(message)
        # prompt_copy += human_message

        # # return prompt_copy.rstrip()
        # return prompt_copy
        return None

    def get_prompt(self) -> str:
        # return self.prompt.rstrip()
        return self.prompt

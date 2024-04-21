import dataclasses
from typing import List


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    sep_style: str
    sep: str = "###"
    sep2: str = None

    def get_prompt(self, messages):
        if self.sep_style == 'single':
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == 'two':
            seps = [self.sep, self.sep2]
            ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == 'plain':
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, message in enumerate(messages):
                ret += message + seps[i % 2]

        elif self.sep_style == 'llama2':
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        return ret


conv_plain = Conversation(
    system="",
    roles=("", ""),
    sep_style='plain',
    sep=" ",
    sep2=""
)

conv_default = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    sep_style='two',
    sep=" ",
    sep2=" ",
)

conv_llava = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    sep_style='two',
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    sep_style='llama2',
    sep="<s>",
    sep2="</s>"
)

conv_templates = {
    "simple": conv_plain,
    "default": conv_default,
    "llava": conv_llava,
    "llama_2": conv_llama_2,
}

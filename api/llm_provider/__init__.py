import os
from typing import List, Dict
import ollama


class LLMProvider:
    def __init__(self, provider: str = "ollama", model_name: str = "llama3.1:latest"):
        self.provider = provider
        self.model_name = model_name
        print(f"Model Provider Selected: {self.provider}, Model Name: {self.model_name}")
        if self.provider == "ollama":
            models_names = list(map(lambda x : x["name"],ollama.list()["models"]))
            if self.model_name not in models_names:
                print("ollama pulling")
                ollama.pull(self.model_name)
            else:
                print("model exists")

    def invoke_llm(self, messages: List[Dict]) -> str:
        """
        Invokes the LLM to generate a response.

        :param messages: A list of messages to provide context and user input.
        :return: The generated response content.
        """

        if self.provider == "ollama":
            response = ollama.chat(model=self.model_name,options={'temperature' : 0}, messages=messages)
            return response.get("message", {}).get("content", "")
        raise NotImplementedError(f"Provider '{self.provider}' is not implemented")


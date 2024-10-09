import boto3


from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM

__all__ = [
    "Bedrock"
]

class Bedrock(LLM):
    def __init__(self,
                 model_id: str,
                 access_key_id:str,
                 access_key: str,
                 region_name: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key,
            region_name=region_name
        )
        self.client = session.client('bedrock-runtime', region_name)

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.1,
             topP: float = 0.9) -> str:
        messages = transform_messages(text, self.system_prompt)

        response = self.client.converse(
            modelId=self.model_id,#"meta.llama2-13b-chat-v1",
            messages=messages,
            inferenceConfig={"maxTokens":max_new_tokens,"temperature":temperature,"topP":topP},
            additionalModelRequestFields={}
        )
        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1) -> str:
        text = f"Continue writing: {text}"
        
        return self.chat(text, max_new_tokens=max_new_tokens, temperature=temperature)

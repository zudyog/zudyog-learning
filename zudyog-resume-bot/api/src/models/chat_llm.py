from typing import List
from pydantic import BaseModel, Field
from src.core.config import get_openai_client
from src.database import Usage, create_usage


client = get_openai_client()


class ChatLLM(BaseModel):
    model: str = 'gpt-4o'  # Default model to use
    temperature: float = 0.0  # Default temperature for generating responses

    class Config:
        arbitrary_types_allowed = True

    # Method to generate a response from the model based on the provided prompt
    def generate(self, prompt: str, stop: List[str] = None):
        # Create a completion request to the OpenAI API with the given parameters
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )

        # Create the Usage object
        usage = Usage(
            prompt=prompt,
            temperature=self.temperature,
            stop=",".join(stop) if stop else None,
            is_openai=True,
            response=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        # Create the usage record
        create_usage(usage=usage)

        # Return the generated response content
        return response.choices[0].message.content

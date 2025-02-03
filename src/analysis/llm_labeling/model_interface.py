from abc import ABC, abstractmethod
import os
import json
from getpass import getpass
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from anthropic import AnthropicVertex


from src.analysis.llm_labeling.helpers import extract_json


class AIModelInterface(ABC):
    """Abstract base class for interacting with different AI models."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model with necessary configurations."""
        pass

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema_path: Optional[str] = None,
    ) -> Dict:
        """Generate a response from the model.

        Args:
            prompt: The input text prompt
            max_tokens: Maximum number of tokens in response
            temperature: Temperature for response generation
            schema_path: Optional path to JSON schema for structured output

        Returns:
            Dict containing the response and any metadata
        """
        pass


class GeminiInterface(AIModelInterface):
    """Implementation for Google's Gemini API."""

    def __init__(self, model_name: str = "gemini-1.5-flash-001"):
        self.model_name = model_name
        self.model = None

    def initialize(self) -> None:
        """Initialize Gemini with API key."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = getpass("Enter your Gemini API key: ")

        if not api_key:
            raise ValueError("API key is required to use Gemini")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema_path: Optional[str] = None,
        retries: int = 3,
    ) -> Dict:
        """Generate response using Gemini."""
        for _ in range(retries):
            try:
                response = self._generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema_path=schema_path,
                )

                # If response is generated, return it
                if response:
                    return response
            except Exception as e:
                print(f"Error generating response: {str(e)}")


    def _generate_response(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            schema_path: Optional[str] = None,
    ) -> Dict | None:

        if not self.model:
            self.initialize()

        generation_config = {}
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        if temperature:
            generation_config["temperature"] = temperature
        if schema_path:
            with open(schema_path) as f:
                json_schema = f.read()
                json_dict = json.loads(json_schema)
            prompt = f"{prompt}\n\n{json_schema}"

        chat = self.model.start_chat(history=[])
        response = chat.send_message(prompt)

        # Extract JSON if present, otherwise return full text
        try:
            json_response = extract_json(response.text)

            # Make sure that all the keys are present in the response
            for k in json_dict["properties"].keys():
                if k not in json_response.keys():
                    return None

            if json_response:
                return {
                    "content": json_response,
                    "raw_response": response.text,
                    "model": self.model_name,
                }
        except Exception:
            pass

        return {"content": response.text, "model": self.model_name}


class ClaudeInterface(AIModelInterface):
    """Implementation for Anthropic's Claude API."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-v2@20241022",
        region: str = "us-east5",
        project_id: str = None,
    ):
        self.model_name = model_name
        self.region = region
        self.project_id = project_id
        self.client = None

    def initialize(self) -> None:
        """Initialize Claude with necessary configurations."""
        if not self.project_id:
            self.project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            if not self.project_id:
                raise ValueError("Project ID is required for Claude")

        self.client = AnthropicVertex(region=self.region, project_id=self.project_id)

    def generate_response(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            schema_path: Optional[str] = None,
            retries: int = 3,
    ) -> Dict:
        """Generate response using Gemini."""
        for _ in range(retries):
            try:
                response = self._generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema_path=schema_path,
                )

                # If response is generated, return it
                if response:
                    return response
            except Exception as e:
                print(f"Error generating response: {str(e)}")


    def _generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema_path: Optional[str] = None,
    ) -> Dict | None:
        """Generate response using Claude."""
        if not self.client:
            self.initialize()

        message_params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model_name,
        }

        if max_tokens:
            message_params["max_tokens"] = max_tokens
        if temperature:
            message_params["temperature"] = temperature
        if schema_path:
            with open(schema_path) as f:
                json_schema = f.read()
                json_dict = json.loads(json_schema)
            message_params["messages"][0]["content"] = f"{prompt}\n\n{json_schema}"

        response = self.client.messages.create(**message_params)

        # Extract JSON if present in the response
        try:
            json_response = extract_json(response.content[0].text)

            # Make sure that all the keys are present in the response
            for k in json_dict["properties"].keys():
                if k not in json_response.keys():
                    return None

            if json_response:
                return {
                    "content": json_response,
                    "raw_response": response.content[0].text,
                    "model": self.model_name,
                    "usage": response.usage,
                }
        except Exception:
            pass

        return {
            "content": response.content[0].text,
            "model": self.model_name,
            "usage": response.usage,
        }


# Helper function to create model interface instances
def create_model_interface(model_type: str, **kwargs) -> AIModelInterface:
    """Factory function to create model interfaces.

    Args:
        model_type: Either 'gemini' or 'claude'
        **kwargs: Additional arguments to pass to the model interface

    Returns:
        An initialized model interface
    """
    if "gemini" in model_type.lower():
        return GeminiInterface(**kwargs)
    elif "claude" in model_type.lower():
        return ClaudeInterface(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

"""
LLM client factory.

Supports two providers:
  - anthropic  (direct Anthropic API, needs ANTHROPIC_API_KEY)
  - bedrock    (AWS Bedrock, needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
                 or an IAM role — standard boto3 credential chain)

Set LLM_PROVIDER=bedrock in .env to switch.
Set MODEL_ID to override the default model for whichever provider is active.
"""
import os
from typing import Union

import anthropic


# Default model IDs per provider.
# Bedrock uses ARN-style cross-region inference profile IDs.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "bedrock": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def get_client() -> Union[anthropic.Anthropic, anthropic.AnthropicBedrock]:
    """Return an Anthropic client configured for the active provider."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()

    if provider == "bedrock":
        # Uses the standard boto3 credential chain:
        #   1. env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
        #   2. ~/.aws/credentials
        #   3. IAM instance/task role
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return anthropic.AnthropicBedrock(aws_region=region)

    # Default: direct Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to .env or set LLM_PROVIDER=bedrock to use AWS credentials instead."
        )
    return anthropic.Anthropic(api_key=api_key)


def get_model_id() -> str:
    """Return the model ID to use, respecting the MODEL_ID override env var."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    return os.getenv("MODEL_ID", DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"]))

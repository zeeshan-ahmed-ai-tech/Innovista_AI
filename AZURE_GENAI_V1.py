import os
from openai import AzureOpenAI
import json

client = AzureOpenAI(
    api_version="2024-02-01",
    #azure_endpoint = "https://openaiforansar.openai.azure.com/",
    azure_endpoint="https://openaiforansar.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview",
    api_key="e0fec83b1f1348fa9f06a6a49dcfdf33",
)

# Define the chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a one line definition of a neural network"}
]

# Call the chat completion endpoint
result = client.chat.completions.create(
    model="gpt-35-turbo",  # Specify your model here
    messages=messages,
    temperature=0.7,  # Adjust temperature for creativity
    max_tokens=150  # Set the maximum number of tokens for the response
)

# Extract and print the assistant's response correctly
assistant_response = result.choices[0].message.content  # Corrected to access content directly
print(assistant_response)

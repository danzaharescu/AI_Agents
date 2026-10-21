

import os
from google import genai

# 1. Initialize the client (automatically uses GEMINI_API_KEY from environment variables)
client = genai.Client()

# 2. Generate content using Gemini Flash

client = genai.Client()

# List available models
# for model in client.models.list():
#     if "generateContent" in model.supported_actions:
#         print(model.name)

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Explain quantum computing in two simple sentences.",
)

# 3. Print the response
print(response.text)
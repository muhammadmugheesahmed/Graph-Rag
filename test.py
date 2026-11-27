from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)

response = client.models.generate_content(
    model="gemini-2.5-flash",  # or another model from Step 1, e.g. "gemini-3.8-flash"
    contents="Hello!",
)
print(response.text)
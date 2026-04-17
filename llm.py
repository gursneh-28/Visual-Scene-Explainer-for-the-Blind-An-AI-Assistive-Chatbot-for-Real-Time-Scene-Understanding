from google import genai
from google.genai import types
import os
import base64
from dotenv import load_dotenv

load_dotenv()

def generate_description(image_bytes, objects, texts, find_object=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LLM error: No API key found")
        return None

    client = genai.Client(api_key=api_key)

    # Convert image to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    if find_object and find_object != 'text_only':
        prompt = f"""
You are an AI assistant for a visually impaired person.
They are looking for: "{find_object}"
Look carefully at this image.
If you can see it, say exactly where it is (left, right, center, close, far).
If not visible, say so clearly and suggest they turn around slowly.
Be brief, max 2 sentences. Speak directly to the person.
"""
    elif find_object == 'text_only':
        prompt = """
You are an AI assistant for a visually impaired person.
Read ALL text visible in this image clearly and naturally.
Include signs, labels, books, screens, anything with text.
If no text is visible, say so briefly.
"""
    else:
        prompt = """
You are an AI assistant helping a visually impaired person understand their surroundings.
Look at this image carefully and describe:
1. What objects are present and where (left, right, center, close, far)
2. Any text visible
3. Any safety concerns (obstacles, people very close, stairs, etc)
Be natural and conversational, max 3 sentences.
Mention safety concerns first if any.
Speak directly to the person using "you" and "your".
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=image_b64
                            )
                        ),
                        types.Part(text=prompt)
                    ]
                )
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None
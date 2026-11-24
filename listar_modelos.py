from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

gemini = genai.Client()  # guardado en una variable: sigue vivo durante el ciclo
groq = Groq()

print("== Gemini ==")
for m in gemini.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)

print("\n== Groq ==")
for m in groq.models.list().data:
    print(m.id)

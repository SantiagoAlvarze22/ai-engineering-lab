import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from groq import Groq

load_dotenv(override=True)  # override=True: siempre manda lo que dice el .env

gemini = genai.Client()
groq = Groq()

pregunta = "Responde solo con la palabra: listo"

modelos_gemini = list(
    dict.fromkeys(
        [
            os.environ["GEMINI_MODEL"],  # gemini-3.8-flash
            "gemini-3.7-flash",  # respaldo 1
            "gemini-3.5-flash-lite",  # respaldo 2
        ]
    )
)

# --- diagnóstico (lo puedes borrar cuando todo funcione) ---
print("Modelo en .env:", os.environ["GEMINI_MODEL"])
print("Lista a probar:", modelos_gemini)
print("-" * 40)

# ---------- Gemini con respaldo ----------
respuesta_gemini = None
for modelo in modelos_gemini:
    try:
        g = gemini.models.generate_content(model=modelo, contents=pregunta)
        respuesta_gemini = f"Gemini ({modelo}): {g.text.strip()}"
        break
    except errors.ServerError as e:
        print(f"  {modelo} no disponible ({e.code}), probando el siguiente...")
    except errors.ClientError as e:
        if e.code == 429:
            print(f"  {modelo} sin cuota (429), probando el siguiente...")
        else:
            raise

print(respuesta_gemini or "Gemini: ningún modelo respondió, intenta más tarde")

# ---------- Groq ----------
r = groq.chat.completions.create(
    model=os.environ["GROQ_MODEL"],
    messages=[{"role": "user", "content": pregunta}],
)
print(f"Groq ({os.environ['GROQ_MODEL']}): {r.choices[0].message.content.strip()}")

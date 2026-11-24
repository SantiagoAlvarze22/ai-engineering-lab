import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)
cliente = genai.Client()

resena = "La comida llegó fría y tardaron 40 minutos, pero el mesero fue muy amable y nos regaló el postre."

respuesta = cliente.models.generate_content(
    model=os.environ["GEMINI_MODEL"],
    contents=resena,  # la tarea concreta
    config=types.GenerateContentConfig(
        system_instruction="Resumes reseñas de restaurantes en una sola frase neutral, sin opinar.",  # rol y reglas
        temperature=1.5,  # baja: respuestas estables
    ),
)

print("Resumen:", respuesta.text.strip())
uso = respuesta.usage_metadata
print(f"Tokens -> entrada: {uso.prompt_token_count} | salida: {uso.candidates_token_count}")

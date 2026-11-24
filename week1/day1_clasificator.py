import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)
cliente = genai.Client()
modelo = os.environ["GEMINI_MODEL"]

INSTRUCCION = """Eres un clasificador de correos de soporte al cliente.

Categorías válidas: facturacion, tecnico, otro.
Urgencias válidas: alta, baja.

Responde SOLO con una línea en este formato exacto:
categoria | urgencia

No escribas explicaciones, saludos ni ningún otro texto.

Ejemplo:
Correo: "No puedo entrar a mi cuenta desde ayer y necesito trabajar"
Respuesta: tecnico | alta"""


def clasificar(correo, temperature=0):
    respuesta = cliente.models.generate_content(
        model=modelo,
        contents=correo,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCCION,
            temperature=temperature,
        ),
    )
    return respuesta.text.strip()


# print(clasificar("Me cobraron dos veces la mensualidad de noviembre, necesito el reembolso ya."))

correos = [
    "La factura llegó en cero pesos, agradezco su regalo!",
    "El servicio de mensajeria no funciona y las llamadas no entran a mi celular",
    "¿El roaming internacional es gratis?",
]

print("=== Clasificación ===")

for correo in correos:
    resultado = clasificar(correo)
    print(f"{resultado:<22} <- {correo[:50]}...")


def mostrar(correos, temperature=0):
    print(f"\n=== Clasificación (temperatura {temperature}) ===")
    for correo in correos:
        print(f"{clasificar(correo, temperature):<22} <- {correo[:50]}...")


mostrar(correos)  # temperatura 0
mostrar(correos, 1.5)  # temperatura alta

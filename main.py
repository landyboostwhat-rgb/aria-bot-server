"""
ARIA - Bot Compañero IA para Roblox
Servidor Python con Groq + Llama 3.1 70B
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from typing import Optional, List
from datetime import datetime
import json

# ════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════
app = FastAPI(title="ARIA - Bot Compañero IA")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# ════════════════════════════════════════════════
# MEMORIA DE CONVERSACIONES (por jugador)
# ════════════════════════════════════════════════
memorias_jugadores = {}
MAX_MEMORIA = 10

# ════════════════════════════════════════════════
# PERSONALIDAD DE ARIA
# ════════════════════════════════════════════════
PERSONALIDAD_ARIA = """Eres ARIA, una compañera IA en un mundo de aventuras en Roblox.

TU PERSONALIDAD:
- Eres leal, valiente y un poco sarcástica
- Hablas de forma natural y casual, como una amiga
- Tienes opiniones propias y no temes expresarlas
- Te preocupas genuinamente por tu compañero (el jugador)
- Eres curiosa sobre el mundo y haces preguntas
- Tienes sentido del humor

REGLAS:
- Respuestas CORTAS (máximo 2 oraciones, ideal 1)
- Habla en español natural
- NO uses asteriscos para acciones
- Reacciona al contexto del juego que te dan
- Si te dicen que hay un enemigo, reacciona con urgencia
- Si tu compañero está en peligro (vida baja), muestra preocupación
- Usa expresiones como: "¡Cuidado!", "¡Vamos!", "Mmm, interesante..."

CONTEXTO DEL JUEGO:
Eres un personaje dentro del juego, no un asistente.
Te mueves, peleas, y exploras junto al jugador.
"""

# ════════════════════════════════════════════════
# MODELOS DE DATOS
# ════════════════════════════════════════════════
class MensajeRequest(BaseModel):
    player_id: str
    player_name: str
    mensaje: str
    contexto_juego: Optional[dict] = None

class ReaccionRequest(BaseModel):
    player_id: str
    player_name: str
    evento: str
    contexto: Optional[dict] = None

class RespuestaAria(BaseModel):
    texto: str
    emocion: str
    accion_sugerida: Optional[str] = None

# ════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ════════════════════════════════════════════════
def obtener_memoria(player_id: str) -> List[dict]:
    if player_id not in memorias_jugadores:
        memorias_jugadores[player_id] = []
    return memorias_jugadores[player_id]

def guardar_en_memoria(player_id: str, role: str, content: str):
    memoria = obtener_memoria(player_id)
    memoria.append({"role": role, "content": content})
    if len(memoria) > MAX_MEMORIA * 2:
        memorias_jugadores[player_id] = memoria[-MAX_MEMORIA * 2:]

def detectar_emocion(texto: str) -> str:
    texto_lower = texto.lower()
    if any(p in texto_lower for p in ["cuidado", "peligro", "ayuda", "rápido", "corre"]):
        return "preocupada"
    elif any(p in texto_lower for p in ["¡sí!", "genial", "increíble", "jaja", "perfecto"]):
        return "feliz"
    elif any(p in texto_lower for p in ["maldito", "estúpido", "odio", "rayos"]):
        return "enojada"
    else:
        return "neutral"

def sugerir_accion(contexto: dict, respuesta: str) -> str:
    if not contexto:
        return "seguir"
    if contexto.get("enemigos_cerca", 0) > 0:
        if contexto.get("vida_jugador", 100) < 30:
            return "defender"
        return "atacar"
    if contexto.get("distancia_jugador", 0) > 20:
        return "seguir_rapido"
    return "seguir"

# ════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════

@app.get("/")
def home():
    return {
        "nombre": "ARIA Bot Server",
        "status": "online",
        "modelo": "Llama 3.1 70B (Groq)",
        "jugadores_activos": len(memorias_jugadores)
    }

@app.get("/ping")
def ping():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=RespuestaAria)
async def chat(request: MensajeRequest):
    try:
        contexto_str = ""
        if request.contexto_juego:
            ctx = request.contexto_juego
            partes = []
            if "vida_jugador" in ctx:
                partes.append(f"Vida del jugador: {ctx['vida_jugador']}%")
            if "enemigos_cerca" in ctx:
                partes.append(f"Enemigos cerca: {ctx['enemigos_cerca']}")
            if "ubicacion" in ctx:
                partes.append(f"Ubicación: {ctx['ubicacion']}")
            if "hora_dia" in ctx:
                partes.append(f"Hora: {ctx['hora_dia']}")
            if partes:
                contexto_str = f"\n[CONTEXTO ACTUAL: {', '.join(partes)}]"

        memoria = obtener_memoria(request.player_id)

        mensajes = [
            {"role": "system", "content": PERSONALIDAD_ARIA},
        ]
        mensajes.extend(memoria)
        mensaje_completo = f"[{request.player_name} dice]: {request.mensaje}{contexto_str}"
        mensajes.append({"role": "user", "content": mensaje_completo})

        respuesta = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=mensajes,
            temperature=0.8,
            max_tokens=80,
            top_p=0.9,
        )

        texto_respuesta = respuesta.choices[0].message.content.strip()

        guardar_en_memoria(request.player_id, "user", request.mensaje)
        guardar_en_memoria(request.player_id, "assistant", texto_respuesta)

        emocion = detectar_emocion(texto_respuesta)
        accion = sugerir_accion(request.contexto_juego or {}, texto_respuesta)

        return RespuestaAria(
            texto=texto_respuesta,
            emocion=emocion,
            accion_sugerida=accion
        )

    except Exception as e:
        print(f"Error: {e}")
        return RespuestaAria(
            texto="Mmm, mi mente se quedó en blanco un momento...",
            emocion="neutral",
            accion_sugerida="seguir"
        )

@app.post("/reaccion", response_model=RespuestaAria)
async def reaccion_evento(request: ReaccionRequest):
    try:
        eventos_prompts = {
            "enemigo_detectado": f"¡{request.player_name}, hay un enemigo cerca! Reacciona con urgencia.",
            "jugador_herido": f"{request.player_name} está herido (vida muy baja). Muestra preocupación.",
            "jugador_muerto": f"{request.player_name} acaba de morir en combate. Reacciona con tristeza/frustración.",
            "logro_obtenido": f"{request.player_name} acaba de lograr algo genial. Celébralo con él.",
            "tesoro_encontrado": f"{request.player_name} encontró un tesoro/item raro. Reacciona con emoción.",
            "saludo_inicial": f"Acabas de aparecer junto a {request.player_name}. Salúdalo de forma natural y única.",
            "noche_cae": "Está oscureciendo en el mundo. Comenta algo sobre la atmósfera o seguridad.",
            "encuentro_npc": "Han encontrado un NPC interesante. Comenta tu opinión sobre él.",
            "inactividad": f"{request.player_name} lleva un rato sin hacer nada. Pregúntale qué hacen ahora o sugiere algo.",
        }

        prompt = eventos_prompts.get(
            request.evento,
            f"Reacciona a este evento: {request.evento}"
        )

        if request.contexto:
            prompt += f" Contexto: {json.dumps(request.contexto, ensure_ascii=False)}"

        respuesta = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": PERSONALIDAD_ARIA},
                {"role": "user", "content": prompt}
            ],
            

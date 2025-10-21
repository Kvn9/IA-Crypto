# server.py
# pip install fastapi uvicorn websockets tensorflow joblib numpy
import asyncio, json, numpy as np, joblib
from fastapi import FastAPI, WebSocket
from tensorflow.keras.models import load_model
import websockets  # client ws pour Binance

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"

app = FastAPI()
model = load_model("model.h5")
assets = joblib.load("preproc.pkl")
scaler = assets["scaler"]; look_back = assets["look_back"]

# Buffer glissant de closes (en non-scalé et scalé)
buffer_raw = []
buffer_scaled = []

clients = set()

async def binance_loop():
    global buffer_raw, buffer_scaled
    async with websockets.connect(BINANCE_WS) as ws:
        async for msg in ws:
            data = json.loads(msg)
            k = data.get("k", {})
            if not k or not k.get("x"):  # x==True => bougie close
                continue
            ts = k["T"]    # close time (ms)
            close = float(k["c"])
            # maj buffers
            buffer_raw.append(close)
            if len(buffer_raw) >= look_back:
                # scaler: fit une fois au démarrage sur la 1ère fenêtre, puis partial update facultatif
                if len(buffer_raw) == look_back:
                    arr = np.array(buffer_raw, dtype="float32").reshape(-1,1)
                    buffer_scaled = scaler.transform(arr).tolist()
                else:
                    # ajouter dernier point
                    last_scaled = float(scaler.transform([[close]])[0,0])
                    buffer_scaled.append([last_scaled])
                    buffer_scaled = buffer_scaled[-look_back:]

                # prédiction
                x = np.array(buffer_scaled, dtype="float32").reshape(1, look_back, 1)
                yhat_s = model.predict(x, verbose=0)
                yhat = scaler.inverse_transform(yhat_s)[0,0]

                # broadcast
                payload = {
                    "t": ts,               # unix ms
                    "close": close,        # prix réel de la bougie qui ferme
                    "pred_next": yhat      # prédiction pour la prochaine bougie
                }
                await broadcast(payload)

async def broadcast(payload):
    dead = []
    msg = json.dumps(payload)
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.discard(d)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)  # keepalive minimal
    except Exception:
        pass
    finally:
        clients.discard(ws)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(binance_loop())

# Lancer: uvicorn server:app --reload --port 8000
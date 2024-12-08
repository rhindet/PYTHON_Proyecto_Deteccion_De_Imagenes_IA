from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Acepta la conexión
    while True:
        message = await websocket.receive_text()  # Recibe un mensaje del cliente
        response = f"Hola, {message}!"  # Crea la respuesta
        await websocket.send_text(response)  # Envía la respuesta al cliente

# Para ejecutar el servidor, usa Uvicorn con el host y puerto deseados
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=90)

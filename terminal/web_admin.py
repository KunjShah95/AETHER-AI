from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import os
import json
import psutil

app = FastAPI(title="Aether AI Admin")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Aether AI Admin</title>
            <style>
                body { font-family: sans-serif; padding: 20px; background: #1a1a1a; color: #fff; }
                .card { background: #333; padding: 20px; margin: 10px 0; border-radius: 8px; }
                h1 { color: #00ff9d; }
            </style>
        </head>
        <body>
            <h1>Aether AI Admin Panel</h1>
            <div class="card">
                <h2>System Status</h2>
                <div id="status">Loading...</div>
            </div>
            <script>
                fetch('/api/status').then(r => r.json()).then(data => {
                    document.getElementById('status').innerHTML = `
                        CPU: ${data.cpu}%<br>
                        RAM: ${data.ram}%<br>
                        OS: ${data.os}
                    `;
                });
            </script>
        </body>
    </html>
    """

@app.get("/api/status")
async def get_status():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "os": os.name
    }

@app.get("/api/logs")
async def get_logs():
    log_path = "ai_assistant.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
            return {"logs": lines[-50:]}
    return {"logs": []}

def start_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()

uvicorn weather_radar.server.__main__:server \
  --workers 16 \
  --port 8000 \
  --host 0.0.0.0

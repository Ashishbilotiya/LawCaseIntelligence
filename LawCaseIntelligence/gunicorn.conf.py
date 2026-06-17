# LawCaseIntelligence — Gunicorn Configuration
import os

# Server socket — Render injects PORT=10000
bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"

# Worker class — gevent required for flask-socketio async_mode="gevent"
# geventwebsocket worker was wrong and crashes on startup
worker_class = "gevent"

# 1 worker only — SocketIO shared state must stay in one process
workers = 1

# Gevent greenlet pool size per worker
worker_connections = 1000

# Timeout generous for PDF pipeline (6-agent LLM calls can take 3-5 min)
timeout      = 300
keepalive    = 5

# Logging — stdout/stderr for Render log drain
loglevel          = os.getenv("LOG_LEVEL", "info").lower()
accesslog         = "-"
errorlog          = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sμs'

# Process naming
proc_name = "lawcase-intelligence"

# Graceful restarts
graceful_timeout    = 120
max_requests        = 500
max_requests_jitter = 50

# Do NOT preload — incompatible with gevent + SocketIO
preload_app = False

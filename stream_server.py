"""Live-streaming server: ZeroMQ SUB -> WebSocket relay (Flask + flask-sock).

    python producer.py              # terminal 1: publishes frames on tcp://*:5556
    python stream_server.py         # terminal 2: open http://localhost:8000/

Also serves the viewer and the same REST API as server.py (/api/fields...).

Endpoints
  GET  /api/stream       status: {"endpoint", "frames", "fps", "bytes", "age"}
  WS   /ws/field         client sends "next" -> server replies with the newest frame
                         (a text meta message precedes it when meta changes)

Flow control is pull-based and "latest wins": the relay keeps only the newest frame, and each
client gets a frame only after asking for it. A slow client simply skips intermediate frames;
nothing is queued, so latency stays at one frame regardless of client speed.
"""
import argparse
import threading
import time
from pathlib import Path

import zmq
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_sock import Sock

import server as rest   # reuse generators and file writers

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
sock = Sock(app)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}


class Latest:
    """Newest frame from the producer(s) + a condition variable to wake waiting clients."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0            # relay counter (independent of producer seq)
        self.meta = None
        self.frame = None
        self.time = None
        self.fps = 0.0

    def put(self, meta, frame):
        with self.cond:
            now = time.monotonic()
            if self.time is not None:
                dt = now - self.time
                self.fps = 0.8 * self.fps + 0.2 / dt if dt > 0 else self.fps
            self.seq += 1
            self.meta, self.frame, self.time = meta, frame, now
            self.cond.notify_all()

    def wait_newer(self, seq, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq, timeout)
            return self.seq, self.meta, self.frame


latest = Latest()


def zmq_loop(endpoint, bind):
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVHWM, 4)
    (s.bind if bind else s.connect)(endpoint)
    while True:
        parts = s.recv_multipart()       # this loop only stores, so it never falls behind
        if len(parts) == 2:
            latest.put(parts[0], parts[1])


class _LockedSocket:
    """simple_websocket writes from two threads (our sends and its ping/close-handling thread)
    using sock.send(), which can interleave or partially write large frames and corrupt the
    stream. Serialize writes and use sendall()."""

    def __init__(self, sock):
        self._sock, self._lock = sock, threading.Lock()

    def send(self, data):
        with self._lock:
            self._sock.sendall(data)
        return len(data)

    def __getattr__(self, name):
        return getattr(self._sock, name)


@sock.route("/ws/field")
def ws_field(ws):
    ws.sock = _LockedSocket(ws.sock)
    try:
        _serve_client(ws)
    finally:
        # simple_websocket flags the connection closed *before* its thread sends the close reply;
        # wait for it, otherwise Werkzeug's trailing HTTP response can reach the browser first
        # ("Invalid frame header" / unclean close).
        if not ws.connected:
            ws.thread.join(timeout=2)


def _serve_client(ws):
    sent, meta_sent = 0, None
    while True:
        if ws.receive() is None:         # wait for the client's "next"
            return
        while True:
            seq, meta, frame = latest.wait_newer(sent, timeout=5)
            if seq > sent:
                break
            if not ws.connected:
                return
        if meta != meta_sent:
            ws.send(meta.decode())
            meta_sent = meta
        ws.send(frame)
        sent = seq


@app.get("/api/stream")
def stream_status():
    age = None if latest.time is None else round(time.monotonic() - latest.time, 2)
    return jsonify(endpoint=app.config["ZMQ_ENDPOINT"], frames=latest.seq, fps=round(latest.fps, 2),
                   bytes=len(latest.frame or b""), age=age)


@app.get("/api/fields")
def fields():
    return jsonify(rest.list_generators())


@app.get("/api/fields/<gen_id>")
def field(gen_id):
    status, ctype, body, headers = rest.field_response(gen_id, request.args.to_dict())
    return Response(body, status=status, content_type=ctype, headers=headers)


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/docs/<path:p>")
def docs(p):
    return send_from_directory(ROOT / "docs", p)


@app.get("/examples/<path:p>")
def examples(p):
    return send_from_directory(ROOT / "examples", p)


@app.get("/vendor/<path:p>")
def vendor(p):
    return send_from_directory(ROOT / "vendor", p)


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


class _NoTrailingHttpForWebSockets:
    """With Werkzeug's dev server, flask-sock lets an ordinary HTTP response be written after the
    WebSocket session ends; browsers read it as garbage ("Invalid frame header", close code 1006).
    Run the handler, discard that response and report the connection as dropped instead."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        if environ.get("HTTP_UPGRADE", "").lower() == "websocket" and "werkzeug.socket" in environ:
            self.wsgi_app(environ, lambda *a, **k: (lambda data: None))
            raise ConnectionError("websocket closed")   # Werkzeug: silently close the socket
        return self.wsgi_app(environ, start_response)


app.wsgi_app = _NoTrailingHttpForWebSockets(app.wsgi_app)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZeroMQ -> WebSocket relay for field-viz")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices (e.g. phones)")
    ap.add_argument("--zmq", default="tcp://localhost:5556", help="producer endpoint to subscribe to")
    ap.add_argument("--bind", action="store_true", help="bind the SUB socket instead of connecting "
                    "(lets several producers connect to the relay)")
    args = ap.parse_args()
    app.config["ZMQ_ENDPOINT"] = args.zmq
    threading.Thread(target=zmq_loop, args=(args.zmq, args.bind), daemon=True).start()
    print(f"viewer: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/   "
          f"subscribing to {args.zmq}")
    app.run(host=args.host, port=args.port, threaded=True)

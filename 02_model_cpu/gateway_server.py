"""
TinyCardio Rural Telemedicine Gateway Server
============================================
Lightweight HTTP/REST service for frontline community health posts and clinic gateways.
Requires zero external web frameworks (built on pure Python http.server standard library).
Provides:
  - Web Oscilloscope Dashboard at http://localhost:8080/
  - REST API endpoint for instantaneous window prediction: POST /api/predict
  - Real-time patient telemetry streaming: GET /api/stream_patient
  - Hardware & diagnostic status: GET /api/status
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Ensure parent and local directory are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
CPU_DIR = os.path.dirname(__file__)
if CPU_DIR not in sys.path:
    sys.path.insert(0, CPU_DIR)

from utils.ecg_loader import load_ecg
from inference_cpu import TinyCardioCPU

# Initialize global model
WEIGHTS_PATH = os.path.join(CPU_DIR, 'model_weights', 'tinycardio_cpu_weights.npz')
MODEL = TinyCardioCPU(WEIGHTS_PATH)

# Cache for simulated patient streams
PATIENT_CACHE = {}


def get_cached_patient(record_path: str):
    """Loads and caches patient ECG signal in memory at 250 Hz."""
    if record_path not in PATIENT_CACHE:
        full_p = os.path.join(BASE_DIR, record_path) if not os.path.isabs(record_path) else record_path
        rec = load_ecg(full_p, n_samples=None, physical=True)
        fs = rec.header.get('sampling_rate', 250.0)
        ch0 = rec.signals[0]
        if fs != 250.0:
            from scipy import signal
            target_n = int(len(ch0) * (250.0 / fs))
            sig = signal.resample(ch0, target_n)
        else:
            sig = np.array(ch0, dtype=np.float32)
        PATIENT_CACHE[record_path] = sig
    return PATIENT_CACHE[record_path]


class TinyCardioHandler(BaseHTTPRequestHandler):

    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # 1. Web Dashboard
        if path == '/' or path == '/index.html':
            html_file = os.path.join(CPU_DIR, 'web', 'index.html')
            if os.path.exists(html_file):
                with open(html_file, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Dashboard HTML not found")
            return

        # 2. System Status API
        if path == '/api/status':
            self._send_json(200, {
                'system': 'TinyCardio Rural Gateway',
                'status': 'ONLINE',
                'engine': 'Pure CPU (NumPy Deterministic)',
                'model_parameters': 465,
                'tensor_arena_ram': '< 2.5 KB',
                'supported_sampling_rate': '250 Hz',
                'threshold': MODEL.threshold
            })
            return

        # 3. Patient Telemetry Stream API
        if path == '/api/stream_patient':
            rec_name = params.get('record', ['data/vfdb/418'])[0]
            try:
                idx = int(params.get('index', [0])[0])
            except ValueError:
                idx = 0

            try:
                sig = get_cached_patient(rec_name)
            except Exception as e:
                self._send_json(404, {'error': f"Failed to load record {rec_name}: {str(e)}"})
                return

            window_size = 250
            step_size = 125
            start = idx * step_size

            if start + window_size > len(sig):
                self._send_json(200, {'finished': True})
                return

            window = sig[start:start + window_size]
            t_sec = start / 250.0
            res = MODEL.predict_window(window)

            self._send_json(200, {
                'finished': False,
                'window_index': idx,
                'time_seconds': round(t_sec, 2),
                'risk_score': res['risk_score'],
                'classification': res['classification'],
                'is_alert': res['is_alert'],
                'latency_ms': res['latency_ms'],
                # Downsample 250 samples to 25 points for efficient browser transmission
                'samples': [round(float(s), 3) for s in window[::10]]
            })
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode('utf-8'))
                raw_sig = np.array(payload.get('signal', []), dtype=np.float32)
                if len(raw_sig) < 250:
                    self._send_json(400, {'error': f"Input signal must have at least 250 samples, got {len(raw_sig)}"})
                    return
                # Take first 250 samples
                res = MODEL.predict_window(raw_sig[:250])
                self._send_json(200, res)
            except Exception as e:
                self._send_json(400, {'error': f"Invalid JSON payload: {str(e)}"})
            return

        self.send_error(404, "Not Found")


def run_server(port: int = 8080):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, TinyCardioHandler)
    print("=" * 65)
    print(f"TINYCARDIO RURAL TELEMEDICINE GATEWAY ONLINE")
    print(f"Web Dashboard: http://localhost:{port}/")
    print(f"REST API:      http://localhost:{port}/api/predict")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nGateway server stopped gracefully.")
        httpd.server_close()


if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)

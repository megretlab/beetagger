"""Subprocess worker for the Inference tab. Runs inside YOUR python env (torch etc.).

    python runner.py <model_script.py> [weights_path]

Protocol (JSON lines): prints {"ready":true} once loaded; then for each stdin line
{"id":N,"paths":[...]} prints {"id":N,"results":[{...}, ...]} (one dict per path).
Model script contract: see models/example_model.py
"""
import sys, json, importlib.util, inspect, traceback

proto = sys.stdout
sys.stdout = sys.stderr          # stray prints from model code must not corrupt the protocol


def send(o):
    proto.write(json.dumps(o) + "\n"); proto.flush()


try:
    script, weights = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    spec = importlib.util.spec_from_file_location("user_model", script)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model = mod.load(weights) if hasattr(mod, "load") else None
    two_arg = len(inspect.signature(mod.predict).parameters) >= 2
    send({"ready": True})
except Exception:
    send({"error": traceback.format_exc()}); sys.exit(1)

for line in sys.stdin:
    req = json.loads(line)
    try:
        res = mod.predict(model, req["paths"]) if two_arg else mod.predict(req["paths"])
        send({"id": req["id"], "results": list(res)})
    except Exception:
        send({"id": req["id"], "error": traceback.format_exc()})

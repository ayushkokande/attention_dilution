"""Minimal Jupyter client: run shell commands on the pod through the RunPod HTTPS proxy."""
import json, os, sys, time, uuid, requests, websocket
BASE = os.environ["POD_URL"]  # e.g. https://<pod-id>-8888.proxy.runpod.net
PW = open(os.path.expanduser("~/.secrets/jupyter_pw")).read().strip()
S = requests.Session()
def login():
    r = S.get(f"{BASE}/login", timeout=30)
    xsrf = S.cookies.get("_xsrf")
    r = S.post(f"{BASE}/login", data={"password": PW, "_xsrf": xsrf}, timeout=30, allow_redirects=True)
    S.headers["X-XSRFToken"] = S.cookies.get("_xsrf") or ""
    r = S.get(f"{BASE}/api/kernels", timeout=30); r.raise_for_status()
    return r.json()
def kernel():
    ks = S.get(f"{BASE}/api/kernels", timeout=30).json()
    if ks: return ks[0]["id"]
    return S.post(f"{BASE}/api/kernels", json={"name": "python3"}, timeout=60).json()["id"]
def execute(code, timeout=600):
    kid = kernel()
    ws_url = BASE.replace("https", "wss") + f"/api/kernels/{kid}/channels"
    cookie = "; ".join(f"{k}={v}" for k, v in S.cookies.items())
    proxy = os.environ.get("HTTPS_PROXY", "").replace("http://", "")
    ph, pp = (proxy.split(":") + [None])[:2] if proxy else (None, None)
    ws = websocket.create_connection(ws_url, header=[f"Cookie: {cookie}", f"X-XSRFToken: {S.headers['X-XSRFToken']}"],
                                     timeout=timeout, http_proxy_host=ph, http_proxy_port=int(pp) if pp else None,
                                     proxy_type="http" if ph else None,
                                     sslopt={"ca_certs": os.path.expanduser("~/.ccr/ca-bundle.crt")})
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "username": "c", "session": uuid.uuid4().hex, "msg_type": "execute_request", "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False}, "channel": "shell"}))
    out = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = json.loads(ws.recv())
        if m.get("parent_header", {}).get("msg_id") != mid: continue
        t = m["msg_type"]
        if t == "stream": out.append(m["content"]["text"])
        elif t == "error": out.append("\n".join(m["content"]["traceback"]))
        elif t in ("execute_result", "display_data"): out.append(m["content"]["data"].get("text/plain", ""))
        elif t == "status" and m["content"]["execution_state"] == "idle": break
    ws.close()
    return "".join(out)
def sh(cmd, timeout=600):
    code = f"import subprocess;r=subprocess.run({cmd!r},shell=True,capture_output=True,text=True,timeout={timeout});print(r.stdout[-6000:]);print(r.stderr[-3000:])"
    return execute(code, timeout + 30)
def upload(local, remote):
    import base64
    data = base64.b64encode(open(local, "rb").read()).decode()
    return execute(f"import base64,os;os.makedirs(os.path.dirname({remote!r}),exist_ok=True);open({remote!r},'wb').write(base64.b64decode({data!r}));print('ok')")
if __name__ == "__main__":
    login()
    print(sh(" ".join(sys.argv[1:])))
def download(remote, local):
    import base64, zlib
    out = execute(f"import base64,zlib;print(base64.b64encode(zlib.compress(open({remote!r},'rb').read())).decode())", 900)
    open(local, "wb").write(zlib.decompress(base64.b64decode(out.strip())))
    return local
def download2(remote, local):
    """gzip on the pod, then fetch through Jupyter's /files endpoint (no output-size limits)."""
    import gzip, shutil
    import uuid
    tmp = f"/workspace/_dl_{uuid.uuid4().hex[:8]}.gz"
    execute(f"import gzip,shutil;shutil.copyfileobj(open({remote!r},'rb'),gzip.open({tmp!r},'wb'))")
    root = execute("import os;print(os.getcwd())").strip()
    rel = os.path.relpath(tmp, root)
    r = S.get(f"{BASE}/files/{rel}", timeout=600)
    r.raise_for_status()
    open(local + ".gz", "wb").write(r.content)
    with gzip.open(local + ".gz") as f, open(local, "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(local + ".gz")
    execute(f"import os;os.remove({tmp!r})")
    return local

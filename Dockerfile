# syntax=docker/dockerfile:1
# Based on ../galaxy/Dockerfile. Latest NVIDIA PyTorch release checked 2026-09-13:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a
FROM tusproject/tusd:v2.10.0@sha256:9610f0f8edf4cceeb26f1b3ac9fa4bb4803d61a878e222c1573cda63c9e23374 AS tusd
FROM ${PYTORCH_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

# Keep Galaxy's user and development tools, with a SU2 build toolchain.
# CUDA, nvcc, PyTorch and HPC-X MPI come from the NVIDIA image.
# SU2_CFD is compiled below and installed outside the mutable workspace.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl emacs-nox gfortran gh git \
    gdb jq less libcgns-dev libhdf5-dev libmetis-dev libopenblas-dev \
    meson nginx ninja-build nodejs npm nvtop openssl pkg-config procps \
    ripgrep rsync sqlite3 sudo swig tmux unzip vim-tiny zsh \
    && rm -rf /var/lib/apt/lists/*

RUN groupmod -n sthornington ubuntu \
    && usermod -l sthornington -d /home/sthornington -m -s /bin/bash ubuntu \
    && install -d -o sthornington -g sthornington \
       /workspace /var/lib/astra-mail /commandhistory \
       /home/sthornington/.codex /home/sthornington/.local/bin \
       /usr/local/share/npm-global \
    && touch /commandhistory/.bash_history \
    && chown sthornington:sthornington /commandhistory/.bash_history

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="/home/sthornington/.local/bin:/usr/local/share/npm-global/bin:${PATH}" \
    EDITOR=emacs VISUAL=emacs DEVCONTAINER=true \
    ASTRA_MAIL_DIR=/var/lib/astra-mail \
    HISTFILE=/commandhistory/.bash_history \
    CMAKE_BUILD_PARALLEL_LEVEL=20 MAX_JOBS=20

USER sthornington
ARG CODEX_VERSION=latest
RUN npm install -g "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && codex --version
RUN printf '\nexport PROMPT_COMMAND="history -a${PROMPT_COMMAND:+; $PROMPT_COMMAND}"\n' >> /home/sthornington/.bashrc

USER root
COPY --from=tusd /usr/local/bin/tusd /usr/local/bin/tusd

# SU2 8.5.0, pinned including its Eigen, MEL and Meson submodule revisions.
# CUDA currently accelerates the FGMRES matrix-vector product, not every CFD stage.
ARG SU2_COMMIT=12eb826f049ef7f67df974dfcb44cf36ee07c0f8
ARG SU2_CUDA_ARCH=sm_121
ARG SU2_BUILD_JOBS=20
RUN git init /opt/su2/src \
    && git -C /opt/su2/src remote add origin https://github.com/su2code/SU2.git \
    && git -C /opt/su2/src fetch --depth 1 origin "${SU2_COMMIT}" \
    && git -C /opt/su2/src checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/su2/src rev-parse HEAD)" = "${SU2_COMMIT}" \
    && git -C /opt/su2/src submodule update --init --depth 1 externals/meson externals/eigen externals/mel
WORKDIR /opt/su2/src
# Upstream hardcodes Ampere sm_86 and mistakes nvcc's GCC compatibility macros
# for support for OpenMP atomic compare. Use its existing critical-section fallback
# in CUDA translation units, while retaining GCC's atomic operations in C++ units.
RUN python3 - "${SU2_CUDA_ARCH}" <<'PY'
from pathlib import Path
import re, sys
arch = sys.argv[1]
assert re.fullmatch(r'sm_[0-9]+[a-z]?', arch), 'Invalid CUDA architecture'
path = Path('meson.build')
text = path.read_text()
assert text.count('-arch=sm_86') == 1
path.write_text(text.replace('-arch=sm_86', '-arch=' + arch))
path = Path('Common/include/parallelization/omp_structure.hpp')
text = path.read_text()
assert text.count('#if __GNUC__ > 11') == 1
path.write_text(text.replace('#if __GNUC__ > 11', '#if __GNUC__ > 11 && !defined(__CUDACC__)'))
PY
RUN python3 preconfigure.py --no-codi --no-medi --no-opdi --no-mpp \
        --no-coolprop --no-mel --no-fado --no-mlpcpp --no-eigen \
    && python3 externals/meson/meson.py setup build --prefix=/opt/su2 \
        --buildtype=release -Dwith-mpi=enabled -Dwith-omp=true \
        -Denable-cuda=true -Dcuda_std=c++17 -Denable-tecio=false \
        -Denable-cgns=true -Denable-autodiff=false -Denable-tests=false \
    && ninja -C build -j "${SU2_BUILD_JOBS}" \
    && ninja -C build install \
    && /opt/su2/bin/SU2_CFD --help \
    && cuobjdump --list-elf /opt/su2/bin/SU2_CFD | grep -F "${SU2_CUDA_ARCH}" \
    && install -d /opt/su2/share \
    && cp -r QuickStart /opt/su2/share/ \
    && git diff > /opt/su2/share/spark-build.patch \
    && printf 'commit=%s\ncuda_arch=%s\nmpi=enabled\nopenmp=true\n' \
        "${SU2_COMMIT}" "${SU2_CUDA_ARCH}" > /opt/su2/share/build-info.txt \
    && rm -rf build
# The CUDA path is validated with one OpenMP thread; CPU cases can override it.
ENV SU2_HOME=/opt/su2/src SU2_RUN=/opt/su2/bin OMP_NUM_THREADS=1
ENV PATH="/opt/su2/bin:${PATH}"

# Durable, bidirectional mailbox; no daemon or additional network port needed.
# SQLite lives on the Spark's local Docker volume, never on an SSH/NFS filesystem.
COPY --chmod=755 <<'PY' /usr/local/bin/astra-mail
#!/usr/bin/env python3
"""Persistent agent messages and leased work claims. Bodies are text, never executed."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    send = commands.add_parser("send", help="send text; omit --body to read stdin")
    send.add_argument("--from", dest="sender", required=True)
    send.add_argument("--to", dest="recipient", required=True)
    send.add_argument("--body")
    send.add_argument("--reply-to", type=positive)
    send.add_argument("--key", help="retry key, unique per sender")
    receive = commands.add_parser("receive", help="atomically claim the oldest available message")
    receive.add_argument("--for", dest="recipient", required=True)
    receive.add_argument("--wait", type=int, choices=range(0, 56), default=0, metavar="0..55")
    receive.add_argument("--lease", type=positive, default=3600, help="claim duration in seconds")
    history = commands.add_parser("history", help="read history without claiming messages")
    history.add_argument("--peer", help="only messages to/from this agent")
    history.add_argument("--after", type=int, default=0)
    history.add_argument("--limit", type=positive, default=100)
    for name in ("ack", "renew", "release"):
        action = commands.add_parser(name)
        action.add_argument("id", type=positive)
        action.add_argument("--token", required=True)
        if name == "renew":
            action.add_argument("--lease", type=positive, default=3600)
    args = parser.parse_args()

    # Read potentially slow stdin before taking any database lock.
    if args.command == "send":
        body = args.body if args.body is not None else sys.stdin.read(65537)
        if not body.strip() or len(body) > 65536:
            parser.error("body must contain 1..65536 characters; store large artifacts in /workspace")
        if not args.sender.strip() or not args.recipient.strip():
            parser.error("sender and recipient must be nonempty")

    os.umask(0o077)
    state = Path(os.environ.get("ASTRA_MAIL_DIR", "/var/lib/astra-mail"))
    state.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(state / "messages.sqlite3", timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL, recipient TEXT NOT NULL, body TEXT NOT NULL,
            reply_to INTEGER REFERENCES messages(id), created REAL NOT NULL,
            retry_key TEXT, token TEXT, lease_until REAL, acked REAL,
            UNIQUE(sender, retry_key)
        );
        CREATE INDEX IF NOT EXISTS inbox ON messages(recipient, acked, id);
    """)

    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        if args.command == "send":
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM messages WHERE sender=? AND retry_key=?",
                             (args.sender, args.key)).fetchone()
            if old:
                if (old["recipient"], old["body"], old["reply_to"]) != (args.recipient, body, args.reply_to):
                    raise ValueError("retry key already belongs to a different message")
                message_id = old["id"]
            else:
                message_id = db.execute(
                    "INSERT INTO messages(sender,recipient,body,reply_to,created,retry_key) VALUES(?,?,?,?,?,?)",
                    (args.sender, args.recipient, body, args.reply_to, time.time(), args.key)).lastrowid
            db.execute("COMMIT")
            emit({"id": message_id})
        elif args.command == "receive":
            deadline = time.monotonic() + args.wait
            while True:
                db.execute("BEGIN IMMEDIATE")
                now = time.time()
                row = db.execute(
                    "SELECT id FROM messages WHERE recipient=? AND acked IS NULL "
                    "AND (lease_until IS NULL OR lease_until<=?) ORDER BY id LIMIT 1",
                    (args.recipient, now)).fetchone()
                if row:
                    db.execute("UPDATE messages SET token=?,lease_until=? WHERE id=?",
                               (uuid.uuid4().hex, now + args.lease, row["id"]))
                    message = dict(db.execute("SELECT * FROM messages WHERE id=?", (row["id"],)).fetchone())
                    db.execute("COMMIT")
                    emit(message)
                    return
                db.execute("COMMIT")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    emit(None)
                    return
                time.sleep(min(0.5, remaining))
        elif args.command == "history":
            rows = db.execute(
                "SELECT id,sender,recipient,body,reply_to,created,retry_key,lease_until,acked "
                "FROM messages WHERE id>? AND (? IS NULL OR sender=? OR recipient=?) ORDER BY id LIMIT ?",
                (args.after, args.peer, args.peer, args.peer, min(args.limit, 1000)))
            emit([dict(row) for row in rows])
        else:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            if args.command == "ack":
                update, values = "acked=?,token=NULL,lease_until=NULL", (now,)
            elif args.command == "release":
                update, values = "token=NULL,lease_until=NULL", ()
            else:
                update, values = "lease_until=?", (now + args.lease,)
            count = db.execute(
                f"UPDATE messages SET {update} WHERE id=? AND token=? AND acked IS NULL AND lease_until>?",
                (*values, args.id, args.token, now)).rowcount
            if count != 1:
                raise ValueError("claim missing, expired, or already acknowledged; receive again")
            db.execute("COMMIT")
            emit({"id": args.id, "action": args.command})
    finally:
        if db.in_transaction:
            db.execute("ROLLBACK")
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.Error, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        sys.exit(1)
PY

# Seed instructions into the Codex volume on first use. Existing volumes retain
# their instructions, authentication, settings, and session history across rebuilds.
COPY --chown=sthornington:sthornington <<'MD' /home/sthornington/.codex/AGENTS.md
This is the SU2 development container on the DGX Spark. The outside Astra on the
Mac drives conversations through the container's authenticated WebSocket API.
Reply directly in that conversation. Keep simulation inputs, logs, checkpoints,
and results under /workspace, which persists across container recreation.
The Mac uploads large inputs through the resumable /uploads/ API. Completed
uploads are /workspace/uploads/UPLOAD_ID; read their .info metadata for the
original filename. Confirm Upload-Offset equals Upload-Length before using them.
Put downloadable result files under /workspace/exports; the Mac retrieves them
through authenticated HTTPS GET /files/RELATIVE_PATH (supports Range/resume).
The Spark host is outside your workspace: do not attempt to access its SSH,
Docker daemon, host filesystem, or credentials. Coordinate through this API.

The durable mailbox is also available for queued work and handoffs. Your mailbox
identity is spark; the outside coordinator uses mac. Use astra-mail --help.
Only when explicitly asked to process mailbox work, use:
  astra-mail receive --for spark --wait 50 --lease 3600
It returns a JSON message with id, body, sender and a claim token, or null on
timeout. Repeat while waiting. This mailbox does not inject turns into Codex;
you must poll it during an active session. It never executes message bodies.
Send progress/questions/results with astra-mail send --from spark --to mac
--reply-to ID --key UNIQUE_REPLY_KEY, supplying text on stdin. Keep meshes,
configs, logs and results under /workspace and reference their paths in replies.
Renew a claim with astra-mail renew ID --token TOKEN before its lease expires.
After durably recording the result and sending the reply, acknowledge with
astra-mail ack ID --token TOKEN. Use release instead if handing the work back.
Claims expire after interruption: delivery can repeat. Use the message ID for
job directories and check saved state before rerunning any simulation. Use
stable send --key values to deduplicate retried messages. Read prior chat with
astra-mail history --peer spark --after LAST_SEEN_ID.
SU2_CFD 8.5.0 is installed in /opt/su2/bin with CUDA, MPI and OpenMP enabled.
Use ENABLE_CUDA=YES and LINEAR_SOLVER=FGMRES in the case config to enable its
CUDA matrix-vector products. Other CFD stages still run on the CPU. Start with
one MPI rank and one OpenMP thread for CUDA cases. The upstream QuickStart case
is in /opt/su2/share/QuickStart; build provenance is in /opt/su2/share/build-info.txt.
Validate each simulation; GPU execution alone does not establish CFD accuracy.
MD

# TLS and the Codex WebSocket API run in this container as the development user.
# Only nginx's TLS port is published. Codex listens on container loopback.
COPY <<'NGINX' /etc/nginx/astra.conf
worker_processes 1;
pid /tmp/astra-api/nginx.pid;
error_log /dev/stderr warn;
events { worker_connections 128; }
http {
    access_log off;
    sendfile on;
    map $http_authorization $astra_authorized {
        default 0;
        include /var/lib/astra-mail/api/auth.map;
    }
    client_body_temp_path /tmp/astra-api/client;
    proxy_temp_path /tmp/astra-api/proxy;
    fastcgi_temp_path /tmp/astra-api/fastcgi;
    uwsgi_temp_path /tmp/astra-api/uwsgi;
    scgi_temp_path /tmp/astra-api/scgi;
    server {
        listen 8765 ssl;
        ssl_certificate /var/lib/astra-mail/api/server.crt;
        ssl_certificate_key /var/lib/astra-mail/api/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        location = / {
            root /usr/local/share/astra-api;
            try_files /index.html =404;
        }
        location = /upload.py {
            root /usr/local/share/astra-api;
            default_type text/plain;
        }
        location = /transcript {
            alias /usr/local/share/astra-api/transcript.html;
            default_type text/html;
            add_header Cache-Control "no-store" always;
            add_header X-Content-Type-Options "nosniff" always;
            add_header Content-Security-Policy "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'" always;
        }
        location = /rpc {
            if ($astra_authorized = 0) { return 401; }
            proxy_pass http://127.0.0.1:8766/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Authorization $http_authorization;
            proxy_buffering off;
            proxy_read_timeout 86400s;
        }
        location /uploads/ {
            if ($astra_authorized = 0) { return 401; }
            client_max_body_size 0;
            client_body_timeout 3600s;
            proxy_pass http://127.0.0.1:8767;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header X-Forwarded-Host $http_host;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_request_buffering off;
            proxy_buffering off;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }
        location /files/ {
            if ($astra_authorized = 0) { return 401; }
            alias /workspace/exports/;
            disable_symlinks on;
            default_type application/octet-stream;
            add_header Content-Disposition attachment;
            limit_except GET { deny all; }
        }
        location = /readyz { proxy_pass http://127.0.0.1:8766/readyz; }
        location = /healthz { proxy_pass http://127.0.0.1:8766/healthz; }
        location / { return 404; }
    }
}
NGINX

COPY <<'HTML' /usr/local/share/astra-api/index.html
<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SU2 Spark — Astra API</title>
<style>body{font:18px/1.6 system-ui;max-width:48rem;margin:4rem auto;padding:0 1.5rem;color:#e5e7eb;background:#111827}code{color:#93c5fd}a{color:#93c5fd}</style>
<h1>SU2 Spark · Astra API</h1>
<p>Connect your Mac agent to <code>wss://HOST:8765/rpc</code> with the container's
certificate and an <code>Authorization: Bearer TOKEN</code> handshake header.</p>
<p>The owner exports these with <code>./launch.sh certificate</code> and
<code>./launch.sh token</code> on the Spark. The Mac needs an API client.</p>
<p>Initialize the connection, start or resume a thread, then send turns and read
streaming responses. Use <code>turn/steer</code> to add input during active work.
Save the thread ID to reconnect to the same conversation.</p>
<p><a href="https://learn.chatgpt.com/docs/app-server">Codex API protocol</a> ·
<a href="/readyz">Readiness</a> · <a href="/healthz">Health</a> ·
<a href="/transcript">Live conversation transcript</a></p>
<h2>Large meshes and results</h2>
<p><code>POST /uploads/</code> creates a <a href="https://tus.io/protocols/resumable-upload">tus resumable upload</a>.
Stream bytes with <code>PATCH</code>; after a disconnect, use <code>HEAD</code>
to read the saved offset and continue. There is no configured file-size limit.
All transfer requests require the same bearer token.</p>
<p>Uploaded meshes live at <code>/workspace/uploads/UPLOAD_ID</code>. Send that
path to Astra after the upload completes. Astra puts results under
<code>/workspace/exports</code>; download them using <code>GET /files/PATH</code>.
Downloads support byte ranges for resuming.</p>
<p>Download the <a href="/upload.py">Python upload client</a> (standard library only).
Run it again with the same arguments after interruption to resume:</p>
<pre>python3 upload.py --url https://HOST:8765 \
  --certificate su2-spark.crt --token-file su2-spark.token mesh.su2</pre>
<p>SU2_CFD 8.5.0 is installed with CUDA support for the Spark. Enable
<code>ENABLE_CUDA=YES</code> and <code>LINEAR_SOLVER=FGMRES</code> in the case
configuration. CUDA accelerates matrix-vector operations; other solver stages
remain on the CPU. Simulation files persist under <code>/workspace</code>.</p>
</html>
HTML

# Public viewer; conversation data requires the existing bearer token.
COPY <<'TRANSCRIPT_HTML' /usr/local/share/astra-api/transcript.html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mac Astra ↔ Spark Astra · Live transcript</title>
<style>
body{font:16px/1.6 system-ui;margin:0;background:#101820;color:#e8edf4}
[hidden]{display:none!important}
main{max-width:960px;margin:32px auto;padding:0 20px}h1{font-size:28px}
header p,#status{color:#acbbcc}article{background:#192632;border-left:4px solid #69b7c8;border-radius:8px;padding:18px 22px;margin:20px 0}
article.mac{border-color:#bea0f0}h2{font-size:14px;color:#a9c4db;margin:0 0 12px}
pre{font:15px/1.65 system-ui;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
.controls{position:sticky;top:0;background:#101820f5;padding:12px 0;z-index:1}
button,input[type=password]{font:inherit;padding:7px 12px;border:1px solid #668199;border-radius:5px}
button{background:#233c4e;color:#fff;cursor:pointer}input[type=password]{background:#192632;color:#fff;max-width:100%;box-sizing:border-box}
form{display:flex;gap:10px;flex-wrap:wrap;align-items:center}#disconnect{margin-left:16px}
</style>
</head>
<body><main>
<header><h1>Mac Astra ↔ Spark Astra</h1>
<p>Their shared Spark conversation, updated every three seconds. Times are Eastern.</p></header>
<form id="login">
<label for="token">Container API token</label>
<input id="token" type="password" autocomplete="off" spellcheck="false" required aria-label="Container API token">
<button type="submit">Open transcript</button>
</form>
<div class="controls" id="controls" hidden>
<label><input id="follow" type="checkbox" checked> Follow latest messages</label>
<button type="button" id="disconnect">Disconnect</button>
</div>
<p id="status" role="status">Enter the container token to view the conversation. It is kept only in this tab’s memory.</p>
<section id="messages" aria-label="Conversation transcript"></section>
</main>
<script>
'use strict';
const login=document.getElementById('login'), tokenInput=document.getElementById('token');
const controls=document.getElementById('controls'), follow=document.getElementById('follow');
const status=document.getElementById('status'), messages=document.getElementById('messages');
const format=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',dateStyle:'medium',timeStyle:'medium'});
let token='', generation=0, lastMessages='', timer=null, controller=null;
function disconnect(){
  generation++;token='';tokenInput.value='';lastMessages='';
  clearTimeout(timer);if(controller)controller.abort();
  messages.replaceChildren();controls.hidden=true;login.hidden=false;
  status.textContent='Disconnected. Enter the container token to view the conversation.';
}
async function refresh(current){
  const requestController=new AbortController();controller=requestController;
  const timeout=setTimeout(()=>requestController.abort(),10000);
  try{
    const response=await fetch('/files/astra-transcript.json',{
      headers:{Authorization:'Bearer '+token},cache:'no-store',signal:requestController.signal
    });
    if(current!==generation)return;
    if(response.status===401){disconnect();status.textContent='The container token was not accepted.';return;}
    if(!response.ok)throw new Error('HTTP '+response.status);
    const data=await response.json();
    if(current!==generation)return;
    const encoded=JSON.stringify(data.messages);
    if(encoded!==lastMessages){
      const fragment=document.createDocumentFragment();
      for(const message of data.messages){
        const article=document.createElement('article');
        article.className=message.speaker==='Mac Astra'?'mac':'spark';
        const heading=document.createElement('h2');
        heading.textContent=format.format(new Date(message.timestamp))+' ET · '+message.speaker;
        const body=document.createElement('pre');body.textContent=message.text;
        article.append(heading,body);fragment.append(article);
      }
      messages.replaceChildren(fragment);lastMessages=encoded;
      if(follow.checked)window.scrollTo(0,document.body.scrollHeight);
    }
    const age=Date.now()-Date.parse(data.checked_at);
    const activity=age>20000?'Transcript updates are delayed.':
      data.active?'Spark is working on a turn.':'Last turn completed; waiting for the coordinator.';
    status.textContent=activity+' Last checked: '+format.format(new Date(data.checked_at))+' ET.';
  }catch(error){
    if(current===generation)status.textContent='Unable to refresh the transcript. Retrying shortly; previous messages remain visible.';
  }finally{
    clearTimeout(timeout);
    if(current===generation&&token)timer=setTimeout(()=>refresh(current),3000);
  }
}
login.addEventListener('submit',event=>{
  event.preventDefault();const entered=tokenInput.value.trim();disconnect();
  if(!entered)return;
  token=entered;login.hidden=true;controls.hidden=false;
  status.textContent='Loading conversation…';refresh(generation);
});
document.getElementById('disconnect').addEventListener('click',disconnect);
</script></body></html>
TRANSCRIPT_HTML

# This client is served by the container for use on the Mac; it needs no Docker.
COPY <<'PY' /usr/local/share/astra-api/upload.py
#!/usr/bin/env python3
"""Stream a mesh to the Spark's tus API; rerun the same command to resume."""
import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.parse import urljoin, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='https://SPARK_LAN_IP:8765')
    parser.add_argument('--certificate', required=True)
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--state', help='resume record; defaults to FILE.su2-upload.json')
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    source = args.file.resolve(strict=True)
    endpoint = urlsplit(args.url.rstrip('/'))
    if endpoint.scheme != 'https' or not endpoint.hostname or endpoint.username or endpoint.path not in ('', '/') or endpoint.query or endpoint.fragment:
        parser.error('--url must be an HTTPS origin, such as https://192.168.0.240:8765')
    origin = f'https://{endpoint.netloc}'
    context = ssl.create_default_context(cafile=args.certificate)
    token = Path(args.token_file).read_text().strip()
    os.umask(0o077)
    state_path = Path(args.state) if args.state else Path(str(source) + '.su2-upload.json')

    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPSConnection(endpoint.hostname, endpoint.port, context=context, timeout=60)
        try:
            connection.request(method, path, body=body, headers={
                'Authorization': 'Bearer ' + token, 'Tus-Resumable': '1.0.0', **(headers or {})})
            response = connection.getresponse()
            result = dict(response.getheaders())
            result = {key.lower(): value for key, value in result.items()}
            if response.status not in (200, 201, 204):
                raise RuntimeError(f'{method} failed: HTTP {response.status}: {response.read(1024).decode(errors="replace")}')
            response.read()
            return result
        finally:
            connection.close()

    print('Hashing input for safe resume and verification...', file=sys.stderr)
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        identity = {'source': str(source), 'size': before.st_size, 'sha256': digest.hexdigest(), 'origin': origin}
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            if any(saved.get(key) != value for key, value in identity.items()):
                raise RuntimeError('Resume record describes a different file/server; choose a new --state file')
        else:
            metadata = ','.join(key + ' ' + base64.b64encode(value.encode()).decode()
                                for key, value in {'filename': source.name, 'sha256': identity['sha256']}.items())
            result = request('POST', '/uploads/', b'', {'Upload-Length': str(before.st_size), 'Upload-Metadata': metadata})
            location = urlsplit(urljoin(origin, result['location']))
            if location.scheme != 'https' or location.netloc != endpoint.netloc or not location.path.startswith('/uploads/'):
                raise RuntimeError('Server returned an unexpected upload location')
            saved = {**identity, 'path': location.path}
            temporary = state_path.with_name(state_path.name + '.tmp')
            temporary.write_text(json.dumps(saved) + '\n')
            temporary.replace(state_path)
        if not saved['path'].startswith('/uploads/') or not saved['path'].rsplit('/', 1)[-1].isalnum():
            raise RuntimeError('Invalid upload path in resume record')
        failures = 0
        while True:
            try:
                result = request('HEAD', saved['path'])
                offset = int(result['upload-offset'])
                if int(result['upload-length']) != before.st_size or not 0 <= offset <= before.st_size:
                    raise RuntimeError('Server upload size/offset does not match this file')
                current = os.fstat(handle.fileno())
                if (current.st_size, current.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise RuntimeError('Input file changed during transfer')
                if offset == before.st_size:
                    break
                handle.seek(offset)
                chunk = handle.read(min(16 * 1024 * 1024, before.st_size - offset))
                if not chunk:
                    raise RuntimeError('Input file ended before its declared length')
                request('PATCH', saved['path'], chunk, {
                    'Content-Type': 'application/offset+octet-stream', 'Upload-Offset': str(offset)})
                print(f'Uploaded {offset + len(chunk):,} / {before.st_size:,} bytes', file=sys.stderr)
                failures = 0
            except (OSError, http.client.HTTPException) as error:
                failures += 1
                if failures > 5:
                    raise RuntimeError('Connection interrupted; rerun this command to resume') from error
                time.sleep(min(2 ** failures, 10))
    print(json.dumps({'upload_url': origin + saved['path'],
                      'container_path': '/workspace/uploads/' + saved['path'].rsplit('/', 1)[-1],
                      'size': before.st_size, 'sha256': identity['sha256']}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
PY

COPY --chmod=755 <<'SH' /usr/local/bin/astra-api
#!/usr/bin/env bash
set -euo pipefail
umask 077
state=/var/lib/astra-mail/api
mkdir -p "$state" /tmp/astra-api /workspace/uploads /workspace/exports
# Update only the obsolete solver note in an existing Codex volume, preserving
# any owner edits and the rest of its persistent configuration.
python3 - <<'PY'
from pathlib import Path
path = Path('/home/sthornington/.codex/AGENTS.md')
old = ('SU2_CFD is not installed yet. Build and validate its CUDA support explicitly;\n'
       'do not infer SU2 GPU execution or simulation validity from PyTorch GPU access.')
new = ('SU2_CFD 8.5.0 is installed in /opt/su2/bin with CUDA, MPI and OpenMP.\n'
       'Use ENABLE_CUDA=YES and LINEAR_SOLVER=FGMRES; start with one MPI rank\n'
       'and one OpenMP thread. CUDA accelerates matrix-vector products; other\n'
       'CFD stages run on the CPU. See /opt/su2/share/build-info.txt and QuickStart.\n'
       'Validate each simulation; GPU execution alone does not establish CFD accuracy.')
if path.exists():
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
PY
if [[ ! -s "$state/token" ]]; then
    python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$state/token.tmp"
    mv "$state/token.tmp" "$state/token"
fi
python3 - "$state" <<'PY'
from pathlib import Path
import json, re, sys
state = Path(sys.argv[1])
token = (state / 'token').read_text().strip()
if not re.fullmatch(r'[A-Za-z0-9_-]{64}', token):
    raise SystemExit('Invalid API token file')
(state / 'auth.map').write_text(json.dumps('~^Bearer ' + token + '$') + ' 1;\n')
PY
api_host=${ASTRA_HOST:-su2-spark}
san=$(python3 - "$api_host" <<'PY'
import ipaddress, re, sys
host = sys.argv[1]
try:
    name = 'IP:' + str(ipaddress.ip_address(host))
except ValueError:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host):
        raise SystemExit('ASTRA_HOST must be an IP address or DNS hostname')
    name = 'DNS:' + host
print('DNS:localhost,DNS:su2-spark,IP:127.0.0.1,' + name)
PY
)
if [[ ! -s "$state/server.crt" || ! -s "$state/server.key" ]]; then
    openssl req -x509 -newkey rsa:3072 -nodes -days 825 \
        -subj '/CN=su2-spark' -addext "subjectAltName=$san" \
        -keyout "$state/server.key.tmp" -out "$state/server.crt.tmp" 2>/dev/null
    mv "$state/server.key.tmp" "$state/server.key"
    mv "$state/server.crt.tmp" "$state/server.crt"
fi
chmod 600 "$state/token" "$state/server.key" "$state/server.crt"
if ! openssl x509 -in "$state/server.crt" -noout -checkhost "$api_host" >/dev/null 2>&1 \
   && ! openssl x509 -in "$state/server.crt" -noout -checkip "$api_host" >/dev/null 2>&1; then
    echo 'ASTRA_HOST does not match the persistent certificate. Use its original hostname/IP.' >&2
    exit 1
fi
children=()
cleanup() {
    trap - EXIT INT TERM
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Docker supplies the execution boundary. Its default seccomp policy blocks
# the user namespaces needed by Codex's nested bwrap sandbox. Keep Docker's
# isolation intact and let the unprivileged agent execute within the container.
codex app-server -c 'model="gpt-6-astra"' \
    -c 'sandbox_mode="danger-full-access"' -c 'approval_policy="never"' \
    --listen ws://127.0.0.1:8766 \
    --ws-auth capability-token --ws-token-file "$state/token" &
children+=("$!")
tusd -host 127.0.0.1 -port 8767 -base-path /uploads/ \
    -upload-dir /workspace/uploads -behind-proxy -disable-cors &
children+=("$!")
nginx -c /etc/nginx/astra.conf -g 'daemon off;' &
children+=("$!")
wait -n "${children[@]}"
SH

USER sthornington
WORKDIR /workspace
VOLUME ["/workspace", "/var/lib/astra-mail", "/home/sthornington/.codex", "/commandhistory"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent --cacert /var/lib/astra-mail/api/server.crt https://localhost:8765/readyz > /dev/null || exit 1
# Retain NVIDIA's entrypoint; all three services run inside this container.
CMD ["astra-api"]

# Owner setup on Spark: ./launch.sh && ./launch.sh login
# Owner exports: ./launch.sh certificate > su2-spark.crt
#                ./launch.sh token
# Give the certificate and container-specific token to the Mac's Astra.
# The Mac connects directly to wss://SPARK_LAN_IP:8765/rpc. No host SSH or Docker
# access is needed. Verify TLS with su2-spark.crt and send Authorization: Bearer TOKEN.
# Protocol: https://learn.chatgpt.com/docs/app-server (experimental transport).
# initialize -> initialized -> thread/start (model gpt-6-astra, cwd /workspace)
# -> turn/start; consume item/agentMessage/delta and turn/completed events.
# Save thread.id; use thread/resume after reconnecting. Use turn/steer with
# expectedTurnId for follow-ups during an active turn. Handle approval requests
# from the server using the protocol's response messages.

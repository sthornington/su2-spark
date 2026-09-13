# syntax=docker/dockerfile:1
# Based on ../galaxy/Dockerfile. Latest NVIDIA PyTorch release checked 2026-09-13:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a
FROM ${PYTORCH_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

# Keep Galaxy's user and development tools, with a SU2 build toolchain.
# CUDA, nvcc, PyTorch and HPC-X MPI come from the NVIDIA image.
# SU2_CFD itself will be built and validated in the persistent workspace.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl emacs-nox gfortran gh git \
    gdb jq less libcgns-dev libhdf5-dev libmetis-dev libopenblas-dev \
    meson ninja-build nodejs npm nvtop openssh-client pkg-config procps \
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
This is the SU2 development container on the DGX Spark. Your mailbox identity is
spark; the outside coordinator on the Mac uses mac. Use astra-mail --help.
When the user asks you to wait for work, use:
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
SU2_CFD is not installed yet. Build and validate its CUDA support explicitly;
do not infer SU2 GPU execution or simulation validity from PyTorch GPU access.
MD

USER sthornington
WORKDIR /workspace
VOLUME ["/workspace", "/var/lib/astra-mail", "/home/sthornington/.codex", "/commandhistory"]
# Retain NVIDIA's entrypoint. Start Codex with docker exec when ready; mailbox
# messages remain queued while no agent is running. No credentials are baked in.
CMD ["sleep", "infinity"]

# Build/run on the Spark (reuse these named volumes when recreating the container):
# docker build --pull -t su2-spark .
# docker run -d --name su2-spark --gpus all --init --restart unless-stopped \
#   --shm-size=8g \
#   -v su2-spark-work:/workspace \
#   -v su2-spark-mail:/var/lib/astra-mail \
#   -v su2-spark-codex:/home/sthornington/.codex \
#   -v su2-spark-history:/commandhistory su2-spark
# docker exec -it su2-spark codex login --device-auth
# docker exec -it su2-spark codex -m gpt-6-astra \
#   'Wait for simulation work in your mailbox and coordinate with mac.'
#
# From the Mac, using its existing SSH access to the Spark host:
# printf '%s\n' 'Please prepare a SU2 CUDA smoke test.' | \
#   ssh SPARK_HOST docker exec -i su2-spark astra-mail send \
#     --from mac --to spark --key first-su2-smoke
# ssh SPARK_HOST docker exec su2-spark astra-mail receive --for mac --wait 50
# ssh SPARK_HOST docker exec su2-spark astra-mail ack MESSAGE_ID --token CLAIM_TOKEN
# ssh SPARK_HOST docker exec su2-spark astra-mail history --peer mac
# Use distinct --key values for new messages; reuse a key only for an exact retry.
# Artifact transfer, also through host SSH (no SSH server in this container):
# ssh SPARK_HOST docker exec su2-spark tar -C /workspace -czf - results > results.tgz

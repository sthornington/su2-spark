#!/usr/bin/env bash
# Run on the Spark. Named volumes survive container recreation.
# Mac agents use only the container's authenticated HTTPS/WebSocket port 8765.
# This owner-side script runs on the Spark; no host credentials go to the Mac agent.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
container=su2-spark
image=su2-spark:latest
owner_label=io.github.sthornington.su2-spark.launcher

usage() {
    cat <<'HELP'
Usage: ./launch.sh [start|build|login|token|certificate|astra|shell|status|stop|recreate]

  start     Start the container; build the image if missing (default).
  build     Build/update the image. Accepts extra docker build options.
  login     Sign Codex in using a browser device code.
  token     Print the container API token for the owner to give to the Mac agent.
  certificate  Export the public TLS certificate (redirect to su2-spark.crt).
  astra     Open/rejoin a local Astra session in tmux. Detach with Ctrl-b, then d.
  shell     Open an interactive Bash shell.
  status    Show container status.
  stop      Stop the container, including any running agent/simulation.
  recreate  Replace a stopped container using the current image.
            Preserves workspace, mailbox, Codex state and command history.

First use: ./launch.sh && ./launch.sh login
Export connection credentials: ./launch.sh certificate > ~/su2-spark.crt
                               (umask 077; ./launch.sh token > ~/su2-spark.token)
After rebuilding: ./launch.sh stop && ./launch.sh recreate
Recreation discards changes outside the four persistent volume directories.

SU2_BIND_IP defaults to the Spark's primary IPv4 address; set it to select an interface.
SU2_HOST defaults to that address and sets the TLS certificate name on first start.
The Mac connects directly to https://SU2_HOST:8765 (information page),
wss://SU2_HOST:8765/rpc (Codex API), /uploads/ (resumable tus uploads), and
/files/PATH (result downloads). Verify TLS with the exported certificate and
send Authorization: Bearer TOKEN. No SSH or Docker access is needed on the Mac.
HELP
}

fail() { printf '%s\n' "$*" >&2; exit 1; }
action=${1:-start}
if (( $# )); then shift; fi
case "$action" in
    -h|--help|help) usage; exit 0 ;;
    start|build|login|token|certificate|astra|shell|status|stop|recreate) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ "$action" != build && $# -ne 0 ]]; then
    fail "Unexpected arguments. Run ./launch.sh --help."
fi
command -v docker >/dev/null || fail "Docker is required. Run this script on the Spark."
docker info >/dev/null 2>&1 || fail "Cannot reach Docker. Check that Docker is running and your user has access."

exists() { docker container inspect "$container" >/dev/null 2>&1; }
state() { docker container inspect --format '{{.State.Status}}' "$container"; }
check_owner() {
    local owner
    owner=$(docker container inspect --format "{{index .Config.Labels \"$owner_label\"}}" "$container")
    [[ "$owner" == true ]] || fail "Container $container exists but was not created by this launcher; manage it with docker directly."
}
build() {
    docker build --pull --tag "$image" "$@" "$project_dir"
}
ensure_image() {
    if ! docker image inspect "$image" >/dev/null 2>&1; then build; fi
}
wait_ready() {
    local attempt health
    for attempt in {1..30}; do
        health=$(docker container inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container")
        if [[ "$health" == healthy ]]; then return; fi
        if [[ "$(state)" != running ]]; then break; fi
        sleep 1
    done
    docker logs --tail 30 "$container" >&2
    fail "The container API is not ready. Check ./launch.sh status and docker logs $container."
}
endpoint() {
    local host
    host=$(docker container inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" | sed -n 's/^ASTRA_HOST=//p')
    printf 'Container API: https://%s:8765   WebSocket: wss://%s:8765/rpc\n' "$host" "$host"
}
start() {
    if exists; then
        check_owner
        case "$(state)" in
            running) ;;
            created|exited) docker start "$container" >/dev/null ;;
            *) fail "Container $container is $(state); inspect it with docker before starting." ;;
        esac
    else
        ensure_image
        local bind_ip api_host
        bind_ip=${SU2_BIND_IP:-$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')}
        [[ -n "$bind_ip" ]] || fail "Set SU2_BIND_IP to the Spark's LAN IPv4 address."
        api_host=${SU2_HOST:-$bind_ip}
        docker run --detach --name "$container" \
            --label "$owner_label=true" \
            --gpus all --init --restart unless-stopped --shm-size=8g \
            --publish "$bind_ip:8765:8765" --env "ASTRA_HOST=$api_host" \
            --mount type=volume,source=su2-spark-work,target=/workspace \
            --mount type=volume,source=su2-spark-mail,target=/var/lib/astra-mail \
            --mount type=volume,source=su2-spark-codex,target=/home/sthornington/.codex \
            --mount type=volume,source=su2-spark-history,target=/commandhistory \
            "$image" >/dev/null
    fi
    wait_ready
    endpoint
}
require_terminal() {
    [[ -t 0 && -t 1 ]] || fail "Run this owner command from an interactive terminal on the Spark."
}

case "$action" in
    build) build "$@" ;;
    start) start ;;
    token|certificate)
        exists || fail "Start the container with ./launch.sh first."
        check_owner
        if [[ "$action" == token ]]; then
            exec docker exec "$container" cat /var/lib/astra-mail/api/token
        else
            exec docker exec "$container" cat /var/lib/astra-mail/api/server.crt
        fi
        ;;
    status)
        if exists; then
            docker container inspect --format '{{.Name}}: {{.State.Status}} ({{if .State.Health}}{{.State.Health.Status}}{{end}})' "$container"
            endpoint
        else
            printf 'Container %s has not been created.\n' "$container"
        fi
        ;;
    stop)
        if exists; then check_owner; docker stop --time 30 "$container"; fi
        ;;
    recreate)
        if exists; then
            check_owner
            case "$(state)" in
                created|exited) ;;
                *) fail "Stop the container with ./launch.sh stop before recreating it." ;;
            esac
        fi
        # Finish any necessary build before removing the old container.
        ensure_image
        if exists; then docker container rm "$container" >/dev/null; fi
        start
        ;;
    login|astra|shell)
        require_terminal
        start
        case "$action" in
            login) exec docker exec -it "$container" codex login --device-auth ;;
            astra)
                exec docker exec -it "$container" tmux new-session -A -s astra \
                    codex -m gpt-6-astra \
                    'Wait for simulation work in your mailbox and coordinate with mac.'
                ;;
            shell) exec docker exec -it "$container" bash ;;
        esac
        ;;
esac

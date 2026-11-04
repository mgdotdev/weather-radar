
IMAGE=weather-radar
CODE_DIR=/home/weather-radar/code
# The image installs everything into this venv; reference it explicitly so we
# don't depend on PATH surviving the shell invocation.
VENV_PY=/home/weather-radar/venv/bin/python

build() {
    docker build --no-cache -t "$IMAGE" -f .docker/Dockerfile.weather-radar .
}

container() {
    docker run --rm -it \
      -p 8000:8000 \
      -v /tmp:/tmp \
      -v "$PWD":"$CODE_DIR" \
      -v /etc/localtime:/etc/localtime:ro \
      -e LOG_LEVEL \
      -e WR_STREAM_TIMING \
      -e WR_RESOLVE_TIMING \
      -e NOAA_MAX_CONCURRENCY \
      "$IMAGE" bash
}

# Run the webserver. On the host this launches a container (publishing :8000
# and mounting the repo so edits are live); inside the container it runs
# uvicorn directly. An optional first arg sets the worker count (default 16).
#   ./run.sh server        # 16 workers
#   ./run.sh server 4      # 4 workers
server() {
    local workers="${1:-16}"
    if command -v docker >/dev/null 2>&1; then
        docker run --rm -it \
          -p 8000:8000 \
          -v /tmp:/tmp \
          -v "$PWD":"$CODE_DIR" \
          -v /etc/localtime:/etc/localtime:ro \
          -e LOG_LEVEL \
          -e WR_STREAM_TIMING \
          -e WR_RESOLVE_TIMING \
          -e NOAA_MAX_CONCURRENCY \
          "$IMAGE" bash -c "cd $CODE_DIR && ./run.sh server $workers"
    else
        local py="$VENV_PY"
        command -v "$py" >/dev/null 2>&1 || py="python"
        PYTHONPATH="$PWD/lib/src:$PWD/server/src${PYTHONPATH:+:$PYTHONPATH}" \
          "$py" -m uvicorn weather_radar.server.__main__:server \
            --factory \
            --workers "$workers" \
            --host 0.0.0.0 \
            --port 8000
    fi
}

# Run the webserver in DEV mode: a single worker that hot-reloads on source
# changes. (uvicorn --reload is incompatible with --workers, so dev mode is
# always single-worker.) The background cache-writer/scheduler are respawned
# on each reload; the startup probe tolerates a lingering manager socket.
dev() {
    if command -v docker >/dev/null 2>&1; then
        docker run --rm -it \
          -p 8000:8000 \
          -v /tmp:/tmp \
          -v "$PWD":"$CODE_DIR" \
          -v /etc/localtime:/etc/localtime:ro \
          -e LOG_LEVEL \
          -e WR_STREAM_TIMING \
          -e WR_RESOLVE_TIMING \
          -e NOAA_MAX_CONCURRENCY \
          "$IMAGE" bash -c "cd $CODE_DIR && ./run.sh dev"
    else
        local py="$VENV_PY"
        command -v "$py" >/dev/null 2>&1 || py="python"
        PYTHONPATH="$PWD/lib/src:$PWD/server/src${PYTHONPATH:+:$PYTHONPATH}" \
          "$py" -m uvicorn weather_radar.server.__main__:server \
            --factory \
            --reload \
            --reload-dir "$PWD/server/src" \
            --reload-dir "$PWD/lib/src" \
            --host 0.0.0.0 \
            --port 8000
    fi
}

# Internal: run pytest against the source tree using the venv interpreter
# (which has requests/scipy/etc. installed). PYTHONPATH points at both `src`
# roots so the `weather_radar` PEP 420 namespace package — split across
# lib/src and server/src — resolves regardless of the editable install.
# Tests are addressed relative to the repo root (e.g. lib/test).
_pytest() {
    local args="$*"
    [ -z "$args" ] && args="lib/test"
    local py="$VENV_PY"
    command -v "$py" >/dev/null 2>&1 || py="python"
    "$py" -m pip install --quiet pytest 2>/dev/null || "$py" -m pip install pytest
    PYTHONPATH="$PWD/lib/src:$PWD/server/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$py" -m pytest --import-mode=importlib $args
}

# Run the test suite. Works both on the host (wraps in `docker run`) and from
# inside the container (runs pytest directly). Extra args pass through to
# pytest, relative to the repo root, e.g.:
#   ./run.sh test                          # everything under lib/test
#   ./run.sh test lib/test/test_area.py    # a single file
#   ./run.sh test lib/test -k cachewriter  # filter
test() {
    if command -v docker >/dev/null 2>&1; then
        # Host: run inside a throwaway container with the repo mounted.
        docker run --rm \
          -v /tmp:/tmp \
          -v "$PWD":"$CODE_DIR" \
          -v /etc/localtime:/etc/localtime:ro \
          "$IMAGE" bash -c "cd $CODE_DIR && ./run.sh _pytest $*"
    else
        # Inside the container already (no docker): run pytest directly.
        _pytest "$@"
    fi
}

"$@"

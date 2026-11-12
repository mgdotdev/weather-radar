
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
      -e WR_WEATHER_SOURCE \
      -e NDFD_SECTOR \
      -e NDFD_TTL_SECONDS \
      -e NDFD_MISS_TTL_SECONDS \
      -e WR_NDFD_UPSAMPLE \
      -e WR_AQI_UPSAMPLE \
      -e WR_NBM_UPSAMPLE \
      -e NBM_REGION \
      -e NBM_TTL_SECONDS \
      -e NAQFC_RECHECK_SECONDS \
      -e WR_AUTH_DIR \
      -e WR_ADMIN_PASSWORD \
      -e WR_ADMIN_PASSWORD_HASH \
      -e WR_ADMIN_SECRET \
      -e WR_ADMIN_SESSION_TTL \
      "$IMAGE" bash
}

# Run the webserver. On the host this launches a container (publishing :8000
# and mounting the repo so edits are live); inside the container it runs
# uvicorn directly. An optional first arg sets the worker count (default 4).
#   ./run.sh server        # 4 workers
#   ./run.sh server 8      # 8 workers
server() {
    local workers="${1:-4}"
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
          -e WR_WEATHER_SOURCE \
          -e NDFD_SECTOR \
          -e NDFD_TTL_SECONDS \
          -e NDFD_MISS_TTL_SECONDS \
          -e WR_NDFD_UPSAMPLE \
          -e WR_AQI_UPSAMPLE \
          -e WR_NBM_UPSAMPLE \
          -e NBM_REGION \
          -e NBM_TTL_SECONDS \
          -e NAQFC_RECHECK_SECONDS \
          -e WR_AUTH_DIR \
          -e WR_ADMIN_PASSWORD \
          -e WR_ADMIN_PASSWORD_HASH \
          -e WR_ADMIN_SECRET \
          -e WR_ADMIN_SESSION_TTL \
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
          -e WR_WEATHER_SOURCE \
          -e NDFD_SECTOR \
          -e NDFD_TTL_SECONDS \
          -e NDFD_MISS_TTL_SECONDS \
          -e WR_NDFD_UPSAMPLE \
          -e WR_AQI_UPSAMPLE \
          -e WR_NBM_UPSAMPLE \
          -e NBM_REGION \
          -e NBM_TTL_SECONDS \
          -e NAQFC_RECHECK_SECONDS \
          -e WR_AUTH_DIR \
          -e WR_ADMIN_PASSWORD \
          -e WR_ADMIN_PASSWORD_HASH \
          -e WR_ADMIN_SECRET \
          -e WR_ADMIN_SESSION_TTL \
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

# Generate a WR_ADMIN_PASSWORD_HASH for the admin page. Prompts (hidden) for a
# password and prints the 'scrypt$salt$hash' string to set as the env var, so
# the plaintext password never needs to live in your compose/env file.
#   ./run.sh hash_password
hash_password() {
    local py="$VENV_PY"
    command -v "$py" >/dev/null 2>&1 || py="python"
    PYTHONPATH="$PWD/lib/src${PYTHONPATH:+:$PYTHONPATH}" "$py" - <<'PY'
import getpass
from weather_radar.lib import auth
p1 = getpass.getpass("Admin password: ")
p2 = getpass.getpass("Confirm: ")
if p1 != p2:
    raise SystemExit("passwords do not match")
if not p1:
    raise SystemExit("empty password")
h = auth.hash_password(p1)
print("\nSet this in your environment (the value uses '.' delimiters, so it is")
print("safe to paste unquoted in a shell or docker-compose file):\n")
print("WR_ADMIN_PASSWORD_HASH=" + h)
PY
}


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

# Install/refresh Python dependencies into the LIVE venv WITHOUT rebuilding the
# image. Intended to be run INSIDE the running container, so the installed
# packages persist in that container's venv (a throwaway `docker run` would
# discard them). Reinstalls lib + server requirements (picks up newly-added
# deps like eccodes/numpy for the AQI/GRIB2 layer).
#
# From the host, run it against your already-running dev/server container:
#   docker exec -it <container> bash -c "cd $CODE_DIR && ./run.sh deps"
# or open a shell in the container (./run.sh container) and run `./run.sh deps`.
# The eccodes wheel bundles libeccodes, so no apt/root step is needed for a
# live install; the Dockerfile still installs the system lib for clean builds.
deps() {
    local py="$VENV_PY"
    command -v "$py" >/dev/null 2>&1 || py="python"
    "$py" -m pip install --upgrade pip
    "$py" -m pip install -r ./lib/requirements.txt
    "$py" -m pip install -r ./server/requirements.txt
    # Re-run the editable installs in case entry points / metadata changed.
    "$py" -m pip install -e ./lib --no-cache-dir
    "$py" -m pip install -e ./server --no-cache-dir
    echo "deps installed into $py"
    "$py" -c "import eccodes, scipy, numpy; print('eccodes', eccodes.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__)"
}

"$@"


IMAGE=weather-radar
# Default GitHub Container Registry repo for published images. Override with
# the IMAGE_REGISTRY env var (e.g. your own fork's ghcr.io/<user>/<repo>).
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/mgdotdev/weather-radar}"
# Tracks the last-published semver (MAJOR.MINOR.PATCH). `publish` with no
# explicit tag bumps the MINOR here and uses the result as the image tag.
VERSION_FILE="VERSION"
CODE_DIR=/home/weather-radar/code
# The image installs everything into this venv; reference it explicitly so we
# don't depend on PATH surviving the shell invocation.
VENV_PY=/home/weather-radar/venv/bin/python

build() {
    docker build --no-cache -t "$IMAGE" -f .docker/Dockerfile.weather-radar .
}

# Build and push the container image to the GitHub Container Registry (GHCR).
# Tags the local `$IMAGE` build as `$IMAGE_REGISTRY:<tag>` and pushes it, so a
# host that installs from a registry tag (see docker-compose `image:`) gets the
# new image.
#
# With NO argument, the MINOR version is auto-incremented: the current semver
# is read from the VERSION file, its minor is bumped (patch reset to 0), the
# new value is written back, and the image is tagged `v<new>` AND `latest`:
#   ./run.sh publish            # VERSION 0.3.2 -> 0.4.0; tags v0.4.0 + latest
#
# An explicit first arg overrides the auto-bump and is used verbatim as the
# tag; if it looks like a vX.Y.Z / X.Y.Z semver, VERSION is updated to match:
#   ./run.sh publish v1.2.0     # -> ghcr.io/.../weather-radar:v1.2.0
#   ./run.sh publish edge       # -> ghcr.io/.../weather-radar:edge (no bump)
#   IMAGE_REGISTRY=ghcr.io/me/wr ./run.sh publish   # push to your own repo
#
# Authentication: GHCR needs a login with a token that has `write:packages`.
# If GHCR_TOKEN (or CR_PAT/GH_TOKEN) is set — with GHCR_USER (or GITHUB_ACTOR)
# for the username — this logs in for you; otherwise it assumes you've already
# run `docker login ghcr.io` and just builds/pushes.
publish() {
    command -v docker >/dev/null 2>&1 || {
        echo "docker not found on PATH; publish must run on a host with docker" >&2
        return 1
    }

    # Decide the tag: explicit arg wins; otherwise auto-bump the minor version.
    local tag bumped=""
    if [ -n "${1:-}" ]; then
        tag="$1"
        # If the explicit tag is a semver (optionally v-prefixed), sync VERSION.
        local semver="${tag#v}"
        if printf '%s' "$semver" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
            bumped="$semver"
        fi
    else
        local cur next
        cur="$(_read_version)" || return 1
        next="$(_next_minor "$cur")" || return 1
        bumped="$next"
        tag="v$next"
        echo "auto-bumping version: $cur -> $next"
    fi

    local ref="${IMAGE_REGISTRY}:${tag}"

    # Optional non-interactive login when a token is provided in the env.
    local token="${GHCR_TOKEN:-${CR_PAT:-${GH_TOKEN:-}}}"
    if [ -n "$token" ]; then
        local user="${GHCR_USER:-${GITHUB_ACTOR:-}}"
        if [ -z "$user" ]; then
            echo "a token was provided but no username; set GHCR_USER (or GITHUB_ACTOR)" >&2
            return 1
        fi
        echo "logging in to ghcr.io as $user"
        echo "$token" | docker login ghcr.io -u "$user" --password-stdin || return 1
    fi

    echo "building $IMAGE"
    build || return 1

    echo "tagging $IMAGE -> $ref"
    docker tag "$IMAGE" "$ref" || return 1

    echo "pushing $ref"
    docker push "$ref" || return 1

    # For a semver release, ALSO move the `latest` tag to this build so the
    # default compose `image:` (…:latest) picks it up.
    if [ -n "$bumped" ] && [ "$tag" != "latest" ]; then
        local latest_ref="${IMAGE_REGISTRY}:latest"
        echo "tagging + pushing $latest_ref"
        docker tag "$IMAGE" "$latest_ref" || return 1
        docker push "$latest_ref" || return 1
    fi

    # Persist the new version only after a successful push, so a failed
    # publish doesn't advance (and skip) a version number.
    if [ -n "$bumped" ]; then
        printf '%s\n' "$bumped" > "$VERSION_FILE"
        echo "VERSION -> $bumped"
    fi

    echo "published $ref"
}

# Read the current semver from the VERSION file, defaulting to 0.0.0 if the
# file is missing/empty. Validates the MAJOR.MINOR.PATCH shape.
_read_version() {
    local v="0.0.0"
    if [ -f "$VERSION_FILE" ]; then
        v="$(tr -d ' \t\r\n' < "$VERSION_FILE")"
        [ -n "$v" ] || v="0.0.0"
    fi
    if ! printf '%s' "$v" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        echo "VERSION file does not contain a valid MAJOR.MINOR.PATCH: '$v'" >&2
        return 1
    fi
    printf '%s' "$v"
}

# Given MAJOR.MINOR.PATCH, print MAJOR.(MINOR+1).0.
_next_minor() {
    local major minor
    major="${1%%.*}"
    minor="${1#*.}"; minor="${minor%%.*}"
    printf '%s.%s.0' "$major" "$((minor + 1))"
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
      -e NOAA_USER_AGENT \
      -e WR_APP_NAME \
      -e WR_CONTACT_EMAIL \
      -e GEOCODE_USER_AGENT \
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
          -e NOAA_USER_AGENT \
          -e WR_APP_NAME \
          -e WR_CONTACT_EMAIL \
          -e GEOCODE_USER_AGENT \
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
          -e NOAA_USER_AGENT \
          -e WR_APP_NAME \
          -e WR_CONTACT_EMAIL \
          -e GEOCODE_USER_AGENT \
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

# Print the list of public subcommands (one per line): every top-level
# function defined in THIS script except the private `_`-prefixed helpers.
# Parsed from the file itself so it stays in sync as commands are added — used
# by the tab-completion below (and handy on its own).
_commands() {
    grep -oE '^[a-zA-Z][a-zA-Z0-9_]*\(\)' "$0" | sed 's/()$//'
}

# Emit a bash completion script for run.sh. Source it to get <TAB> completion
# of the subcommand names:
#   source <(./run.sh completion)          # current shell
#   ./run.sh completion > ~/.run.sh-completion.bash   # then source from ~/.bashrc
# The completion asks the script for its own command list (`_commands`), so new
# subcommands are picked up automatically with no edits here.
completion() {
    cat <<'BASH'
# bash completion for run.sh (weather-radar). Source this file.
_run_sh_complete() {
    local cur prev words cword
    cur="${COMP_WORDS[COMP_CWORD]}"
    # Only complete the FIRST argument (the subcommand); let args complete as
    # filenames (e.g. `./run.sh test lib/test/...`).
    if [ "$COMP_CWORD" -ne 1 ]; then
        COMPREPLY=()
        return 0
    fi
    local script="${COMP_WORDS[0]}"
    local cmds
    cmds="$("$script" _commands 2>/dev/null)"
    COMPREPLY=( $(compgen -W "$cmds" -- "$cur") )
}
# Register for the common invocation forms (./run.sh, run.sh, bash run.sh).
complete -o default -F _run_sh_complete run.sh ./run.sh
BASH
}

"$@"

# Weather Radar

This repo is an attempt to build a self-hosted weather forcasting api and map
app.

NOAA forecasting models are free to the public. In an attempt to democratize
the public's access to that data, *their* data, this repository wraps the NOAA
api's behind a local api for more rapid consumption, with the necessary caching
and interpolation methods for functionalizing data across time and space.

This is still a work in progress! MVP will be a precipitation model, but expect
the supported models to expand.

## Running with Docker Compose

The repo ships a `docker-compose.yml` and `.env.example`. The image self-starts
the server (uvicorn `--factory`, `WR_WORKERS` workers) and listens on port 8000.
All state — the NOAA cache, downloaded GRIB files, the server log, and the
minted access tokens — lives under `/data`, so mount a persistent volume there
to keep tokens and the warm cache across restarts.

### 1. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

- `WR_CONTACT_EMAIL` — your email. api.weather.gov requires a User-Agent and
  asks that it include a contact; it's sent as `weather-radar (your@email)`.
- `WR_ADMIN_PASSWORD_HASH` and `WR_ADMIN_SECRET` — see below.

### 2. Generate admin credentials

The app is always access-token gated, and the first token can only be minted
from the password-protected `/admin` page. Generate a password hash and a
stable session secret:

```bash
./run.sh hash_password                                      # -> WR_ADMIN_PASSWORD_HASH=scrypt$...
python -c "import secrets; print(secrets.token_hex(32))"    # -> WR_ADMIN_SECRET
```

### 3. Start

```bash
docker compose up -d
```

Then open `/admin`, sign in with your admin password, and mint an access token.

### Notes

- **Persistence:** back `/data` with a host path or named volume. The image
  runs as the unprivileged `weather-radar` user (UID 1000), so a bind-mounted
  host path must be writable by UID 1000.
- **Access control:** `/health` is unauthenticated (for liveness checks);
  everything under `/api/` and the map UI require the access token, supplied as
  `?token=...` or the `X-Access-Token` header.
- **Reverse proxy:** the app speaks plain HTTP on 8000 with no TLS of its own.
  Put it behind a reverse proxy to terminate HTTPS. No WebSocket support is
  needed (streams use chunked HTTP responses).

### Building and publishing the image

`docker compose up` builds the image locally (from
`.docker/Dockerfile.weather-radar`) when it isn't already present, so a plain
checkout runs with no extra steps.

To publish a pre-built image to a registry — e.g. for a host that installs from
a registry tag rather than building — build and push the `image:` tag:

```bash
docker compose build
docker push ghcr.io/mgdotdev/weather-radar:latest
```

Change the `image:` field in `docker-compose.yml` to your own registry/tag
first if you're not pushing to the default.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
| --- | --- |
| `WR_CONTACT_EMAIL` | Contact email sent in the NOAA / Nominatim User-Agent. |
| `WR_ADMIN_PASSWORD_HASH` | scrypt hash gating the `/admin` token-management page. |
| `WR_ADMIN_SECRET` | Stable key signing admin sessions (survives restarts). |
| `WR_WEATHER_SOURCE` | `gridpoints` (default), `ndfd`, or `nbm`. |
| `WR_WORKERS` | uvicorn worker count. |
| `CACHE_DIR` / `WR_AUTH_DIR` / `LOG_DIR` | State locations (default `/data`). |


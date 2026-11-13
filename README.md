# Weather Radar

This repo is an attempt to build a self-hosted weather forcasting api and map
app.

NOAA forecasting models are free to the public. In an attempt to democratize
the public's access to that data, *their* data, this repository wraps the NOAA
api's behind a local api for more rapid consumption, with the necessary caching
and interpolation methods for functionalizing data across time and space.

This is still a work in progress! MVP will be a precipitation model, but expect
the supported models to expand.

## Deploying with Docker Compose (TrueNAS "Install via YAML")

The repo ships a `docker-compose.yml` and `.env.example` for a container
deployment. TrueNAS SCALE's "Custom App → Install via YAML" flow runs a
**pre-built image** (it does not build from the Dockerfile), so the steps are:
build the image, push it to a registry, then install the compose YAML pointing
at that tag.

### 1. Build and push the image

```bash
# From the repo root. Tag for your registry (default compose target is GHCR).
docker build -t ghcr.io/mgdotdev/weather-radar:latest -f .docker/Dockerfile.weather-radar .
echo "$GHCR_TOKEN" | docker login ghcr.io -u mgdotdev --password-stdin
docker push ghcr.io/mgdotdev/weather-radar:latest
```

The image self-starts the server (uvicorn `--factory`, `WR_WORKERS` workers)
and exposes port 8000. All state (NOAA cache, GRIB files, minted access
tokens, log) lives under `/data` — mount a persistent dataset there.

### 2. Generate the admin credentials

The app is always access-token gated, and the first token can only be minted
from the password-protected `/admin` page. Generate a password hash and a
session secret:

```bash
./run.sh hash_password                       # -> WR_ADMIN_PASSWORD_HASH=scrypt$...
python -c "import secrets; print(secrets.token_hex(32))"   # -> WR_ADMIN_SECRET
```

### 3. Install on TrueNAS

Paste the contents of `docker-compose.yml` into **Apps → Custom App → Install
via YAML**, then:

- Set `image:` to the tag you pushed.
- Fill `WR_ADMIN_PASSWORD_HASH` and `WR_ADMIN_SECRET` in the `environment:`
  block (TrueNAS may not read a `.env` file).
- Point the `/data` volume at a dataset writable by **UID 1000** (the image's
  `weather-radar` user), e.g. replace the named volume with
  `- /mnt/pool/apps/weather-radar:/data`.
- Adjust the published host port (`30050:8000` by default) to match your
  reverse-proxy config.

Then wire it into the homelab the usual way (Caddy on the VPS, a Nginx Proxy
Manager proxy host, and an AdGuard DNS rewrite for `weather.vancleave.io`).

### Run it anywhere else

```bash
cp .env.example .env    # fill in the admin hash/secret
docker compose up -d
```

`/health` is unauthenticated for liveness checks; everything under `/api/` and
the map UI require the access token (`?token=` or the `X-Access-Token` header).


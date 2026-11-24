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

## Embedding in Home Assistant

Set the map up in the web UI, then give Home Assistant the page URL (with an
access token appended: mint one on the admin tokens page — use a separate,
labelled token for the dashboard so you can revoke it on its own).

To see how it will fit **before** touching your dashboard, open
`/ha-preview/` on your server (it needs the same access token as every page):
your real app in an iframe of exactly the size you pick (Home Assistant's default
card height is half its width; sections cards are 320–500 px wide; a panel fills
the page), with a control for every URL parameter, a pin editor, "paste an
existing URL", and the URL plus a ready-to-paste card snippet. With **follow**
on, anything you change inside the preview (pins, zoom, legend position) flows
back into the controls.

`/graph-preview/` does the same for the forecast **graph** page: add or change
curves, set the point and time step, switch the compact Home Assistant layout
(`ha=true`) and dark mode, and set the title — click, drag or resize the title
inside the preview and the controls follow. The two previews link to each other.

Both previews render from built-in **sample data** by default ("Sample data" in
the top bar; the frame's requests carry `fixture=1`), so the page loads in about
a second without the forecast pipeline. Untick it to see your real data. Purely
visual settings (pin and legend size, lock, cursor box, area box, floating scrub
bar, credits) and the view (center and zoom) are applied inside the map in place,
without reloading at all. The
copied URL and card never include `fixture`. Sample data still requires an
access token, and `WR_PREVIEW_FIXTURES=0` turns it off.

The legend and pin boxes are sized for a full browser window (the legend is
360 px wide). In a small card, add `legendSize=0.5&pinSize=0.5&locked=1` to the
URL (the preview has a checkbox for it) so they take a fraction of the card.

Add `floatScrubber=1` to move the play button, scrub bar and time label out of the
legend into a bar floating at the bottom-center of the map (also a Settings
checkbox, and a control on `/ha-preview/`). Place the legend away from the bottom
edge so the two don't overlap.

`mapLat` / `mapLon` set where the map is **looking** (the view center). They are
separate from `lat` / `lon`, the center of the data area, which only changes when
you capture or recenter the area — so panning never changes which data is
requested. The two params follow the map as you drag (written while you move and
once more when it settles), are removed while the view is on the area's center,
survive a reload and a model switch, and invalid values are ignored. On
`/ha-preview/` they have their own fields: typing a center moves the map in
place, and panning inside the preview fills them in.

The map's credit button (the basemap's required © OpenStreetMap / CARTO notice) is
hidden by default and the distance scale sits in the top-right corner. Tick
"Show map credits" in Settings, or add `attribution=1`, to show the button.
Untick "Show scale bar" in Settings, or add `scaleBar=0`, to hide the distance scale.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
| --- | --- |
| `WR_CONTACT_EMAIL` | Contact email sent in the NOAA / Nominatim User-Agent. |
| `WR_ADMIN_PASSWORD_HASH` | scrypt hash gating the `/admin` token-management page. |
| `WR_ADMIN_SECRET` | Stable key signing admin sessions (survives restarts). |
| `WR_WEATHER_SOURCE` | `nbm` (default), `ndfd`, or `gridpoints`. |
| `WR_WORKERS` | uvicorn worker count. |
| `CACHE_DIR` / `WR_AUTH_DIR` / `LOG_DIR` | State locations (default `/data`). |


"""Credentialed access control for the app.

Two independent mechanisms:

  * ACCESS TOKENS — long random secrets that grant access to the whole app.
    Designed for embedding the app as a cross-origin iframe (e.g. a Home
    Assistant dashboard on iOS): the token rides in the URL (`?token=`) because
    WebKit's ITP blocks third-party iframe cookies, so auth can't rely on a
    persisted cookie. MULTIPLE tokens are valid at once, enabling overlap-free
    rotation (add the new token, repoint the dashboard, then revoke the old).
    Tokens are persisted to the filesystem so they survive restarts.

  * ADMIN PASSWORD — gates the token-management admin page. Verified against a
    hashed password (WR_ADMIN_PASSWORD_HASH, preferred) or a plaintext one
    (WR_ADMIN_PASSWORD, convenience fallback). A correct password mints a
    short-lived, HMAC-signed admin SESSION token (stateless — no server store,
    survives across workers) delivered as a same-origin cookie.

Dev defaults (nothing configured): no access tokens AND no admin password ->
the app is OPEN and the admin page is localhost-only, matching the existing
hot-cache admin's safe-by-default behavior. Set a token/password to lock down.

Stdlib only (hashlib/hmac/secrets); no new dependencies.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

# --- Paths / config ---------------------------------------------------------

# Token store dir: WR_AUTH_DIR, else CACHE_DIR, else /tmp. Mount a persistent
# volume here so tokens survive a host reboot / tmp cleanup.
_AUTH_DIR = (os.environ.get("WR_AUTH_DIR")
             or os.environ.get("CACHE_DIR")
             or "/tmp")
_TOKENS_PATH = os.path.join(_AUTH_DIR, "weather-radar-tokens.json")

# Admin password config (hash preferred).
_ADMIN_PASSWORD_HASH = os.environ.get("WR_ADMIN_PASSWORD_HASH")
_ADMIN_PASSWORD = os.environ.get("WR_ADMIN_PASSWORD")

# Admin session: HMAC-signed so any worker can verify without shared state.
# The signing key is WR_ADMIN_SECRET if set (stable across restarts -> sessions
# survive a restart); otherwise a per-process random key (sessions drop on
# restart, which is acceptable — the operator just re-logs-in).
_ADMIN_SECRET = (os.environ.get("WR_ADMIN_SECRET")
                 or secrets.token_hex(32)).encode("utf-8")
_ADMIN_SESSION_TTL = int(os.environ.get("WR_ADMIN_SESSION_TTL", str(12 * 3600)))

_LOCALHOSTS = {"127.0.0.1", "::1", "localhost"}

# Small mtime-keyed cache of the parsed token file, so the per-request auth
# check (every API/stream request) doesn't re-read+parse JSON from disk each
# time. Invalidated automatically when the file's mtime changes (e.g. the admin
# page generates/revokes a token).
_tokens_cache = {"mtime": None, "tokens": []}


# --- Token store ------------------------------------------------------------

def _load_tokens():
    """Load the token list from disk (mtime-cached). Each entry: {token, label,
    created, expires|null, id}. Returns [] on any error (treated as no tokens)."""
    try:
        mtime = os.path.getmtime(_TOKENS_PATH)
    except OSError:
        _tokens_cache["mtime"] = None
        _tokens_cache["tokens"] = []
        return []
    if _tokens_cache["mtime"] == mtime:
        return _tokens_cache["tokens"]
    try:
        with open(_TOKENS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("tokens", [])
        tokens = data if isinstance(data, list) else []
    except Exception:
        logger.warning("failed reading token store %s", _TOKENS_PATH, exc_info=True)
        tokens = []
    _tokens_cache["mtime"] = mtime
    _tokens_cache["tokens"] = tokens
    return tokens


def _save_tokens(tokens):
    """Persist the token list atomically with restrictive (0600) perms."""
    os.makedirs(_AUTH_DIR, exist_ok=True)
    tmp = _TOKENS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"tokens": tokens}, f)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, _TOKENS_PATH)
    # Invalidate the mtime cache so this process sees the write immediately
    # (coarse filesystem mtime granularity could otherwise mask a same-second
    # change).
    _tokens_cache["mtime"] = None


def _not_expired(entry, now=None):
    exp = entry.get("expires")
    if not exp:
        return True
    return (now or time.time()) < exp


def access_tokens_configured():
    """True if at least one (non-expired) access token exists -> the app is
    locked down. No tokens -> open (dev)."""
    now = time.time()
    return any(_not_expired(e, now) for e in _load_tokens())


def verify_access_token(supplied):
    """Constant-time check that `supplied` matches a stored, non-expired token.
    Returns True/False. An empty/None supplied is always False."""
    if not supplied:
        return False
    now = time.time()
    ok = False
    # Compare against EVERY candidate (don't short-circuit) to keep timing
    # roughly uniform; compare_digest is itself constant-time per comparison.
    for e in _load_tokens():
        tok = e.get("token", "")
        if tok and _not_expired(e, now) and hmac.compare_digest(str(tok), str(supplied)):
            ok = True
    return ok


def list_tokens():
    """Return token metadata WITHOUT the secret values (for the admin UI):
    [{label, created, expires, preview}] where preview is the last 4 chars."""
    out = []
    now = time.time()
    for e in _load_tokens():
        tok = e.get("token", "")
        out.append({
            "label": e.get("label", ""),
            "created": e.get("created"),
            "expires": e.get("expires"),
            "expired": not _not_expired(e, now),
            "preview": ("…" + tok[-4:]) if len(tok) >= 4 else "",
            "id": e.get("id") or (tok[-8:] if tok else ""),
        })
    return out


def generate_token(label="", ttl_seconds=None):
    """Create, persist, and RETURN a new access token (the full secret — shown
    once by the admin UI). `ttl_seconds` optionally sets an expiry."""
    token = secrets.token_urlsafe(32)
    entry = {
        "token": token,
        "id": secrets.token_hex(4),
        "label": label or "",
        "created": time.time(),
        "expires": (time.time() + float(ttl_seconds)) if ttl_seconds else None,
    }
    tokens = _load_tokens()
    tokens.append(entry)
    _save_tokens(tokens)
    return {"token": token, "id": entry["id"], "label": entry["label"],
            "created": entry["created"], "expires": entry["expires"]}


def newest_token():
    """Return the SECRET of the most-recently-created non-expired token, or
    None if the store is empty. Used by the password login path to hand back an
    existing token (login never mints/rotates)."""
    now = time.time()
    valid = [e for e in _load_tokens() if _not_expired(e, now) and e.get("token")]
    if not valid:
        return None
    valid.sort(key=lambda e: e.get("created") or 0, reverse=True)
    return valid[0]["token"]


def revoke_token(token_id):
    """Remove the token whose `id` (or trailing-8 fallback) matches. Returns
    True if something was removed."""
    tokens = _load_tokens()
    kept = []
    removed = False
    for e in tokens:
        ident = e.get("id") or (e.get("token", "")[-8:])
        if ident == token_id:
            removed = True
            continue
        kept.append(e)
    if removed:
        _save_tokens(kept)
    return removed


# --- Admin password + session ----------------------------------------------

def admin_password_configured():
    """True if an admin password (hash or plaintext) is set. If not, the admin
    page falls back to localhost-only."""
    return bool(_ADMIN_PASSWORD_HASH or _ADMIN_PASSWORD)


def _scrypt_hash(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=16384, r=8, p=1, dklen=32)


def hash_password(password):
    """Produce a 'scrypt.<salt_hex>.<hash_hex>' string for WR_ADMIN_PASSWORD_HASH.

    NOTE: the delimiter is '.' (NOT '$') on purpose — a '$' in the value gets
    mangled by shells and docker-compose env interpolation (they treat $NAME as
    a variable), silently corrupting the hash. '.' is safe to paste unquoted.
    """
    salt = secrets.token_bytes(16)
    digest = _scrypt_hash(password, salt)
    return "scrypt.%s.%s" % (salt.hex(), digest.hex())


def verify_admin_password(password):
    """Constant-time verify a submitted admin password against the configured
    hash (preferred) or plaintext fallback. Accepts hashes delimited by '.'
    (current) or '$' (legacy) so older values keep working."""
    if not password:
        return False
    if _ADMIN_PASSWORD_HASH:
        raw = _ADMIN_PASSWORD_HASH.strip()
        # Split on '.' if present, else legacy '$'.
        delim = "." if raw.count(".") >= 2 else "$"
        try:
            scheme, salt_hex, hash_hex = raw.split(delim, 2)
        except ValueError:
            logger.warning("WR_ADMIN_PASSWORD_HASH is malformed")
            return False
        if scheme != "scrypt":
            logger.warning("WR_ADMIN_PASSWORD_HASH: unsupported scheme %r", scheme)
            return False
        try:
            expected = bytes.fromhex(hash_hex)
            salt = bytes.fromhex(salt_hex)
        except ValueError:
            logger.warning("WR_ADMIN_PASSWORD_HASH: bad hex (was it mangled by "
                           "shell/compose $-expansion?)")
            return False
        got = _scrypt_hash(password, salt)
        return hmac.compare_digest(got, expected)
    if _ADMIN_PASSWORD:
        return hmac.compare_digest(password, _ADMIN_PASSWORD)
    return False


def mint_admin_session():
    """Return a signed admin session token: '<expiry>.<hexsig>'. Stateless —
    verified by any worker via the shared/derived secret."""
    exp = int(time.time()) + _ADMIN_SESSION_TTL
    msg = str(exp).encode("utf-8")
    sig = hmac.new(_ADMIN_SECRET, msg, hashlib.sha256).hexdigest()
    return "%d.%s" % (exp, sig)


def verify_admin_session(cookie_val):
    """True if `cookie_val` is a valid, unexpired admin session token."""
    if not cookie_val or "." not in cookie_val:
        return False
    exp_str, sig = cookie_val.split(".", 1)
    try:
        exp = int(exp_str)
    except ValueError:
        return False
    if time.time() >= exp:
        return False
    expected = hmac.new(_ADMIN_SECRET, exp_str.encode("utf-8"),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def is_localhost(client_host):
    return client_host in _LOCALHOSTS

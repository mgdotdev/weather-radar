"""Tests for credentialed access control (weather_radar.lib.auth).

auth.py reads its config from env vars at import time, so each test loads a
FRESH module instance via importlib with the desired env set + an isolated
WR_AUTH_DIR (so the on-disk token store doesn't leak between tests).
"""

import importlib.util
import os

import pytest


def _load_auth(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("WR_AUTH_DIR", str(tmp_path))
    # Stable signing secret so session tests are deterministic.
    monkeypatch.setenv("WR_ADMIN_SECRET", "test-secret")
    for k in ("WR_ADMIN_PASSWORD", "WR_ADMIN_PASSWORD_HASH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(
        "wrauth_test",
        os.path.join(os.path.dirname(__file__), "..", "src",
                     "weather_radar", "lib", "auth.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAccessTokens:
    def test_open_when_no_tokens(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        assert a.access_tokens_configured() is False
        assert a.verify_access_token("anything") is False

    def test_generate_and_verify(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        t = a.generate_token("HA")
        assert len(t["token"]) >= 32
        assert a.access_tokens_configured() is True
        assert a.verify_access_token(t["token"]) is True
        assert a.verify_access_token("wrong") is False
        assert a.verify_access_token("") is False
        assert a.verify_access_token(None) is False

    def test_persists_across_reload(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        t = a.generate_token("keep")
        # Fresh module instance, same WR_AUTH_DIR -> token still present.
        b = _load_auth(monkeypatch, tmp_path)
        assert b.verify_access_token(t["token"]) is True

    def test_list_hides_secret(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        a.generate_token("one")
        a.generate_token("two")
        lst = a.list_tokens()
        assert len(lst) == 2
        assert all("token" not in row for row in lst)
        assert all(row["preview"].startswith("…") for row in lst)

    def test_revoke(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        t1 = a.generate_token("a")
        t2 = a.generate_token("b")
        assert a.revoke_token(t1["id"]) is True
        assert a.verify_access_token(t1["token"]) is False
        assert a.verify_access_token(t2["token"]) is True
        assert a.revoke_token("missing") is False

    def test_expired_token_rejected(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        t = a.generate_token("short", ttl_seconds=-1)   # already expired
        assert a.verify_access_token(t["token"]) is False
        assert a.access_tokens_configured() is False     # expired doesn't count

    def test_newest_token(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        assert a.newest_token() is None
        t1 = a.generate_token("first")
        t2 = a.generate_token("second")
        # Most-recently-created wins.
        assert a.newest_token() == t2["token"]
        a.revoke_token(t2["id"])
        assert a.newest_token() == t1["token"]

    def test_newest_token_skips_expired(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        a.generate_token("expired", ttl_seconds=-1)
        live = a.generate_token("live")
        assert a.newest_token() == live["token"]


class TestAdminPassword:
    def test_not_configured(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        assert a.admin_password_configured() is False
        assert a.verify_admin_password("x") is False

    def test_plaintext(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path, WR_ADMIN_PASSWORD="hunter2")
        assert a.admin_password_configured() is True
        assert a.verify_admin_password("hunter2") is True
        assert a.verify_admin_password("nope") is False
        assert a.verify_admin_password("") is False

    def test_hashed(self, monkeypatch, tmp_path):
        # Hash with one instance, verify with another configured to use it.
        a0 = _load_auth(monkeypatch, tmp_path)
        h = a0.hash_password("s3cret")
        # '.'-delimited (shell/compose-safe) — no '$' that would get mangled.
        assert h.startswith("scrypt.")
        assert "$" not in h
        a = _load_auth(monkeypatch, tmp_path, WR_ADMIN_PASSWORD_HASH=h)
        assert a.verify_admin_password("s3cret") is True
        assert a.verify_admin_password("wrong") is False

    def test_malformed_hash_rejects(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path, WR_ADMIN_PASSWORD_HASH="garbage")
        assert a.verify_admin_password("anything") is False

    def test_legacy_dollar_hash_accepted(self, monkeypatch, tmp_path):
        # A hash produced in the old 'scrypt$salt$hash' form still verifies.
        a0 = _load_auth(monkeypatch, tmp_path)
        h = a0.hash_password("s3cret")            # new '.' form
        _, salt, digest = h.split(".", 2)
        legacy = "scrypt$%s$%s" % (salt, digest)
        a = _load_auth(monkeypatch, tmp_path, WR_ADMIN_PASSWORD_HASH=legacy)
        assert a.verify_admin_password("s3cret") is True
        assert a.verify_admin_password("wrong") is False


class TestAdminSession:
    def test_mint_and_verify(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        s = a.mint_admin_session()
        assert a.verify_admin_session(s) is True

    def test_tampered_session_rejected(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        s = a.mint_admin_session()
        exp, sig = s.split(".", 1)
        assert a.verify_admin_session(exp + ".deadbeef") is False
        assert a.verify_admin_session("garbage") is False
        assert a.verify_admin_session("") is False

    def test_expired_session_rejected(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path, WR_ADMIN_SESSION_TTL="-1")
        s = a.mint_admin_session()
        assert a.verify_admin_session(s) is False


class TestLocalhost:
    def test_localhost_detection(self, monkeypatch, tmp_path):
        a = _load_auth(monkeypatch, tmp_path)
        assert a.is_localhost("127.0.0.1") is True
        assert a.is_localhost("::1") is True
        assert a.is_localhost("10.0.0.5") is False
        assert a.is_localhost(None) is False

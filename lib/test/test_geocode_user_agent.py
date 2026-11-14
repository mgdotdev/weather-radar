"""Tests for the geocode (Nominatim) User-Agent builder.

Nominatim's usage policy requires a valid, contactable User-Agent. Like the
NOAA UA, it's built from env (sharing WR_APP_NAME / WR_CONTACT_EMAIL) so a
self-hosted deployment supplies its own contact.

FastAPI is only present in the server container, so SKIPPED on a host without
it (run via `./run.sh test`).
"""
import logging

import pytest

pytest.importorskip("fastapi")
import weather_radar.server.api.geocode as geocode  # noqa: E402


def _ua(monkeypatch, **env):
    for k in ("GEOCODE_USER_AGENT", "WR_APP_NAME", "WR_CONTACT_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    geocode._geocode_ua_warned = False
    return geocode._user_agent()


class TestGeocodeUserAgent:
    def test_full_override_used_verbatim(self, monkeypatch):
        assert _ua(monkeypatch, GEOCODE_USER_AGENT="myapp/2.0 (c@x)",
                   WR_CONTACT_EMAIL="ignored@x") == "myapp/2.0 (c@x)"

    def test_app_and_email(self, monkeypatch):
        assert _ua(monkeypatch, WR_CONTACT_EMAIL="you@example.com") \
            == "weather-radar (self-hosted); you@example.com"

    def test_custom_app_and_email(self, monkeypatch):
        assert _ua(monkeypatch, WR_APP_NAME="wx", WR_CONTACT_EMAIL="a@b.com") \
            == "wx (self-hosted); a@b.com"

    def test_bare_default_when_nothing_set(self, monkeypatch):
        assert _ua(monkeypatch) == "weather-radar (self-hosted)"

    def test_blank_override_falls_through(self, monkeypatch):
        assert _ua(monkeypatch, GEOCODE_USER_AGENT="  ",
                   WR_CONTACT_EMAIL="a@b.com") \
            == "weather-radar (self-hosted); a@b.com"

    def test_warns_once_when_no_contact(self, monkeypatch, caplog):
        for k in ("GEOCODE_USER_AGENT", "WR_APP_NAME", "WR_CONTACT_EMAIL"):
            monkeypatch.delenv(k, raising=False)
        geocode._geocode_ua_warned = False
        with caplog.at_level(logging.WARNING, logger=geocode.logger.name):
            geocode._user_agent()
            geocode._user_agent()  # second call must NOT re-warn
        warns = [r for r in caplog.records if "geocode contact" in r.getMessage()]
        assert len(warns) == 1

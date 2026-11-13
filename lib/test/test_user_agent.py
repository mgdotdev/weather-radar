"""Tests for the NOAA User-Agent builder (connection.noaa_user_agent).

api.weather.gov blocks requests without a User-Agent and asks that it identify
the application and include a contact. The UA is built from env so a
self-hosted deployment sets its own contact; these verify the precedence.
"""
import importlib

import weather_radar.lib.connection as connection


def _ua(monkeypatch, **env):
    for k in ("NOAA_USER_AGENT", "WR_APP_NAME", "WR_CONTACT_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Reset the one-time warning latch so each case is independent.
    connection._ua_warned = False
    return connection.noaa_user_agent()


class TestNoaaUserAgent:
    def test_full_override_used_verbatim(self, monkeypatch):
        assert _ua(monkeypatch, NOAA_USER_AGENT="custom/1.0 (me@x.com)",
                   WR_CONTACT_EMAIL="ignored@x.com") == "custom/1.0 (me@x.com)"

    def test_app_and_email(self, monkeypatch):
        assert _ua(monkeypatch, WR_APP_NAME="weather-radar",
                   WR_CONTACT_EMAIL="self@michaelgreen.dev") \
            == "weather-radar (self@michaelgreen.dev)"

    def test_email_only_uses_default_app(self, monkeypatch):
        assert _ua(monkeypatch, WR_CONTACT_EMAIL="a@b.com") \
            == "weather-radar (a@b.com)"

    def test_bare_default_when_nothing_set(self, monkeypatch):
        # Still returns a valid (non-empty) UA so requests aren't blocked.
        assert _ua(monkeypatch) == "weather-radar"

    def test_custom_app_no_email(self, monkeypatch):
        assert _ua(monkeypatch, WR_APP_NAME="myapp") == "myapp"

    def test_blank_override_falls_through(self, monkeypatch):
        # An empty/whitespace override is ignored (treated as unset).
        assert _ua(monkeypatch, NOAA_USER_AGENT="   ",
                   WR_CONTACT_EMAIL="a@b.com") == "weather-radar (a@b.com)"

    def test_warns_once_when_no_contact(self, monkeypatch, caplog):
        for k in ("NOAA_USER_AGENT", "WR_APP_NAME", "WR_CONTACT_EMAIL"):
            monkeypatch.delenv(k, raising=False)
        connection._ua_warned = False
        import logging
        with caplog.at_level(logging.WARNING, logger=connection.logger.name):
            connection.noaa_user_agent()
            connection.noaa_user_agent()  # second call must NOT re-warn
        warnings = [r for r in caplog.records if "NOAA contact" in r.getMessage()]
        assert len(warnings) == 1

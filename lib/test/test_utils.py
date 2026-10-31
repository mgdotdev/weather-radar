from weather_radar.lib.models.utils import to_camel_case


class TestToCamelCase:
    def test_single_token_unchanged(self):
        assert to_camel_case("temperature") == "temperature"

    def test_kebab_converted(self):
        assert to_camel_case("snowfall-amount") == "snowfallAmount"
        assert to_camel_case("quantitative-precipitation") == "quantitativePrecipitation"

    def test_idempotent_on_camel_case(self):
        # Feeding already-camelCase input must return it unchanged, so the
        # conversion is safe to apply more than once along the request path
        # (e.g. map route + data-layer NOAA lookup).
        for value in ("snowfallAmount", "quantitativePrecipitation", "temperature"):
            assert to_camel_case(value) == value

    def test_applying_twice_equals_once(self):
        for value in ("snowfall-amount", "quantitative-precipitation", "temperature"):
            once = to_camel_case(value)
            assert to_camel_case(once) == once

# Weather Radar

This repo is an attempt to build a self-hosted weather forcasting api and map
app.

NOAA forecasting models are free to the public. In an attempt to democratize
the public's access to that data, *their* data, this repository wraps the NOAA
api's behind a local api for more rapid consumption, with the necessary caching
and interpolation methods for functionalizing data across time and space.

This is still a work in progress! MVP will be a precipitation model, but expect
the supported models to expand.

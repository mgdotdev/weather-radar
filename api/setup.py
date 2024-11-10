from setuptools import find_packages, setup

setup(
    name="weather_radar.api",
    version="0.0.1",
    author="Michael Green",
    author_email="self@michaelgreen.dev",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)

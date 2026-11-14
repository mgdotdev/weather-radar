from setuptools import find_namespace_packages, setup

setup(
    name="weather_radar.server",
    version="0.0.1",
    author="Michael Green",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"}
)

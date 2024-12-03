
build() {
    docker build --no-cache -t weather-radar -f .docker/Dockerfile.weather-radar .
}

container() {
    docker run --rm -it \
      -p 8000:8000 \
      -v /tmp:/tmp \
      -v $PWD:/home/weather-radar/code \
      -v /etc/localtime:/etc/localtime:ro \
      weather-radar bash
}

$@


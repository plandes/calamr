# CALAMR Docker image

This repository builds and uses a Docker image to run the source code from the
[zensols.calamr] library.  The [docker image] provides stand-alone container
with all models, configuration and the adhoc micro corpus installed.


## Obtaining

To get the image use `docker pull plandes/calamr`.  However, the [GNU make]
automation (see [usage](#usage)) using `make up` will download the image and
then start it.  Once the image is started, `make align` will align an example
(see the [makefile](./makefile)) and writes the results to `mnt/results`.


## Usage

If you are unfamiliar with Docker, I recommend installing [GNU make] and
[Docker Compose].  The image is completely contained, but for easy access to
results, a volume will need to be mapped.  Those details and GPU hardware
configuration are in the Docker Compose [configuration
file](./docker-compose.yml).

The usage of the container are given in the [GNU makefile](./makefile) `exec`
and `align` targets.  These can be adapted into a script, but the `docker exec`
command will need to be invoked using `su - devusr` since the container is
configured to run under a user account.  The user configuration provides
derivative containers that might want to add a REST service additional
organization and perhaps additional security.

To use the command line directly, use `make ARGS="<CLI arguments>" exec`.  For
example, `make ARGS="--help" exec` prints the command line usage and actions.


## License

MIT License

Copyright (c) 2023 - 2024 Paul Landes


<!-- links -->
[GNU make]: https://www.gnu.org/software/make/
[Docker Compose]: https://docs.docker.com/compose/install/
[docker image]: https://hub.docker.com/r/plandes/calamr
[zensols.calamr]: https://github.com/plandes/calamr

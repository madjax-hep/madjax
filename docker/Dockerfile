ARG BASE_IMAGE=scailfin/madgraph5-amc-nlo:mg5_amc3.5.0
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} as base

USER moby

COPY --chown=moby . /code

# hadolint ignore=DL3003
RUN cd /code && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir . && \
    python -m pip list && \
    cp -r "$(madjax-config)" "$(dirname $(dirname $(command -v mg5_aMC)))/PLUGIN/"

ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]

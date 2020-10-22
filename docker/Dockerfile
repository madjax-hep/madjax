ARG BASE_IMAGE=scailfin/madgraph5-amc-nlo:mg5_amc2.8.1
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} as base

USER root
SHELL [ "/bin/bash", "-c" ]

COPY . /code

# hadolint ignore=DL3003
RUN cd /code && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir . && \
    python -m pip list && \
    cp -r $(madjax-config) /usr/local/MG5_aMC_v2_8_1/PLUGIN/


ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]

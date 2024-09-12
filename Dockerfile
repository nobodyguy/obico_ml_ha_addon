# Ubuntu + CUDA stuff
ARG BUILD_FROM=thespaghettidetective/ml_api_base:1.4
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

########################### HA ADDON SPECIFIC PART ###########################
# Environment variables
ENV \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    DEBIAN_FRONTEND="noninteractive" \
    HOME="/root" \
    LANG="C.UTF-8" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PS1="$(whoami)@$(hostname):$(pwd)$ " \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0 \
    S6_CMD_WAIT_FOR_SERVICES=1 \
    YARN_HTTP_TIMEOUT=1000000 \
    TERM="xterm-256color"

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install base system
ARG BUILD_ARCH=amd64
ARG BASHIO_VERSION="v0.16.2"
ARG S6_OVERLAY_VERSION="3.2.0.0"
ARG TEMPIO_VERSION="2021.09.0"
RUN \
    apt-get update \
    \
    && apt-get install -y --no-install-recommends \
        ca-certificates=20240203 \
        curl=8.5.0-2ubuntu10.1 \
        jq=1.7.1-3build1 \
        tzdata=2024a-2ubuntu1 \
        xz-utils=5.6.1+really5.4.5-1 \
    \
    && S6_ARCH="${BUILD_ARCH}" \
    && if [ "${BUILD_ARCH}" = "i386" ]; then S6_ARCH="i686"; \
    elif [ "${BUILD_ARCH}" = "amd64" ]; then S6_ARCH="x86_64"; \
    elif [ "${BUILD_ARCH}" = "armv7" ]; then S6_ARCH="arm"; fi \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-arch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && mkdir -p /etc/fix-attrs.d \
    && mkdir -p /etc/services.d \
    \
    && curl -J -L -o /tmp/bashio.tar.gz \
        "https://github.com/hassio-addons/bashio/archive/${BASHIO_VERSION}.tar.gz" \
    && mkdir /tmp/bashio \
    && tar zxvf \
        /tmp/bashio.tar.gz \
        --strip 1 -C /tmp/bashio \
    \
    && mv /tmp/bashio/lib /usr/lib/bashio \
    && ln -s /usr/lib/bashio/bashio /usr/bin/bashio \
    \
    && curl -L -s -o /usr/bin/tempio \
        "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}" \
    && chmod a+x /usr/bin/tempio \
    \
    && apt-get purge -y --auto-remove \
        xz-utils \
    && apt-get clean \
    && rm -fr \
        /tmp/* \
        /var/{cache,log}/* \
        /var/lib/apt/lists/*

# Copy root filesystem
COPY rootfs /

# Copy s6-overlay adjustments
COPY s6-overlay /package/admin/s6-overlay-${S6_OVERLAY_VERSION}/

# Entrypoint & CMD
ENTRYPOINT [ "/init" ]

# Build arugments
ARG BUILD_DATE
ARG BUILD_REF
ARG BUILD_VERSION
ARG BUILD_REPOSITORY

# Labels
LABEL \
    io.hass.name="Addon Ubuntu base for ${BUILD_ARCH}" \
    io.hass.description="Home Assistant Community Add-on: ${BUILD_ARCH} Ubuntu base image" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="base" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.base.version=${BUILD_VERSION} \
    io.hass.base.name="ubuntu" \
    io.hass.base.image="hassioaddons/ubuntu-base" \
    maintainer="Franck Nijhof <frenck@addons.community>" \
    org.opencontainers.image.title="Addon Ubuntu base for ${BUILD_ARCH}" \
    org.opencontainers.image.description="Home Assistant Community Add-on: ${BUILD_ARCH} Ubuntu base image" \
    org.opencontainers.image.vendor="Home Assistant Community Add-ons" \
    org.opencontainers.image.authors="Franck Nijhof <frenck@addons.community>" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.url="https://addons.community" \
    org.opencontainers.image.source="https://github.com/${BUILD_REPOSITORY}" \
    org.opencontainers.image.documentation="https://github.com/${BUILD_REPOSITORY}/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}


########################### OBICO ML API SPECIFIC PART ###########################
EXPOSE 3333

WORKDIR /app
ADD . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN echo 'Downloading the latest failure detection AI model in Darknet format...'
RUN wget -O model/model-weights.darknet $(cat model/model-weights.darknet.url | tr -d '\r')
RUN echo 'Downloading the latest failure detection AI model in ONNX format...'
RUN wget -O model/model-weights.onnx $(cat model/model-weights.onnx.url | tr -d '\r')
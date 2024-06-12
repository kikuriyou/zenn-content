FROM node:22

WORKDIR /workspace

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    git \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN npm init --yes \
    && npm install -g zenn-cli@latest

# Locale Japanese
ENV LC_ALL=ja_JP.UTF-8
# Timezone jst
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
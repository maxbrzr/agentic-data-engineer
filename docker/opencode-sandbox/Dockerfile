FROM node:22-trixie-slim

ARG OPENCODE_VERSION=1.18.9

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global "opencode-ai@${OPENCODE_VERSION}"

WORKDIR /opt/agentic-data-engineer

RUN python3 -m venv /opt/agent-runtime \
    && /opt/agent-runtime/bin/pip install --no-cache-dir \
        numpy==2.5.1 \
        openpyxl==3.1.5 \
        pandas==3.0.5 \
        pyarrow==25.0.0 \
        scikit-learn==1.9.0

COPY src ./src
COPY docker/opencode-sandbox/attestation_server.py ./attestation_server.py
COPY docker/opencode-sandbox/opencode.json /etc/opencode/opencode.json
COPY docker/opencode-sandbox/entrypoint.sh /usr/local/bin/opencode-sandbox-entrypoint

RUN chmod 0555 /usr/local/bin/opencode-sandbox-entrypoint

ENV PATH="/opt/agent-runtime/bin:${PATH}" \
    PYTHONPATH="/opt/agentic-data-engineer/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCODE_CONFIG=/etc/opencode/opencode.json

USER node

ENTRYPOINT ["opencode-sandbox-entrypoint"]
CMD ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "54321"]

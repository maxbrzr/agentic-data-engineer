FROM node:24-trixie-slim

ARG PI_VERSION=0.82.1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global --ignore-scripts \
    "@earendil-works/pi-coding-agent@${PI_VERSION}"

WORKDIR /opt/agentic-data-engineer

RUN python3 -m venv /opt/agent-runtime \
    && /opt/agent-runtime/bin/pip install --no-cache-dir \
        numpy==2.5.1 \
        openpyxl==3.1.5 \
        pandas==3.0.5 \
        pyarrow==25.0.0 \
        scikit-learn==1.9.0

COPY src ./src
COPY docker/pi/models.json /etc/pi/models.json
COPY docker/pi/runner.py ./pi_runner.py
COPY docker/pi/entrypoint.sh /usr/local/bin/pi-entrypoint

RUN chmod 0555 /usr/local/bin/pi-entrypoint

ENV PATH="/opt/agent-runtime/bin:${PATH}" \
    PYTHONPATH="/opt/agentic-data-engineer/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PI_CODING_AGENT_DIR=/tmp/pi-home/agent \
    PI_OFFLINE=1 \
    PI_TELEMETRY=0

USER node

ENTRYPOINT ["pi-entrypoint"]
CMD ["python3", "/opt/agentic-data-engineer/pi_runner.py"]

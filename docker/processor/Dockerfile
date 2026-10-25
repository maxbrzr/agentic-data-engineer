# Runtime for checking a generated processor outside the agent: the same
# Debian Python and packages as the agent images, without the agent or network.
FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY docker/processor/requirements.txt /opt/processor-runtime/requirements.txt
RUN python3 -m venv /opt/processor-runtime \
    && /opt/processor-runtime/bin/pip install --no-cache-dir \
        -r /opt/processor-runtime/requirements.txt

COPY docker/processor/run_processor.py /opt/processor-runtime/run_processor.py

ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/opt/processor-runtime/bin/python", "/opt/processor-runtime/run_processor.py"]

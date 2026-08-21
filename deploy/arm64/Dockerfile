FROM --platform=linux/arm64 python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/animal-monitor

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install .

COPY config ./config
COPY models ./models

ENTRYPOINT ["animal-monitor"]
CMD ["info"]

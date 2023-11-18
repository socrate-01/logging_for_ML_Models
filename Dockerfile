# syntax=docker/dockerfile:1
FROM python:3.9-slim as base

WORKDIR /dependencies

# installing git because we need to install the model package from the github repository
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git

# creating and activating a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# installing dependencies
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install --no-cache -r service_requirements.txt

FROM base as runtime

ARG DATE_CREATED
ARG REVISION
ARG VERSION

LABEL org.opencontainers.image.title="Logging for ML Models"
LABEL org.opencontainers.image.description="Logging for machine learning models."
LABEL org.opencontainers.image.created=$DATE_CREATED
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/schmidtbri/logging-for-ml-models"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.revision=$REVISION
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="python:3.9-slim"

WORKDIR /service

# install packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/venv ./venv

COPY ./ml_model_logging ./ml_model_logging
COPY ./LICENSE ./LICENSE

ENV PATH /service/venv/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/service"

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

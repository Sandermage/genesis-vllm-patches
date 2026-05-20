# SPDX-License-Identifier: Apache-2.0
# SNDR spec-decode gateway image.
#
# This image carries ONLY the gateway service. It does not include
# vLLM, torch, or any model weights. The gateway speaks OpenAI-compat
# HTTP to two upstream vLLM containers (default + structured) over
# the network.
#
# Build context: the repo root (`genesis-vllm-patches/`). The build
# only consumes the sub-tree it needs.
#
# Example:
#   docker build -t sndr-gateway:dev \
#       -f vllm/sndr_core/integrations/spec_decode/gateway/deploy/Dockerfile \
#       .

FROM python:3.12-slim

# Minimal runtime deps. Pinned loosely; the gateway is small enough
# that minor version drift is fine.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "httpx>=0.27" \
        "prometheus-client>=0.20" \
    && rm -rf /root/.cache

WORKDIR /app

# Copy ONLY the sndr_core sub-tree the gateway actually imports.
# Heavy runtime/integrations modules are lazy-imported and remain
# absent at runtime (gateway never triggers them).
COPY vllm/sndr_core/__init__.py        /app/vllm/sndr_core/__init__.py
COPY vllm/sndr_core/version.py         /app/vllm/sndr_core/version.py
COPY vllm/sndr_core/brand.py           /app/vllm/sndr_core/brand.py
COPY vllm/sndr_core/env.py             /app/vllm/sndr_core/env.py
COPY vllm/sndr_core/locations          /app/vllm/sndr_core/locations
COPY vllm/sndr_core/detection          /app/vllm/sndr_core/detection
COPY vllm/sndr_core/integrations/__init__.py \
                                       /app/vllm/sndr_core/integrations/__init__.py
COPY vllm/sndr_core/integrations/spec_decode/__init__.py \
     vllm/sndr_core/integrations/spec_decode/kv_contract.py \
     vllm/sndr_core/integrations/spec_decode/kv_bridge.py \
     vllm/sndr_core/integrations/spec_decode/functional_artifact.py \
     vllm/sndr_core/integrations/spec_decode/request_router.py \
     vllm/sndr_core/integrations/spec_decode/safety_guard.py \
     vllm/sndr_core/integrations/spec_decode/pn274_install.py \
                                       /app/vllm/sndr_core/integrations/spec_decode/
COPY vllm/sndr_core/integrations/spec_decode/mapping \
                                       /app/vllm/sndr_core/integrations/spec_decode/mapping
COPY vllm/sndr_core/integrations/spec_decode/artifacts \
                                       /app/vllm/sndr_core/integrations/spec_decode/artifacts
COPY vllm/sndr_core/integrations/spec_decode/gateway \
                                       /app/vllm/sndr_core/integrations/spec_decode/gateway

# The sndr_core __init__.py contains import-time hooks (G4_19 family
# + PN274) that try to import vllm internals. In this image vllm is
# absent; the hooks catch ImportError silently. To skip the noise
# entirely we set the documented disable env at image-build time.
# Operator can override at run time.
ENV SNDR_DISABLE_SPEC_DECODE_SAFETY_GUARD=1

# Default gateway config. Override at `docker run` / compose time.
ENV SNDR_GATEWAY_BIND_HOST=0.0.0.0 \
    SNDR_GATEWAY_BIND_PORT=8100 \
    SNDR_GATEWAY_DEFAULT_URL=http://default:8101 \
    SNDR_GATEWAY_STRUCTURED_URL=http://structured:8102 \
    SNDR_GATEWAY_PROFILE=gemma4-tq-mtp-structured-k4 \
    SNDR_GATEWAY_HEALTH_INTERVAL=5 \
    SNDR_GATEWAY_TIMEOUT=120 \
    SNDR_GATEWAY_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8100

HEALTHCHECK --interval=10s --timeout=2s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8100/healthz', timeout=1).status == 200 else 1)" \
    || exit 1

# Run via the package's __main__.
CMD ["python", "-m", "vllm.sndr_core.integrations.spec_decode.gateway"]

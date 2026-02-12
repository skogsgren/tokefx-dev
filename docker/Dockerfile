FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -c "import torch; print(torch.version.cuda); print(torch.backends.cudnn.version())"
WORKDIR /workspace
CMD ["/bin/bash"]

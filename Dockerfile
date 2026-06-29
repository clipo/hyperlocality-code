# Reproducible environment for the Rapa Nui hyperlocality analysis.
#
# Build:  docker build -t hyperlocality-code .
# Run the full pipeline (results -> ./output, figures -> ./figures on the host):
#   docker run --rm -v "$PWD/output:/work/output" -v "$PWD/figures:/work/figures" \
#       hyperlocality-code
# Open a shell instead:
#   docker run --rm -it hyperlocality-code bash
#
# The numeric pipeline runs fully offline. The figures step downloads a
# shaded-relief basemap, so it needs network access at run time.
FROM python:3.12-slim

WORKDIR /work

# PyMC/PyTensor compile C at runtime, so a C/C++ toolchain is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*

# Pinned dependencies first, so the layer caches independently of the code.
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Analysis code and data.
COPY . .

# Default: reproduce everything.
CMD ["bash", "reproduce.sh"]

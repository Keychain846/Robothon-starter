# Build from the REPOSITORY ROOT (the submission depends on the shared
# robot meshes in <repo>/assets/):
#
#   docker build -f submissions/ff_master_futurist/Dockerfile -t ff-futurist .
#   docker run --rm -v "$PWD/out:/app/output" ff-futurist            # full video
#   docker run --rm ff-futurist python scripts/record_robot_video.py --quick
#
FROM python:3.11-slim

# libosmesa6 provides software off-screen rendering (no GPU required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libglu1-mesa libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

WORKDIR /app
COPY submissions/ff_master_futurist/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# shared robot meshes, referenced as ../../assets/... from the submission
COPY assets/Master   /assets/Master
COPY assets/Futurist /assets/Futurist
COPY submissions/ff_master_futurist /app

CMD ["python", "scripts/record_robot_video.py", "--seed", "42", "--n-episodes", "8"]

# The image the live app runs in (Hugging Face Space, CPU only).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_PROGRESS_BARS=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user . .

# Benchmarks, the fine-tuned verifier and the SciFact index are built into the image,
# so the first visitor does not wait for them
RUN python -m verity.data && python -m verity.models verifier && python -m deploy.warm

EXPOSE 7860
# XSRF protection blocks file uploads inside the Space's iframe
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]


FROM python:3.11-slim

WORKDIR /code
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app /code/app
WORKDIR /code/app

# Initialize DB on first run (best-effort)
CMD ["bash", "-lc", "python -m db --init && sleep infinity"]

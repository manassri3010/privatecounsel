# Built from the repository root (see docker-compose.yml): the API imports ingestion/ and retrieval/.
FROM python:3.12-slim
WORKDIR /srv
# CPU-only PyTorch for the embedding model; the default wheel bundles GPU libraries the API doesn't use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt api-requirements.txt
COPY ingestion/requirements.txt ingestion-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt -r ingestion-requirements.txt
COPY api/app ./app
COPY ingestion ./ingestion
COPY retrieval ./retrieval
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

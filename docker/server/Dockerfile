FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY server_aggregator.py /app/server_aggregator.py
COPY static/ /app/static/

EXPOSE 8080

CMD ["python", "/app/server_aggregator.py"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY client_transfer.py /app/client_transfer.py

CMD ["python", "/app/client_transfer.py"]

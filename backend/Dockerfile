FROM python:3.11-slim

WORKDIR /api/main

COPY requirements.txt .

# Upgrade pip + increase timeout
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
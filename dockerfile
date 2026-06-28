FROM python:3.11-slim

COPY requirements.txt .

COPY . .
EXPOSE 5000
CMD ["python", "application.py"]
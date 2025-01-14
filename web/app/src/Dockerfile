# Use an official Python runtime as a parent image
# FROM python:3.13-slim
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Copy the SSL certificates into the container
COPY /ssl-certs /app/certs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install SQLite
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Change port from 443 to 8080
EXPOSE 443
EXPOSE 80
# do port 80 for testing local

# Updated Gunicorn command without SSL parameters, this is for local testing docker conatiner
#cCMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "--log-level=debug", "app:app"]

# Expose the application port
# EXPOSE 443

# Run Gunicorn with SSL support
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:443", "--certfile=/app/certs/fullchain.pem", "--keyfile=/app/certs/privkey.pem", "app:app"]
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:443", "--certfile=/app/certs/fullchain.pem", "--keyfile=/app/certs/privkey.pem", "--log-level=debug", "app:app"]

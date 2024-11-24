# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Copy the SSL certificates into the container
COPY /ssl-certs /app/certs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 443

# Run Gunicorn with SSL support
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:443", "--certfile=/app/certs/fullchain.pem", "--keyfile=/app/certs/privkey.pem", "app:app"]

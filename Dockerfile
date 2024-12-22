# Use an official Python runtime as the base image (compatible version)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application files to the container
COPY . .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords

# Download spaCy model separately
RUN python -m spacy download en_core_web_sm

# Expose the port that Streamlit runs on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "application.py", "--server.port=8080", "--server.enableCORS=false"]

# Use a slim Python 3.10 image as the base layer
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the project's root requirements.txt to the working directory
COPY requirements.txt ./

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the GraphQL API code (schema definition) into the container
COPY graphql_api/schema.py ./

# Define the command to run the GraphQL API using Uvicorn on port 8002
CMD ["uvicorn", "schema:app", "--host", "0.0.0.0", "--port", "8002"]

# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the project root to the container
COPY requirements.txt ./

# Install Python dependencies without cache to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI service code into the container
COPY python_service/main.py ./

# Define the command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

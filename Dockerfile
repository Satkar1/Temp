# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install uvicorn

# Copy the rest of the app
COPY . .

# Expose port (Vercel expects 3000 for Docker web services)
EXPOSE 3000

# Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]


Act as an expert Software Engineer and career coach. I need you to write my yearly goals for my performance management form at PayU based on my actual work history, project contributions, and technical stack. 


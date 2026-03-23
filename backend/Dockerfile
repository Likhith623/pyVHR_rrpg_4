FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app
COPY requirements_rppg.txt .

# Install dependencies needed for opencv and ML libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libomp-dev

RUN pip install --no-cache-dir -r requirements_rppg.txt

COPY . .

CMD ["uvicorn", "service_rppg:app", "--host", "0.0.0.0", "--port", "8001"]
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && apt-get install -y tk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY main_app.py polygon_person.py .  
COPY sort/ sort/                      
COPY yolov5/ yolov5/                  

EXPOSE 8000

CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]

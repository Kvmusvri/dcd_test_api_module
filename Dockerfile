FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Нейро-стек ray-конвейера: torch (CUDA 12.4, работает и на CPU) +
# intrinsic decomposition (compphoto/Intrinsic, официальное издание пакета).
# git обязателен: зависимости Intrinsic (altered_midas, chrislib) ставятся
# с git+https — без него build падает "Cannot find command 'git'".
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir https://github.com/compphoto/Intrinsic/archive/main.zip

# opencv 5.x (тянет chrislib) не упаковывает модуль cv2 → откат на 4.10 headless.
RUN pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.10.0.84

COPY app ./app
COPY frontend ./frontend

EXPOSE 8100

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]

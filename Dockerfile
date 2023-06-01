# Base image
FROM python:3.8-slim-buster

# Set the working directory

WORKDIR /workspace
   
# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html



COPY train.py /workspace/

CMD ["python", "train.py"]
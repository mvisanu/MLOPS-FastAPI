# MLOPS-FastAPI

ML Ops

https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

conda create --name mlops-student python=3.8 pip

conda activate mlops-student

//pytorch model need
pip install Pillow torch torchvision torchaudio

//tensorflow keras
pip install tensorflow keras

python pytorch_prediction.py

python tensorflow_prediction.py

---FastApi---
pip install fastapi uvicorn[standard]
-- to run fastapi server cd to folder --with debug mode can use hot reload
uvicorn main:app --reload

GRPC - google remote procedure call
pip install grpcio grpcio-tools

python -m grpc_tools.protoc --python_out=protos --grpc_python_out=protos cat_vs_dogs.proto --proto_path=protos

python -m grpc_tools.protoc --python_out=. --grpc_python_out=. cats_vs_dogs.proto --proto_path=.

------------------------------------
- Protocol: http2
- Payload: (binary, small)
- API Contract: Strict, required (.proto)
- Code Generation: Built-in (protoc)
- Security: TLS/SSL
- Streaming: Bidrectional steaming
- Browser support: Limited (require gRPC-Web)

REST API
------------------------------------
- Protocol: HTTP/1.1 (slow)
- Payload: JSON (text, large)
- API contract: Loose, optional (OpenAPI)
- Code generation: Third-party tools (Swagger)
- Security: TLS/SSL
- Streaming: Client -> server request only
- Browser support: Yes


Docker 
Dockerfile --> DockerImage --> Docker Container
pip freeze > requirements.txt

Dockerfile
FROM python:3.8

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app"]

--commad to build dockerfile --
docker build -t mlops-beginner-fastapi .

--run docker command--
docker run --rm -it -p 8000:8000 mlops-beginner-fastapi

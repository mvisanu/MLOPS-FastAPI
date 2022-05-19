## Generate Protos
> cd grpc
python -m grpc_tools.protoc --python_out=./protos --grpc_python_out=./protos ./protos/cat_vs_dogs.proto --proto_path=./protos

import grpc
import protos.cats_vs_dogs_pb2 as cats_vs_dogs_pb2
import protos.cats_vs_dogs_pb2_grpc as cats_vs_dogs_pb2_grpc

graham_img = "../graham2.jpg"
channel = grpc.insecure_channel("localhost:50000")
client = cats_vs_dogs_pb2_grpc.CatsVsDogsServiceStub(channel)

with open(graham_img, 'rb') as img_file:
    img_bytes = img_file.read()

request = cats_vs_dogs_pb2.CatsVsDogsRequest(image=img_bytes)

tf_response = client.CatsVsDogsTensorflowInference(request)

#print(tf_response)

pytorch_response = client.CatsVsDogsPyTorchInference(request)

print("tf: ", tf_response, "pytorch: ", pytorch_response)
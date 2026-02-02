from utils import predict_gpt, client

test = predict_gpt(client, messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

print(test)
import zmq, json
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect('tcp://127.0.0.1:5600')
s.setsockopt(zmq.SUBSCRIBE, b'js8/decode')
while True:
    topic, payload = s.recv_multipart()
    print(json.loads(payload))

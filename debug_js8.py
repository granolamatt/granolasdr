import zmq, json, sys
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect('tcp://127.0.0.1:5590')
s.setsockopt(zmq.SUBSCRIBE, b'')
while True:
    print(json.loads(s.recv()))


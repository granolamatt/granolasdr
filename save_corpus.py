import zmq, sys
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect('tcp://localhost:5580')
s.setsockopt(zmq.SUBSCRIBE, b'')
while True: print(s.recv_string(), flush=True)


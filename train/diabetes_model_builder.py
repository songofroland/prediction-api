import json

import zmq

from utils import train

port = "5555"
socket = zmq.Context().socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

while True:
    print("Job executor is waiting for a new job request.")
    msg = socket.recv()
    print("Job executor has received a new job request.")
    msg_dict = json.loads(msg.decode())
    print(msg_dict)
    print("Job execution is starting.")

    train(dev=True)

    print("Job is completed.")
    print("Job executor is about to report the completion of the job back to the requester.")
    msg_content_dict = {"executed_job_id": msg_dict["job_id"]}
    msg_content = str(json.dumps(msg_content_dict))
    msg_string = msg_content.encode()
    socket.send(msg_string)

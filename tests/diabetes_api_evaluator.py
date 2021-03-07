import json
import time

import zmq

from utils import run_test

EXECUTOR_CONN_STR = "tcp://localhost:5555"

socket = zmq.Context().socket(zmq.PAIR)
socket.connect(EXECUTOR_CONN_STR)

job_id = str(hash(time.time()))
msg_string = json.dumps({"job_id": job_id}).encode()
print("A new job request is about to be sent to the job executor.")
socket.send(msg_string)
print("Job requester is waiting for confirmation of the new job completion.")
response = socket.recv()
parsed_response = json.loads(response.decode())

if parsed_response["executed_job_id"] != job_id:
    raise ValueError("Job requester has received some unexpected message.")

print("Job requester has received the confirmation of the new job completion.")
print("Running tests...")
run_test()
print("Tests completed.")

from flask import Flask, request, jsonify
from imutils.video import VideoStream
import json
import cv2
import socket
import threading
import time

speech_response = None

def send(ioi_name):

    global speech_response

    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

    try:

        sock.sendto(ioi_name.encode(), (UDP_IP, UDP_PORT))
        print('ioi name sent for', ioi_name)
        print('waiting for response...')
        data, server = sock.recvfrom(4096)

        if data:
            speech_response = data.decode()
            print("I got the speech response back:", speech_response)

    finally:
        sock.close()

    # return speech_response

#     # UDP listener, separate thread
# def listen():

#     global speech_response

#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 5001
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

#     sock.bind((UDP_IP, UDP_PORT))

#     while True:
#         response, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
#         speech_response = response.decode()
#         print("I got the speech response back:", speech_response)


app = Flask(__name__)

@app.route('/<uuid>', methods=['GET', 'POST'])
def parse_post_request(uuid):

    global speech_response

    # parse the json to get the ioi name
    content = request.json  # get the json object
    ioi_name = content['result']['parameters']['Object']  # parse the ioi

    response = {}  # declare the response var

    # change ioi name for mobile phone to be consistent
    if ioi_name == 'phone' or ioi_name == 'mobile phone' or ioi_name == 'cellphone':
        ioi_name = 'cell phone'

    if ioi_name == "teddy bears":
        ioi_name == 'teddy bear'

    print('ioi name from google home after change', ioi_name)

    # send ioi data to camera listener
    send(ioi_name)
    print('sent ioi to camera listener')

    
    # wait until the bbox is received from listener
    while not speech_response:
        # wait until it's received
        pass

    print('speech response received from camera listener!', speech_response)

    response = {}

    response['speech'] = speech_response

    json_data = json.dumps(response)  # put text in a json object

    # reset the speech response to None
    speech_response = None

    return json_data  # return the json object

if __name__ == '__main__':

    # t1 = threading.Thread(target=listen)

    # t1.setDaemon(True)
    # # starting thread 1 
    # t1.start()

        # run the server
    app.run()

    try:
        while True:

            time.sleep(1)

    except KeyboardInterrupt:
        print("exiting")
        exit(0)



    
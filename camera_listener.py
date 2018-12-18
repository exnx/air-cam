import socket
import threading
from flask import Flask, request, jsonify
from imutils.video import VideoStream
import cv2
import time
import random
import math

from ssd_inference import SSD
from yolo_inference import Yolo


#  -----------------------------------streaming files ---------------------------------------#



# global variables
do_inference_on_ioi = None  # track when to do inference


bbox_queue = []  # track when new bbox received





        # append the data to queue
        # check_inference.append(run_inference)

# def send(data):
#     # variables for UDP
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 5001
     
#     sock = socket.socket(socket.AF_INET, # Internet
#                           socket.SOCK_DGRAM) # UDP
#     sock.bind((UDP_IP, UDP_PORT))

#     print('sending bounding boxes back to json server')
#     sock.sendto(data.encode(), addr)

    






#  ---------------------------------object finder files------------------------------------------  #


def get_dist(ioi_bbox, other_bbox):

    ioi_x1, ioi_y1, ioi_x2, ioi_y2 = ioi_bbox  # unpack ioi bbox
    oth_x1, oth_y1, oth_x2, oth_y2 = other_bbox  # unpack other item_bbox

    # calc the ioi centroid
    ioi_xc = int((ioi_x2 - ioi_x1)/2 + ioi_x1)
    ioi_yc = int((ioi_y2 - ioi_y1)/2 + ioi_y2)

    # calc the other object centroid
    # calc the ioi centroid
    oth_xc = int((oth_x2 - oth_x1)/2 + oth_x1)
    oth_yc = int((oth_y2 - oth_y1)/2 + oth_y2)

    # calc the x and y dist
    x_dist = ioi_xc - oth_xc
    y_dist = ioi_yc - oth_yc

    # calc euclidian distance
    dist = math.sqrt( (math.pow(x_dist,2) + math.pow(y_dist,2)) )

    return int(dist)  # return the dist

def find_close_items(ioi_data, other_items):

    '''

    input: a list of 2 or more objects, the first is the item of interest (ioi)

    return:  a list of tuples of close object names and their distance to the ioi

    '''

    # items_found_tuple:  list of tuples.  (item-name, x-coord, y-coord) 

    object_dist_threshold = 125

    ioi_name, ioi_x1, ioi_y1, ioi_x2, ioi_y2 = ioi_data  # unpack ioi
    ioi_bbox = ioi_x1, ioi_y1, ioi_x2, ioi_y2

    close_item_distances = []  # store tuples of other items and dist to ioi

    print('other items before checking threshold', other_items)

    # loop through the other objects and get the distances bwn IoI and other objects 
    for other_item in other_items:

        other_item_name, curr_x1, curr_y1, curr_x2, curr_y2 = other_item  # unpack the other item
        other_bbox = curr_x1, curr_y1, curr_x2, curr_y2

        dist = get_dist(ioi_bbox, other_bbox)
        print('getting distance from {} to {} is {} pixels:'.format(ioi_data[0], other_item, dist))

        # check if meets threshold for close enough
        if dist < object_dist_threshold and other_item_name != 'book':  # remove book from the items, false positive

            print('distance is less than threshold!', other_item)
            close_item_distances.append( (other_item_name, dist) )  # add to list

    # sort by distance
    close_item_distances.sort(key=lambda tup : tup[1])

    print('close item distances sorted:', close_item_distances)

    return close_item_distances[:3]


def retrieve_ioi_from_list(ioi_name, found_objects):

    # print('ioi name inside retrieve_from_list', ioi_name)

    ioi_data = None  # empty tuple

    # print('found objects after before ioi', found_objects)

    for index, item in enumerate(found_objects):

        item_name = item[0]  # first element, first entry the item name

        print('item 0 0', item_name)

        if item_name == ioi_name:

            print("ioi item found at index:", index)

            ioi, x1, y1, x2, y2 = found_objects.pop(index)
            ioi_data = (ioi_name, x1, y1, x2, y2)

    # print('found objects after removing ioi', found_objects)
    # print('ioi data in retrieve ioi from list:', ioi_data)

    return ioi_data, found_objects  # if ioi not found, then is None

def find_objects(frame, ioi_name):

    frame, gg_name, gg_found, ggx1, ggy1, ggx2, ggy2 = ssd.run_inference_for_single_image(frame)  # find the glue gun

    # found_objects = list of objects and bounding boxes
    frame, found_objects = yolo.infer(frame)  # find other objects and yolo draw bboxes too

    # if glue gun found, need to combine glue gun with other found objects
    if gg_found:
        found_objects.append([gg_name, ggx1, ggy1, ggx2, ggy2])  # insert at beginning of list

    return found_objects

def generate_google_speech(ioi_name, ioi_data, other_objects):

    # if ioi_name == "teddy bear":
    #     ioi_name = "pluto"

    response = ''  # speech response to return

    # if ioi is found
    if ioi_data:

        # store close items, a list of tuples, with idx item
        close_item_distances = []

        # if found only the ioi
        if len(other_objects) < 1:

            # say where it is, table or floor
            print('ioi found, but no other objects found, so show last location of ioi')

            response = "I found the {}, it's on the table!".format(ioi_name)  # build the speech text

        # found 1 or more other ojects, get distances to other objects
        elif len(other_objects) > 0:
            close_item_distances = find_close_items(ioi_data, other_objects)  # only pass first 2 other objects
            print('close items distances', close_item_distances)

        # if found 1 close object
        if close_item_distances and len(close_item_distances) == 1:

            other_item_name = close_item_distances[0][0]

            print('say its next to the other object')
            response = 'the {} is by the {}'.format(ioi_name, other_item_name)  # build the speech text

        elif close_item_distances and len(close_item_distances) > 1:

            print('say its next to two objects')

            other_item1 = close_item_distances[0][0]  # unpack the 1st item name
            other_item2 = close_item_distances[1][0]  # unpack the 2nd item name

            response = 'the {} is by the {} and the {}'.format(ioi_name, other_item1, other_item2)  # build the speech text

        else:

            print('close item distances', close_item_distances)

            print('found the ioi, but its not close to anything')

            if ioi_data[4] > 350:
                absolute_location = 'floor'

            else:
                absolute_location = "table"

            response = "I found the {}, it's on the {}".format(ioi_name, absolute_location)  # build the speech text


    # didnt find the item of interest
    else:

        print('didnt find any objects, so find the last location of the ioi')
        # retrieve from data base...
        response = "sorry, I could't find the {}, but it was last on the table".format(ioi_name)  # build the speech text

    # return the speech response
    return response

def write_to_csv(found_objects):

    with open('box.csv', mode='w') as box_file:
        item_writer = csv.writer(box_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # loop through all found items
        for item in found_objects:
            item_writer.writerow(item)  # write row to csv

def get_color():

    colors = [(255,0,0), (0,255,0), (102,204,0), (204,0,102), (0,255,255), (127,0,255), (255,128,0)]

    num = random.randint(0,6)

    print('num', num)

    return colors[num]


def draw_bounding_box(frame, found_items):

    # draw the data on the frame

    # print('found_items to draw', found_items)

    for ind, item in enumerate(found_items):

        item_name, x1, y1, x2, y2 = item

            # draw a bounding box rectangle and label on the frame
        # color_tup = get_color()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, item_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


ssd = SSD()
yolo = Yolo()

# video streamer, separate thread
class MyCam:

    def __init__(self):
        self.vs = VideoStream(src=1).start()
        time.sleep(1.5)

    def get_image(self):
        return self.vs.read()

frame = None

def run_inference(ioi_name):

    global bbox_queue, frame

    print('do inference now...')
    frame, gg_name, gg_found, ggx1, ggy1, ggx2, ggy2 = ssd.run_inference_for_single_image(frame)  # find the glue gun
    frame, found_objects = yolo.infer(frame)  # find other objects and yolo draw bboxes too

    print('found_objects before retrieving ioi', found_objects)

    if gg_found:
        found_objects.append(['glue gun', gg_xc, gg_yc])  # insert at beginning of list

    ioi_data, other_objects = retrieve_ioi_from_list(ioi_name, found_objects)  # will scan through and find which is ioi, and rest of objects

    print('generating speech...')
    speech_response = generate_google_speech(ioi_name, ioi_data, other_objects)

    print('ioi data', ioi_data)
    print('found_objects after retrieving ioi', found_objects)


    # need to add back the ioi to draw
    found_objects.append(ioi_data)

    print('found_objects after appending ioi', found_objects)

    # # append bounding boxes to bbox_queue
    bbox_queue.append(found_objects)

    # return bboxes by udp to post request server
    # send(speech_response)
    print('speech sent back')

    return speech_response




    #### ----------------------



# UDP listener, separate thread
def listen():

    global do_inference_on_ioi

    # variables for UDP
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
     
    sock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT))

    

    while True:
        data, address = sock.recvfrom(1024) # buffer size is 1024 bytes
        print("do inference received!")

        if data:

            ioi_name = data.decode()

            speech_text = run_inference(ioi_name)

            # wait for bounding boxes........

            print('sending the data back right away')
            sent = sock.sendto(speech_text.encode(), address)




def stream():

    scale = 2

    threshold = 2000  # number of frames to display bounding box

    global do_inference_on_ioi, bbox_queue, frame

    my_cam = MyCam()

    display_count = 0

    while True:

        frame = my_cam.get_image()

        # print('do inference value', do_inference_on_ioi)

        # check if something in queue
        if len(bbox_queue) > 0:

            # print('have a new bounding box:', bbox_queue)

            # get coordinates from 
            # sent a udp message back to the server, to allow it to return the json object 

            # make up the frame
            # cv2.rectangle(frame, (x1, y1), (x2, y2), get_color(row_num), 2)

            # data = bbox_queue[0]  # retrieve data from queue
            # coords = data.decode("utf-8")   # decode byte data

            #  display the bbox on frame
            found_items = bbox_queue[0]
            draw_bounding_box(frame, found_items)

            display_count += 1
            # print('display_count', display_count )

            # count reaches threshold, remove from queue and reset count
            if display_count > threshold:
                bbox_queue.pop()
                display_count = 0

        enlarge = cv2.resize(frame, (0,0), fx=scale, fy=scale) 
        cv2.imshow('file', enlarge)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    my_cam.vs.stop()
    cv2.destroyAllWindows()


def main():

    t1 = threading.Thread(target=listen)
    # t2 = threading.Thread(target=stream)

    # starting thread 1 
    t1.start() 
    # starting thread 2 
    stream()


if __name__ == '__main__':

    main()
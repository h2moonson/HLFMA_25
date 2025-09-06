#!/usr/bin/python3

import rospy
from datetime import datetime

#from nmea_msgs.msg import Sentence
from rtcm_msgs.msg import Message

from base64 import b64encode
from threading import Thread

from http.client import HTTPConnection
from http.client import IncompleteRead

''' This is to fix the IncompleteRead error
    http://bobrochel.blogspot.com/2010/11/bad-servers-chunked-encoding-and.html'''
import http.client

import requests
from requests.auth import HTTPBasicAuth



def NiceToICY(self):
    class InterceptedHTTPResponse():
        pass
    import io
    line = self.fp.readline().replace(b"ICY 200 OK\r\n", b"HTTP/1.0 200 OK\r\n")
    InterceptedSelf = InterceptedHTTPResponse()
    InterceptedSelf.fp = io.BufferedReader(io.BytesIO(line))
    InterceptedSelf.debuglevel = self.debuglevel
    InterceptedSelf._close_conn = self._close_conn
    return ORIGINAL_HTTP_CLIENT_READ_STATUS(InterceptedSelf)

ORIGINAL_HTTP_CLIENT_READ_STATUS = http.client.HTTPResponse._read_status
http.client.HTTPResponse._read_status = NiceToICY


def patch_http_response_read(func):
    def inner(*args):
        try:
            return func(*args)
        except http.client.IncompleteRead as e:
            return e.partial
    return inner
http.client.HTTPResponse.read = patch_http_response_read(http.client.HTTPResponse.read)


class ntripconnect(Thread):
    def __init__(self, ntc):
        super(ntripconnect, self).__init__()
        self.ntc = ntc
        self.stop = False

    def run(self):
        print("[DEBUG] NTRIP Connection thread started")
        headers = {
            'Ntrip-Version': 'Ntrip/2.0',
            'User-Agent': 'NTRIP ntrip_ros',
            'Connection': 'keep-alive',
            'Authorization': 'Basic ' + b64encode((self.ntc.ntrip_user + ':' + self.ntc.ntrip_pass).encode()).decode("ascii")
        }
        connection = HTTPConnection(self.ntc.ntrip_server, timeout=30)
        connection.set_debuglevel(3)
        print("[DEBUG] Sending GET request to NTRIP caster")
        connection.request('GET', '/' + self.ntc.ntrip_stream, self.ntc.nmea_gga, headers)
        response = connection.getresponse()
        print(f"[DEBUG] Server response: {response.status} {response.reason}")
        
        if response.status != 200:
            print("[ERROR] Failed to connect to NTRIP server")
            return
        
        buf = ""
        rmsg = Message()
        restart_count = 0

        while not self.stop:
            print("[DEBUG] Waiting for RTCM data...")
            data = response.read(1)
            rospy.loginfo(f"{not data}")
            if len(data) != 0:
                if data[0] == 211:
                    print("[DEBUG] RTCM header detected")
                    buf = []
                    buf.append(data[0])
                    data = response.read(2)
                    buf.append(data[0])
                    buf.append(data[1])
                    cnt = data[0] * 256 + data[1]
                    data = response.read(2)
                    buf.append(data[0])
                    buf.append(data[1])
                    typ = (data[0] * 256 + data[1]) / 16
                    print(f"[DEBUG] RTCM Message: Length={cnt}, Type={typ}")
                    cnt = cnt + 1
                    for x in range(cnt):
                        data = response.read(1)
                        buf.append(data[0])
                    rmsg.message = buf
                    rmsg.header.seq += 1
                    rmsg.header.stamp = rospy.get_rostime()
                    self.ntc.pub.publish(rmsg)
                    buf = []
                else:
                    print(f"[WARNING] Unexpected data received: {data}")
            else:
                print("[ERROR] Received zero-length data, restarting connection...")
                restart_count += 1
                if restart_count > 5:
                    print("[CRITICAL] Too many restarts, exiting thread.")
                    break
                connection.close()
                connection = HTTPConnection(self.ntc.ntrip_server)
                print("[DEBUG] Reconnecting to NTRIP caster")
                connection.request('GET', '/' + self.ntc.ntrip_stream, self.ntc.nmea_gga, headers)
                response = connection.getresponse()
                print(f"[DEBUG] Reconnection response: {response.status}")
                if response.status != 200:
                    print("[ERROR] Reconnection failed")
                    return
                buf = ""
        print("[DEBUG] Closing NTRIP connection")
        connection.close()



class ntripclient:
    def __init__(self):
        rospy.init_node('ntripclient', anonymous=True)

        self.rtcm_topic = rospy.get_param('~rtcm_topic', 'rtcm')
        self.nmea_topic = rospy.get_param('~nmea_topic', 'nmea')

        self.ntrip_server = rospy.get_param('~ntrip_server')
        self.ntrip_user = rospy.get_param('~ntrip_user')
        self.ntrip_pass = rospy.get_param('~ntrip_pass')
        self.ntrip_stream = rospy.get_param('~ntrip_stream')
        self.nmea_gga = rospy.get_param('~nmea_gga')

        self.pub = rospy.Publisher(self.rtcm_topic, Message, queue_size=10)

        self.connection = None
        self.connection = ntripconnect(self)
        self.connection.start()

    def run(self):
        rospy.spin()
        if self.connection is not None:
            self.connection.stop = True

if __name__ == '__main__':
    c = ntripclient()
    c.run()

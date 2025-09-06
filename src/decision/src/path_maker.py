#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import rospy
import rospkg
from sensor_msgs.msg import NavSatFix
from pyproj import CRS, Transformer


class GPSFixLogger:
    def __init__(self):
        # ── ROS node ───────────────────────────────────────────────────────────
        rospy.init_node('path_maker', anonymous=True)

        # ── WGS‑84 (EPSG:4326) ➜ UTM‑52N (EPSG:32652) transformer ─────────────
        self.transformer = Transformer.from_crs(
            CRS.from_epsg(4326),      # lon, lat
            CRS.from_epsg(32652),     # x, y  (E, N)
            always_xy=True)

        # ── output directory: <decision>/path/ ────────────────────────────────
        rospack = rospkg.RosPack()
        decision_path = os.path.join(rospack.get_path('decision'), 'path')
        os.makedirs(decision_path, exist_ok=True)

        # ── file name: GPS_YYYYMMDD_HHMMSS.txt ────────────────────────────────
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = os.path.join(decision_path, f'GPS_{ts}.txt')
        self.fp = open(self.file_path, 'w')
        rospy.loginfo('[path_maker] Writing to %s', self.file_path)

        # ── internal counter ──────────────────────────────────────────────────
        self.idx = 1

        # ── subscriber ────────────────────────────────────────────────────────
        rospy.Subscriber('/ntrip_rops/ublox_gps/fix', NavSatFix,
                         self.gps_callback, queue_size=5)

        # close file cleanly on shutdown
        rospy.on_shutdown(self._cleanup)
        rospy.spin()

    def gps_callback(self, msg: NavSatFix):
        lon = msg.longitude
        lat = msg.latitude

        # convert to UTM‑52N (Easting, Northing)
        utm_x, utm_y = self.transformer.transform(lon, lat)

        # compose line
        line = f"{self.idx},{lon:.6f},{lat:.6f},{utm_x:.4f},{utm_y:.4f}\n"
        self.fp.write(line)
        self.fp.flush()          # ensure immediate disk write

        rospy.loginfo_throttle(1.0, line.strip())
        self.idx += 1

 
    def _cleanup(self):
        if not self.fp.closed:
            self.fp.close()
        rospy.loginfo('[path_maker] Finished. %d fixes logged.', self.idx-1)


if __name__ == '__main__':
    try:
        GPSFixLogger()
    except rospy.ROSInterruptException:
        pass
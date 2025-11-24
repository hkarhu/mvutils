#!/usr/bin/env python3
from skimage.metrics import structural_similarity
import os
import numpy as np
import cv2
from tqdm import trange
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('limit', nargs='?', type=float, default=0.1, help='blur threshold')
parser.add_argument('source_file', help='source video file')
parser.add_argument('target_path', help='target path')
args = parser.parse_args()

cap = cv2.VideoCapture(args.source_file)
#f = open('log.txt', 'w')

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_span = 0
best_frame = None
last_cmp_frame = None
best_fm = 99999
best_nedg = 0

target_path = args.target_path
target_base = os.path.basename(args.source_file)

try:
    os.mkdir(target_path, 0o755)
except OSError:
    print ("Creation of the directory %s failed" % target_path)
else:
    print ("Successfully created the directory %s" % target_path)

for index in trange(frame_count, unit=' frames', leave=False, dynamic_ncols=True, desc='Calculating blur ratio'):
    frame_span = frame_span + 1
    ret, frame = cap.read()
    if frame is None:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray,100,200)

    cmp_frame = cv2.resize(gray, (512,512))
    #cmp_frame = gray.copy()
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    if index == 0:
        last_cmp_frame = cmp_frame

    best_fm = 0
    if fm > best_fm:
        best_fm = fm
        best_frame = frame

    (score, diff) = structural_similarity(last_cmp_frame, cmp_frame, full=True)

    if score < args.limit or frame_span > 24:
        #print("W@ " + str(best_fm))
        last_fm = best_frame
        last_cmp_frame = cmp_frame
        #cv2.imwrite(args.source_file+f"_{index:05}_"+str(best_fm)+".jpg", best_frame)
        cv2.imwrite(target_path+ "/" + target_base + f"_{index:05}.jpg", best_frame)
        frame_span = 0
        best_fm = 99999
        best_nedg = 0
        im = cv2.resize(best_frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Output", im)

    im = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    cv2.rectangle(im, (0,0), (256,54), 0, -1)
    cv2.putText(im,f"SSIM: {score:05}", (8,15), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
    cv2.putText(im,f"FM: {fm:05}", (8,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
    cv2.putText(im,f"FB: {best_fm:05}", (8,45), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
    cv2.imshow("Processing", im)


    k = cv2.waitKey(1) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

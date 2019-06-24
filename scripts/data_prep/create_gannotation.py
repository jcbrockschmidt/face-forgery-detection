#!/usr/bin/env python3

"""TODO"""

from utils import get_seq_combos

import argparse
import cv2
import dlib
import GANnotation.utils as gann_utils
import importlib.util
import numpy as np
import os
import sys

from sys import stderr

COMPRESSION_LEVEL = 'c0'  # c0, c23, c40

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def compute_video_encoding(video):
    """
    Computes the GANnotation encoding for a video.

    Args:
        video: A cv2.VideoCapture video to encode.

    Return:
        Numpy array encoding.
    """

    video_points = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Find landmarks/points in frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 1)
        if (len(rects) == 0):
            break  # No face found.
        landmarks = face_predictor(gray, rects[0])

        # Convert landmarks to a numpy array.
        points = []
        for i in range(0, landmarks.num_parts):
            if i == 60 or i == 64:
                continue
            point = landmarks.part(i)
            points.append([point.x, point.y])
        points = np.array(points)

        img, maps, pts = gann_utils.process_image(frame, points)
        video_points.append(pts)

    video_points = np.array(video_points).transpose().swapaxes(0, 1)
    return video_points

get_encoding_path = lambda enc_dir, driver_id: '{}/{}.txt'.format(enc_dir, driver_id)

def main(data_dir, gann_dir):
    """
    Create videos with GANnotation using the same driving video and source video combinations
    as with Face2Face.

    Args:
        data_dir: Base directory of the FaceForensics++ dataset.
        gann_dir: Path to local GANnotation repository/installation.
    """

    face2face_dir = '{}/manipulated_sequences/Face2Face/c0/videos'.format(data_dir)
    orig_dir = '{}/original_sequences/c0/videos'.format(data_dir)
    base_dir = '{}/manipulated_sequences/GANnotation'.format(data_dir)
    output_enc_dir = '{}/encodings'.format(base_dir)
    output_vid_dir = '{}/{}/videos'.format(base_dir, COMPRESSION_LEVEL)

    pairs = get_seq_combos(face2face_dir)

    # Compute all video encodings and save them to disk.
    # We precompute these because they take roughly 10 times as long to compute as the
    # reenactments, and we may want to recompute the reenactments with different images later.
    if not os.path.exists(output_enc_dir):
        os.makedirs(output_enc_dir)
    enc_count = 0
    for driver_id, _ in pairs:
        encoding_path = get_encoding_path(output_enc_dir, driver_id)
        if os.path.exists(encoding_path):
            continue  # Encoding already calculated for this video sequence.
        print('Computing encoding for sequence {}...'.format(driver_id))
        video_path = '{}/{}.mp4'.format(orig_dir, driver_id)
        cap = cv2.VideoCapture(video_path)
        points = compute_video_encoding(cap)
        cap.release()
        try:
            np.savetxt(encoding_path, points.reshape((132,-1)).transpose())
        except KeyboardInterrupt as e:
            # Safely handle premature termination.
            # Remove unfinished file.
            if os.exists(encoding_path):
                os.remove(encoding_path)
            raise e
        enc_count += 1

    print()
    if enc_count == 0:
        print('No encodings were calculated')
    else:
        print('{} video sequences encoded'.format(enc_count))

    print()
    print('Computing reenactments...')
    image_dir = '{}/original_sequences_images/{}/images'.format(data_dir, COMPRESSION_LEVEL)
    if not os.path.exists(output_vid_dir):
        os.makedirs(output_vid_dir)
    reenact_count = 0
    for driver_id, source_id in pairs:
        output_path = '{}/{}_{}.mp4'.format(output_vid_dir, driver_id, source_id)
        if os.path.exists(output_path):
            # Do not recreate a video if it already exists.
            # If the user wants to recreated a video
            # the existing video must be deleted first.
            continue

        print('Computing reenactment for {} onto {}...'.format(driver_id, source_id))
        # Validate that input files exist.
        encoding_path = get_encoding_path(output_enc_dir, driver_id)
        if not os.path.isfile(encoding_path):
            print('Failed to find encoding for video sequence {}'.format(driver_id),
                  file=stderr)
            continue
        image_path = '{}/{}.png'.format(image_dir, source_id)
        if not os.path.isfile(image_path):
            print('Failed to find image for sequence {}'.format(source_id),
                  file=stderr)
            continue

        # TODO: Compute reenactments.

    print()
    if reenact_count == 0:
        print('No reenactments were created')
    else:
        print('{} reenactments created'.format(reenact_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracts faces from each original videos')
    parser.add_argument('data_dir', type=str, nargs=1,
                        help='Base directory for FaceForensics++ data')
    parser.add_argument('gann_dir', metavar='GANnotation_dir',
                        type=str, nargs=1,
                        help='Base directory for GANnotation')
    args = parser.parse_args()

    # Validate arguments.
    data_dir = args.data_dir[0]
    gann_dir = args.gann_dir[0]
    if not os.path.isdir(data_dir):
        print('"{}" is not a directory'.format(data_dir), file=stderr)
        exit(2)
    if not os.path.isdir(gann_dir):
        print('"{}" is not a directory'.format(gann_dir), file=stderr)
        exit(2)

    main(data_dir, gann_dir)

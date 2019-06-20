#!/usr/bin/env python3

"""
Extracts a single face from each original downloaded video in the
FaceForensics++ dataset.
"""

import argparse
import cv2
import face_recognition
import json
import os

from glob import glob
from sys import exit, stderr

COMPRESSION_LEVEL = 'c0'  # c0, c23, c40
ZOOMOUT_FACTOR = 1.8  # [1, ...]
ZOOMOUT_Y_BIAS = 0.7  # [0, 1]
CROP_SIZE = 256

class FFVideoSeq:
    """Represents a video sequence.

    Attributes:
        seq_id: String identifier for the video.
        path: Path to video file.
    """

    def __init__(self, seq_id, path):
        self.seq_id = seq_id
        self.path = path

    def __lt__(self, other):
        return self.seq_id < other.seq_id

face_size = lambda tp, rt, bt, lt: (bt - tp) * (rt - lt)

def crop_face(img, location, scale_size=256, zoomout=ZOOMOUT_FACTOR):
    """Crops a square area around a face.

    Args:
        img: Image to crop.
        location: Location of the face, as returned by
            face_recognition.face_locations().
        scale_size: Width and height of cropped image.  Scaling will occur.
        zoomout: Percentage of the cropped area to include.
            Can (and probably should) be greater than 1.
    Returns:
        Cropped image containing the face.
    """
    top, rt, bot, lt = location

    # Make region square.  Choose largest side as width and height.
    size = max(rt - lt, bot - top)

    # Expand by zoomout factor.
    zoomout_size = int(size * zoomout)

    # Calculate new crop boundaries.
    # Constrain new boundaries within image boundaries.
    img_h, img_w, _ = img.shape
    center_x = int((lt + rt) / 2)
    center_y = int((top + bot) / 2)
    shift = int(zoomout_size / 2)
    size_diff = zoomout_size - size
    y1shift = int(size / 2 + size_diff * ZOOMOUT_Y_BIAS)
    y2shift = int(size / 2 + size_diff * (1 - ZOOMOUT_Y_BIAS))
    x1 = max(center_x - shift, 0)
    x2 = min(center_x + shift, img_w)
    y1 = max(center_y - y1shift, 0)
    y2 = min(center_y + y2shift, img_h)

    # Readjust boundaries to be square again if boundary correction occurred.
    new_size = min(x2 - x1, y2 - y1)
    if new_size < zoomout_size:
        print("collision")
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        shift = int(new_size / 2)
        # No boundary correction is necessary since we are shrinking.
        x1 = center_x - shift
        x2 = center_x + shift
        y1 = center_y - shift
        y2 = center_y + shift

    # Crop and scale image.
    cropped = img[y1:y2, x1:x2]
    scaled = cv2.resize(cropped, (scale_size, scale_size))

    return scaled

def extract_face(seq):
    """Extracts a single face image from a video sequence.

    Arguments:
        seq: FFVideoSeq representing video sequence.
        output_fn: Path to write image to.

    Returns:
        A 256x256 image of a face on success. None on failure.
    """
    # Open video for reading.
    cap = cv2.VideoCapture(seq.path)

    # Keep looking for a face until we find one or reach the last frame.
    while True:
        # Attempt to find face in the next frame of the video.
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert image from BGR to RGB.
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        if len(face_locations) == 0:
            # No face found.  Continue to next frame.
            continue
        else:
            # We found a face.
            # If there are multiple faces, choose the largest.
            largest_i = 0
            largest_size = face_size(*face_locations[0])
            for i in range(1, len(face_locations)):
                cur_size = face_size(*face_locations[i])
                if cur_size > largest_size:
                    largest_i = i
                    largest_size = cur_size

            cropped = crop_face(frame, face_locations[largest_i], CROP_SIZE)

            return cropped

def get_orig_sequences(data_dir, comp='c0'):
    """
    Gets every original video sequence from a given compression level.

    Args:
        data_dir: Base directory of the FaceForensics++ dataset.
        comp: Compression level ('c0', 'c23', 'c40')
    Returns:
        List of sequences in ascending order by ID.
    """
    # TODO: Check compression type and raise error if it is invalid.

    # Get all video file paths.
    seq_dir = '{}/original_sequences/{}/videos'.format(data_dir, comp)
    paths = glob('{}/*.mp4'.format(seq_dir))

    # Create sequence objects for each video file.
    seqs = []
    for p in paths:
        abs_path = os.path.abspath(p)
        seq_id, _ = os.path.splitext(os.path.basename(abs_path))
        seq = FFVideoSeq(seq_id, abs_path)
        seqs.append(seq)
    seqs.sort()
    return seqs

def main(data_dir):
    """Extracts faces from FaceForensics++ dataset.

    Args:
        data_dir: Base directory of the FaceForensics++ dataset.
    Returns:
        The number of faces extracted and written to disk.
    """

    extract_count = 0

    try:
        # Validate argument.  Exit if invalid.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(1)

        # Create directory for output images, if it does not already exist.
        output_dir = '{}/original_sequences_faces/{}/images'.format(
            data_dir, COMPRESSION_LEVEL)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Extracting faces...")
        seqs = get_orig_sequences(data_dir, COMPRESSION_LEVEL)
        for s in seqs:
            output_fn = '{}/{}.png'.format(output_dir, s.seq_id)
            if os.path.exists(output_fn):
                # Do not recreate an image if it already exists.
                # If the user wants to recreated an image,
                # the old image must be deleted first.
                continue

            print('Extracting face for sequence {}...'.format(s.seq_id))
            face_img = extract_face(s)

            # Write face image to disk.
            try:
                cv2.imwrite(output_fn, face_img)
                extract_count += 1
            except KeyboardInterrupt as e:
                # Safely handle premature termination.  Remove unfinished file.
                if os.exists(output_fn):
                    os.remove(output_fn)
                raise e
    except KeyboardInterrupt:
        print('Program terminated prematurely')
    finally:
        if extract_count == 0:
            print('No faces extracted')
        else:
            print('{} faces extracted'.format(extract_count))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Extracts faces from each original videos')
        parser.add_argument('data_dir', metavar='dataset_dir',
                            type=str, nargs=1,
                            help='Base directory for FaceForensics++ data')
        args = parser.parse_args()

        main(args.data_dir[0])

    except KeyboardInterrupt:
        pass

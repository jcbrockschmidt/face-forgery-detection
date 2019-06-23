#!/usr/bin/env python3

"""
Extracts a single face from each original downloaded video in the
FaceForensics++ dataset.
"""

from utils import crop_face, get_largest_face, FFVideoSeq

import argparse
import cv2
import face_recognition
import os

from glob import glob
from sys import exit, stderr

COMPRESSION_LEVEL = 'c0'  # c0, c23, c40
ZOOMOUT_FACTOR = 1.6  # [1, ...]
CROP_SIZE = 256

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
            loc = get_largest_face(face_locations)
            cropped = crop_face(frame, loc, CROP_SIZE, ZOOMOUT_FACTOR)

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

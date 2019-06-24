#!/usr/bin/env python3

"""
Extracts a single face from each original downloaded video in the
FaceForensics++ dataset and adds them to the dataset.
"""

from utils import crop_face, extract_image, get_largest_face, \
    get_orig_sequences, FFVideoSeq

import argparse
import cv2
import face_recognition
import os

from glob import glob
from sys import stderr

COMPRESSION_LEVEL = 'c0'  # c0, c23, c40
ZOOMOUT_FACTOR = 1.6  # [1, ...]
CROP_SIZE = 256

def extract_face(seq):
    """
    Extracts a single face image from a video sequence.

    Args:
        seq: FFVideoSeq representing video sequence.
        output_fn: Path to write image to.

    Returns:
        A 256x256 image of a face on success. None on failure.
    """
    img, locations = extract_image(seq)
    if img is None:
        # No frame with a face was found.
        return None
    else:
        # We found a frame with a face.
        # If there are multiple faces, choose the largest.
        loc = get_largest_face(locations)
        cropped = crop_face(img, loc, CROP_SIZE, ZOOMOUT_FACTOR)
        return cropped

def main(data_dir):
    """
    Extracts faces from FaceForensics++ dataset.

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
            if face_img is None:
                print("    No face found")
            else:
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

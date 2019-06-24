#!/usr/bin/env python3

"""
Extracts a single image/frame from each original video
in the FaceForensics++ dataset and adds them to the dataset.
"""

from utils import extract_image, get_orig_sequences, FFVideoSeq

import argparse
import cv2
import os

from sys import stderr

COMPRESSION_LEVEL = 'c0'  # c0, c23, c40

def main(data_dir):
    """
    Extracts faces from FaceForensics++ dataset.

    Args:
        data_dir: Base directory of the FaceForensics++ dataset.

    Returns:
        The number of images extracted and written to disk.
    """

    extract_count = 0

    try:
        # Validate argument.  Exit if invalid.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(1)

        # Create directory for output images, if it does not already exist.
        output_dir = '{}/original_sequences_images/{}/images'.format(
            data_dir, COMPRESSION_LEVEL)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Extracting images...")
        seqs = get_orig_sequences(data_dir, COMPRESSION_LEVEL)
        for s in seqs:
            output_fn = '{}/{}.png'.format(output_dir, s.seq_id)
            if os.path.exists(output_fn):
                # Do not recreate an image if it already exists.
                # If the user wants to recreated an image,
                # the old image must be deleted first.
                continue

            print('Extracting image for sequence {}...'.format(s.seq_id))
            img, _ = extract_image(s)

            # Write image to disk.
            try:
                cv2.imwrite(output_fn, img)
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
            print('No images extracted')
        else:
            print('{} images extracted'.format(extract_count))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Extracts a single image from each original video')
        parser.add_argument('data_dir', metavar='dataset_dir',
                            type=str, nargs=1,
                            help='Base directory for FaceForensics++ data')
        args = parser.parse_args()

        main(args.data_dir[0])

    except KeyboardInterrupt:
        pass

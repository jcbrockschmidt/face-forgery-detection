#!/usr/bin/env python3

"""Generates facial landmark encodings with OpenFace for use by ICface."""

import argparse
import cv2
import os
import shutil
import subprocess
from sys import stderr
from time import sleep

from utils import get_orig_sequences

# Compression level of original video sequences.
COMPRESSION_LEVEL = 'c0'  # 'c0', 'c23', or 'c40'

def compute_openface_encoding(openface_bin, video_path, output_path):
    """
    Compute a facial landmark encoding with OpenFace and write it to disk.

    Args:
        openface_bin: Path to bin for OpenFace executables.
        video_path: Path of video to encode.
        output_path: Path to save CSV encoding to.

    Returns:
        True if encoding was successfully created and saved.
        False if something went wrong.
    """

    # Use OpenFace's facial landmark extraction executable.
    exe = os.path.join(openface_bin, 'FeatureExtraction')
    cmd = [exe, '-q', '-f', video_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait until process terminates.
    while proc.poll() is None:
        sleep(0.5)

    name = os.path.splitext(os.path.split(video_path)[-1])[0]
    csv_enc_path = os.path.join('processed', '{}.csv'.format(name))
    if proc.returncode != 0 or not os.path.isfile(csv_enc_path):
        # Something went wrong.  CSV facial landmark encoding was not created.
        return False
    os.rename(csv_enc_path, output_path)
    return True

def cleanup_openface():
    """
    Clean up files generated by OpenFace.
    """
    # Remove directory for processed data.
    if os.path.exists('processed'):
        shutil.rmtree('processed')
    
def main(data_dir, openface_bin):
    """
    Generates facial landmark encodings with OpenFace for every original video
    sequence using in the FaceForensics++ dataset.

    Args:
        data_dir: Directory for FaceForensics++ dataset.
        openface_bin: Path to bin for OpenFace executables.
    """

    print('Computing video encodings...')

    # Base directory for ICface data.
    icface_dir = os.path.join(data_dir, 'manipulated_sequences/ICface')

    # Directory to store video encodings in.
    enc_dir = os.path.join(icface_dir, 'encodings')
    if not os.path.exists(enc_dir):
        os.makedirs(enc_dir)

    sequences = get_orig_sequences(data_dir)
    enc_count = 0
    for seq in sequences:
        output_path = os.path.join(enc_dir, '{}.csv'.format(seq.seq_id))
        if os.path.exists(output_path):
            # Do not recreate an encoding if it already exists.
            # If the user wants to recreated an encoding
            # the existing encoding must be deleted first.
            continue

        print('Computing encoding for sequence {}...'.format(seq.seq_id))
        ret = compute_openface_encoding(openface_bin, seq.path, output_path)
        if not ret:
            print('ERROR: Something went wrong.  ' \
                  'Encoding not computed for sequence {}'.format(seq.seq_id),
                  file=stderr)
        else:
            enc_count += 1

    if enc_count == 0:
        print('No encodings were calculated')
    else:
        print('{} video sequences encoded'.format(enc_count))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Generates video encodings for ICface')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='Directory for FaceForensics++ dataset')
        parser.add_argument('openface_bin', type=str, nargs=1,
                            help='Bin directory for OpenFace')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        openface_bin = args.openface_bin[0]
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)

        main(data_dir, openface_bin)
    except KeyboardInterrupt:
        print('Program terminated prematurely')
    finally:
        cleanup_openface()

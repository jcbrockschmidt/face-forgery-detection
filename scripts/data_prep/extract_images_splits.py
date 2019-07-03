#!/usr/bin/env python3

"""
Creates images for training, testing, and validation.

The resulting file structure will be:
    +---data
        +---train
        |   +---c0
        |   |   +---df
        |   |   +---f2f
        |   |   +---gann
        |   |   +---icf
        |   |   +---x2f
        |   +---c23
        |   |   +--- ...
        |   +---c40
        |   |   +--- ...
        +---test
        |   +--- ...
        +---val
            +--- ...

where `data` is the output directory.
"""

import argparse
import cv2
import face_recognition
from multiprocessing.pool import Pool
import os
from sys import stderr

from utils import crop_face
from split_utils import get_mani_paths, get_orig_paths, get_splits

# Frames between each image capture.
ELAPSE = 30
ZOOMOUT = 1.6

def extract_images_worker(video_path, output_dir, overwrite=False):
    """
    Extracts an image from every 30th frame in a video, starting on the 30th.

    Args:
        video_path: Path to video.
        output_dir: Directory to output images to.
        overwrite: Whether to overwrite images already in the output directory.

    Returns:
        Number of images written to disk.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        name, _ = os.path.splitext(os.path.basename(video_path))
        img_count = 0
        for i in range(0, int(frame_cnt // ELAPSE) - 1):
            output_path = os.path.join(output_dir, '{}-{}.png'.format(name, i))
            if not overwrite and os.path.exists(output_path):
                # Image already exists and will not be overwritten.
                continue

            # Jump to next frame.
            fi = (i + 1) * ELAPSE
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            # Crop image around a face.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_recognition.face_locations(gray, model='hog')
            if len(face_locations) == 0:
                # No face found.  Continue to next frame.
                continue
            loc = face_locations[0]
            cropped = crop_face(frame, loc, zoomout=ZOOMOUT)

            # Write image to disk.
            try:
                if not cv2.imwrite(output_path, cropped):
                    print('ERROR: Failed to write to "{}"'.format(output_path),
                          file=stderr)
            except Exception as e:
                if os.path.exists(output_path):
                    # Delete partially written file.
                    os.remove(output_path)
                raise e
            img_count += 1

        cap.release()
        return img_count

    except KeyboardInterrupt:
        pass

def extract_images(video_paths, output_dir, overwrite=False):
    """
    Extracts images from all videos in a split.  A frame is captured every
    30 frames and cropped around a face, starting on the 30th frame.

    Args:
        video_paths: A list of paths to videos.
        output_dir: Directory to write images to.
        overwrite: Whether to overwrite images already in the output directory.

    Returns:
        Number of images written to disk.
    """
    pool = Pool()
    results = []
    for path in video_paths:
        if not os.path.exists(path):
            continue

        res = pool.apply_async(extract_images_worker, (path, output_dir, overwrite))
        results.append(res)

    pool.close()
    pool.join()

    img_count = sum([res.get() for res in results])

    return img_count

def main(data_dir, output_dir, overwrite=False):
    """
    Extracts frames from each video.

    Args:
        data_dir: Directory for FaceForensics++ dataset.
        output_dir: Directory to put extracted images in.
        overwrite: Whether to overwrite images already in the output directory.
    """

    split_dir = os.path.join(os.path.dirname(__file__), 'splits')
    train_split, test_split, val_split = get_splits(split_dir)

    orig_dir = os.path.join(data_dir, 'original_sequences')
    df_dir = os.path.join(data_dir, 'manipulated_sequences/Deepfakes')
    f2f_dir = os.path.join(data_dir, 'manipulated_sequences/Face2Face')
    gann_dir = os.path.join(data_dir, 'manipulated_sequences/GANnotation')
    icf_dir = os.path.join(data_dir, 'manipulated_sequences/ICface')
    x2f_dir = os.path.join(data_dir, 'manipulated_sequences/X2Face')

    # Extract images for training, testing, and validation.
    splits = (('train', train_split), ('test', test_split), ('val', val_split))
    img_count = 0
    for split_name, split in splits:
        print('\nExtracting images for {}...'.format(split_name))

        # Extract frames from every compression level.
        for comp in ('c0', 'c23', 'c40'):
            print('Extracting images for compression level {}...'.format(comp))
            base_dir = os.path.join(output_dir, comp, split_name)

            orig_out_dir = os.path.join(base_dir, 'real')
            df_out_dir = os.path.join(base_dir, 'df')
            f2f_out_dir = os.path.join(base_dir, 'f2f')
            gann_out_dir = os.path.join(base_dir, 'gann')
            icf_out_dir = os.path.join(base_dir, 'icf')
            x2f_out_dir = os.path.join(base_dir, 'x2f')
            dirs = (orig_out_dir, df_out_dir, f2f_out_dir,
                    gann_out_dir, icf_out_dir, x2f_out_dir)
            for direct in dirs:
                if not os.path.exists(direct):
                    os.makedirs(direct)

            if os.path.isdir(orig_dir):
                print('    Original videos...')
                orig_paths = get_orig_paths(orig_dir, split, comp)
                img_count += extract_images(
                    orig_paths, orig_out_dir, overwrite=overwrite)

            if os.path.isdir(df_dir):
                print('    Deepfakes...')
                df_paths = get_mani_paths(df_dir, split, comp)
                img_count += extract_images(
                    df_paths, df_out_dir, overwrite=overwrite)

            if os.path.isdir(f2f_dir):
                print('    Face2Face...')
                f2f_paths = get_mani_paths(f2f_dir, split, comp)
                img_count += extract_images(
                    f2f_paths, f2f_out_dir, overwrite=overwrite)

            if os.path.isdir(gann_dir):
                print('    GANnotation...')
                gann_paths = get_mani_paths(gann_dir, split, comp)
                img_count += extract_images(
                    gann_paths, gann_out_dir, overwrite=overwrite)

            if os.path.isdir(icf_dir):
                print('    ICface...')
                icf_paths = get_mani_paths(icf_dir, split, comp)
                img_count += extract_images(
                    icf_paths, icf_out_dir, overwrite=overwrite)

            if os.path.isdir(x2f_dir):
                print('    X2Face...')
                x2f_paths = get_mani_paths(x2f_dir, split, comp)
                img_count += extract_images(
                    x2f_paths, x2f_out_dir, overwrite=overwrite)

    print('\n{} images written to disk'.format(img_count))
    print('Done')

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Creates images for training, testing, and validation')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory for FaceForensics++ dataset')
        parser.add_argument('output_dir', type=str, nargs=1,
                            help='directory to put extracted images in')
        parser.add_argument('-f', '--overwrite',
                            action='store_const', const=True, default=False,
                            help='overwrite existing videos in the output directory')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        output_dir = args.output_dir[0]
        overwrite = args.overwrite

        main(data_dir, output_dir, overwrite=overwrite)
    except KeyboardInterrupt:
        print('Program terminated prematurely')

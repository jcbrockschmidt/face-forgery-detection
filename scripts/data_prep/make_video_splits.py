#!/usr/bin/env python3

"""
Seperate videos in the FaceForensics++ dataset into training, testing, and
validation sets.  Videos are linked symbolically.  No processing or copying
is done.

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
import os
import subprocess
from sys import stderr

from split_utils import get_mani_paths, get_orig_paths, get_splits

def link_videos(video_paths, output_dir):
    """
    Puts symbolic links a list of videos in an directory.

    Args:
        video_paths: A list of paths to videos.
        output_dir: Directory to put symbolic links in.

    Returns:
        Number of links created.
    """
    link_count = 0
    for path in video_paths:
        name = os.path.basename(path)
        linkto = os.path.abspath(path)
        if not os.path.exists(linkto):
            print('ERROR: Missing {}, no link will be made'.format(name),
                  file=stderr)
        link = os.path.join(output_dir, name)
        cmd = ['ln', '-sf', linkto, link]
        exit_code = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=stderr)
        if exit_code != 0:
            print('ERROR: Failed to create symbolic link for {}'.format(linkto),
                  file=stderr)
            continue
        link_count += 1
    return link_count

def main(data_dir, output_dir):
    """
    Makes symbolic links for all videos for all splits.

    Args:
        data_dir: Directory for FaceForensics++ dataset.
        output_dir: Directory to put extracted images in.
    """
    
    split_dir = os.path.join(os.path.dirname(__file__), 'splits')
    train_split, test_split, val_split = get_splits(split_dir)

    orig_dir = os.path.join(data_dir, 'original_sequences')
    df_dir = os.path.join(data_dir, 'manipulated_sequences/Deepfakes')
    f2f_dir = os.path.join(data_dir, 'manipulated_sequences/Face2Face')
    gann_dir = os.path.join(data_dir, 'manipulated_sequences/GANnotation')
    icf_dir = os.path.join(data_dir, 'manipulated_sequences/ICface')
    x2f_dir = os.path.join(data_dir, 'manipulated_sequences/X2Face')

    # Link images for training, testing, and validation.
    splits = (('train', train_split), ('test', test_split), ('val', val_split))
    link_count = 0
    for split_name, split in splits:
        print('\nLinking videos for {}...'.format(split_name))

        # Extract frames from every compression level.
        for comp in ('c0', 'c23', 'c40'):
            print('Linking videos for compression level {}...'.format(comp))
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
                link_count += link_videos(orig_paths, orig_out_dir)

            if os.path.isdir(df_dir):
                print('    Deepfakes...')
                df_paths = get_mani_paths(df_dir, split, comp)
                link_count += link_videos(df_paths, df_out_dir)

            if os.path.isdir(f2f_dir):
                print('    Face2Face...')
                f2f_paths = get_mani_paths(f2f_dir, split, comp)
                link_count += link_videos(f2f_paths, f2f_out_dir)

            if os.path.isdir(gann_dir):
                print('    GANnotation...')
                gann_paths = get_mani_paths(gann_dir, split, comp)
                link_count += link_videos(gann_paths, gann_out_dir)

            if os.path.isdir(icf_dir):
                print('    ICface...')
                icf_paths = get_mani_paths(icf_dir, split, comp)
                link_count += link_videos(icf_paths, icf_out_dir)

            if os.path.isdir(x2f_dir):
                print('    X2Face...')
                x2f_paths = get_mani_paths(x2f_dir, split, comp)
                link_count += link_videos(x2f_paths, x2f_out_dir)

    print('\n{} videos linked'.format(link_count))
    print('Done')

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Creates images for training, testing, and validation')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory for FaceForensics++ dataset')
        parser.add_argument('output_dir', type=str, nargs=1,
                            help='directory to copy video to')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        output_dir = args.output_dir[0]

        main(data_dir, output_dir)
    except KeyboardInterrupt:
        print('Program terminated prematurely')

#!/usr/bin/env python3

"""
Creates new, compressed copies of MP4 videos using the H.264 codec.
New videos will be placed in a directory with the names of each video
corresponding to their original.

Make sure you have ffmpeg installed before running this command.
"""

import argparse
from glob import glob
from multiprocessing.pool import Pool
import os
import subprocess
from sys import stderr

def get_videos(orig_dir, output_dir, overwrite=False):
    """
    Get all MP4 videos in a directory and their output paths.

    Args:
        orig_dir: Path to directory with original videos.
        output_dir: Directory that will contain 

    Returns:
        List of tuples of original video files and their output paths
        in the form (orig_path, output_path).
    """
    orig_paths = glob('{}/*.mp4'.format(orig_dir))
    orig_paths.sort()

    output_paths = []
    for path in orig_paths:
        name = os.path.basename(path)
        output = os.path.join(output_dir, name)
        if overwrite or not os.path.exists(output):
            output_paths.append((path, output))

    return output_paths

def compress_worker(video_path, output_path, crf):
    """
    Compresses an MP4 with the H.264 codec.

    Args:
        video_path: Path to video to compress.
        output_path: Path to write video to.
        crf: Constant rate factor for compression.
    """
    name = os.path.basename(video_path)
    print('Compressing {}...'.format(name))
    cmd = 'ffmpeg -y -i {} -c:v libx264 -crf {} {}'.format(video_path, crf, output_path)
    exit_code = subprocess.call(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exit_code != 0:
        print('ERROR: Something went wrong when compressing {}'.format(name),
              file=stderr)
    print('Finished compressing {}'.format(name))

def main(original_dir, output_dir, crf, thread_count=None, overwrite=False):
    """
    Compresses every MP4 video in a directory using the H.264 codec.

    Args:
        original_dir: Directory containing original videos.
        output_dir: Directory to output new videos to.  Will be created if
            it does not exist.
        crf: Constant rate factor, i.e. the amount of compressing to perform.
            A higher number means higher compression.
        thread_count: Amount of threads to use.  Set to None automatically
            choose a number of threads.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    videos = get_videos(original_dir, output_dir, overwrite)

    print('Compressing {} videos...'.format(len(videos)))
    pool = Pool(thread_count)
    for video, output in videos:
        pool.apply_async(compress_worker, (video, output, crf))

    pool.close()
    pool.join()

    print('Done')

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Generates videos with ICFace')
        parser.add_argument('original_videos', type=str, nargs=1,
                            help='directory containing original videos')
        parser.add_argument('output_videos', type=str, nargs=1,
                            help='directory to output new videos to')
        parser.add_argument('crf', type=int, nargs=1,
                            help='constant rate parametr, higher is more compressed')
        parser.add_argument('-t', '--threads', type=int, required=False, nargs=1,
                            help='number of threads to use.')
        parser.add_argument('-f', '--overwrite',
                            action='store_const', const=True, default=False,
                            help='overwrite existing videos in the output directory')
        args = parser.parse_args()

        original_videos = args.original_videos[0]
        if not os.path.isdir(original_videos):
            print('"{}" is not a directory'.format(original_videos), file=stderr)
            exit(2)
        output_videos = args.output_videos[0]
        crf = args.crf[0]
        if args.threads is None:
            threads = None
        else:
            threads = args.threads[0]
        overwrite = args.overwrite

        main(original_videos, output_videos, crf,
             thread_count=threads, overwrite=overwrite)
    except KeyboardInterrupt:
        print('Program terminated')

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
from sys import exit, stderr
from time import sleep  # DEBUG

ZOOMOUT_FACTOR = 1.7
CROP_SIZE = 256

class FFVideoSeqID:
    """Identifying information for an unprocessed video sequence."""

    def __init__(self, seq_id, frames_id, vid_id, data_dir):
        self.seq_id = seq_id
        self.frames_id = frames_id
        self.vid_id = vid_id
        self.orig_base = os.path.abspath('{}/downloaded_videos/{}'.format(
            data_dir, vid_id))

    def get_frames(self):
        """
        Get the range of frames this video sequence takes from the
        original video.

        Returns:
            [first_frame, last_frame], both inclusive.
        """
        # Read JSON file containing frame information.
        frame_fn = '{}/extracted_sequences/{}.json'.format(
            self.orig_base, self.frames_id)
        with open(frame_fn, 'r') as f:
            data = json.load(f)
            first_frame = data['first frame']
            last_frame = data['last frame']
            # TODO: Check for `None`s.

        return [first_frame, last_frame]

    def get_video_path(self):
        """Gets the path for the video associated with this sequence."""
        return '{}/{}.mp4'.format(self.orig_base, self.vid_id)

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
    size = int(size * zoomout)

    # Calculate new crop boundaries.
    # Constrain new boundaries within image boundaries.
    img_h, img_w, _ = img.shape
    center_x = int((lt + rt) / 2)
    center_y = int((top + bot) / 2)
    shift = int(size / 2)
    x1 = max(center_x - shift, 0)
    x2 = min(center_x + shift, img_w)
    y1 = max(center_y - shift, 0)
    y2 = min(center_y + shift, img_h)

    # Readjust boundaries to be square again if boundary correction occurred.
    new_size = min(x2 - x1, y2 - y1)
    if new_size < size:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        shift = int(new_size / 2)
        # No boundary correction is necessary since we are shrinking.
        x1 = max(center_x - shift, 0)
        x2 = min(center_x + shift, img_w)
        y1 = max(center_y - shift, 0)
        y2 = min(center_y + shift, img_h)

    # Crop and scale image.
    cropped = img[y1:y2, x1:x2]
    scaled = cv2.resize(cropped, (scale_size, scale_size))

    return scaled

def extract_face(seq):
    """Extracts a single face image from a video sequence.

    Arguments:
        seq: FFVideoSeqID representing video sequence.
        output_fn: Path to write image to.

    Returns:
        A 256x256 image of a face on success. None on failure.
    """
    first_frame, last_frame = seq.get_frames()

    # Open video for reading.
    video_path = seq.get_video_path()
    cap = cv2.VideoCapture(video_path)

    # Jump to first frame in our range.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    # Keep looking for a face within the frame range until we find one or
    # reach the last frame.
    for i in range(first_frame, last_frame + 1):
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

            return frame

def read_conv_file(conv_fn, data_dir):
    """Reads the contents of the conversion file.

    Args:
        conv_fn: Path to conversion JSON file.
        data_dir: Base directory of FaceForensics++ dataset.
    Returns:
        List of `FFVideoSeqID`s in ascending ordered of ID number.
    """
    seqs = []
    with open(conv_fn, 'r') as f:
        data = json.load(f)
        for seq_id, seq_info in data.items():
            seq_id = int(seq_id)
            tokens = seq_info.split(' ')
            vid_id = tokens[0]
            frames_id = tokens[1]
            seqs.append(FFVideoSeqID(seq_id, frames_id, vid_id, data_dir))
    return seqs

def main(data_dir, conv_fn):
    """Extracts faces from FaceForensics++ dataset.

    Args:
        data_dir: Base directory of FaceForensics++ dataset.
        conv_fn: Location of conversion JSON file.
    Returns:
       The number of faces extracted and written to disk.
    """

    extract_count = 0

    try:
        # Validate arguments. Exit if invalid.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(1)
        if not os.path.isfile(conv_fn):
            print('"{}" is not a file'.format(conv_fn), file=stderr)
            exit(1)

        # Create directory for output images, if it does not already exist.
        output_dir = '{}/original_sequences_faces/images'.format(data_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Extracting faces...")
        seqs = read_conv_file(conv_fn, data_dir)
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
            description='Extracts faces from each original downloaded video')
        parser.add_argument('data_dir', metavar='dataset_dir',
                            type=str, nargs=1,
                            help='Base directory for FaceForensics++ data')
        parser.add_argument('conv_fn', metavar='conversion_file',
                            type=str, nargs=1,
                            help='JSON conversion dictionary')
        args = parser.parse_args()

        main(args.data_dir[0], args.conv_fn[0])

    except KeyboardInterrupt:
        print('Program terminated prematurely')

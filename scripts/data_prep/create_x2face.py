#!/usr/bin/env python3

"""Generates videos with X2Face for the FaceForensics++ dataset."""

import argparse
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Compose, Scale
from torch.autograd import Variable
from sys import stderr
from X2Face.UnwrapMosaic.UnwrappedFace import UnwrappedFaceWeightedAverage,\
    UnwrappedFaceWeightedAveragePose

from utils import crop_face, get_seq_combos, rect_from_landmarks, write_video

ZOOMOUT_FACTOR = 1.6
COMPRESSION_LEVEL = 'c0'

# Number of frames process at a time.
# If you are running out of CUDA memory, try decreasing this number.
BATCH_SIZE = 10

def load_x2face_model():
    dirname = os.path.dirname(__file__)
    state_dict = torch.load(os.path.join(dirname, 'models/x2face_model.pth'),
                            encoding='latin1')
    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
    model.load_state_dict(state_dict['state_dict'])
    model = model.cuda()
    model = model.eval()
    return model

def cv2_img_to_x2face_img(image):
    """
    Converts an image in for format used by OpenCV 2 to the format
    used by X2Face.

    Args:
        image: BGR image as a numpy.ndarray.

    Return:
        Image as an torch.autograd.Variable.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TODO: Find a way to bypass this OpenCV 2 to PIL image conversion.
    pil_img = Image.fromarray(rgb)
    transform = Compose([Scale((256, 256)), ToTensor()])
    return Variable(transform(pil_img))

def cv2_vid_to_x2face_input(video, frame, frame_cnt, crop=False):
    """
    Converts a OpenCV 2 video to driving input for X2Face.

    Args:
        video: Video as a cv2.VideoCapture to convert.
        frame: Frame to start on.  Frame indexing starts at 0.
        frame_cnt: Number of frames to read.  Will read fewer frames if there
            are not enough.
        crop: Whether to crop each frame around a face.

    Returns:
        X2Face driving input as a torch.autograd.Variable.
    """
    # Start reading from the specified frame.
    video.set(frame, cv2.CAP_PROP_POS_FRAMES)

    # Transform each frame and concatenate them together.
    new_driver = None
    for i in range(0, frame_cnt):
        ret, frame = video.read()
        if not ret:
            break

        cur_location = None
        if crop:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_landmarks = face_recognition.face_landmarks(gray)
            if len(face_landmarks) == 0:
                if cur_location is None:
                    continue
                # Will use last face location if no face was found
                # in the current frame.
            else:
                cur_location = rect_from_landmarks(face_landmarks[0])
            cropped_frame = crop_face(frame, cur_location, ZOOMOUT_FACTOR)

        processed_img = cv2_img_to_x2face_img(cropped_frame).unsqueeze(0)
        if new_driver is None:
            new_driver = processed_img
        else:
            new_driver = torch.cat((new_driver, processed_img), 0)

    return new_driver

def run_batch(model, source_imgs, driver_imgs):
    """
    Reenacts the face of a set of driver images onto a set of source images.

    Args:
        model: X2Face model used to process the reenactment.
        source_img: List of 4D torch.Tensors of images with the dimensions
            (frame, rgb, width, height).
        driver_img: 4D torch.Tensor of images with the dimensions
            (frame, rgb, width, height).

    Returns:
        Frames of the generated video as a torch.Tensor with the dimensions
        (frame, rgb, width, height).
    """
    return model(driver_imgs, source_imgs[0])

def reenact(model, driver_vid, source_img):
    """
    Reenacts a face in a driver video on a source image.

    Args:
        model: X2Face model that will generate reenactment.
        driver_vid: Video as a cv2.VideoCapture to reenact.
        source_img: Source image as a numpy.ndarray to reenact onto.

    Returns:
        A list of video frames as numpy.ndarrays.
    """
    total_frames = int(driver_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process frames in batches of BATCH_SIZE.
    result_frames = []
    for i in range(0, total_frames, BATCH_SIZE):
        # Transform our input.
        driver_vid_trans = cv2_vid_to_x2face_input(
            driver_vid, i, BATCH_SIZE, crop=True)
        if driver_vid_trans is None:
            # No faces found in this batch of frames.
            # Stop generating to avoid a discontinuous video.
            print('Lost face.  ' \
                  'Stopping reenactment early on frame {}'.format(i))
            break
        driver_vid_trans = driver_vid_trans.cuda()
        num_frames = driver_vid_trans.shape[0]
        source_img_trans = cv2_img_to_x2face_img(source_img)
        source_img_trans = source_img_trans.unsqueeze(0).repeat(num_frames, 1, 1, 1)
        source_img_trans = [source_img_trans.cuda()]

        # Generate reenactments.
        frames = run_batch(model, source_img_trans, driver_vid_trans)
        frames = frames.clamp(min=0, max=1)

        # Convert frames to numpy.ndarrays.
        numpy_frames = np.uint8(frames.cpu().data.permute(0, 2, 3, 1).numpy() * 255)
        for frame in numpy_frames:
            result_frames.append(frame)

    return result_frames

def main(data_dir):
    """
    Generates videos with X2Face using the same driving video and source
    video combinations used with Face2Face.

    Args:
        data_dir: Base directory of the FaceForensics++ dataset.
    """

    face2face_dir = os.path.join(data_dir, 'manipulated_sequences/Face2Face/c0/videos')
    orig_dir = os.path.join(data_dir, 'original_sequences/c0/videos')
    image_dir = os.path.join(data_dir, 'original_sequences_images',
                             COMPRESSION_LEVEL, 'images')
    output_dir = os.path.join(data_dir, 'manipulated_sequences/X2Face',
                              COMPRESSION_LEVEL, 'videos')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_x2face_model()
    pairs = get_seq_combos(face2face_dir)

    reenact_count = 0
    for source_id, driver_id in pairs:
        output_path = os.path.join(output_dir, '{}_{}.mp4'.format(source_id, driver_id))
        if os.path.exists(output_path):
            # Do not recreate a video if it already exists.
            # If the user wants to recreated a video
            # the existing video must be deleted first.
            continue

        print('Computing reenactment for {} onto {}...'.format(driver_id, source_id))
        # Validate that input files exist.
        video_path = os.path.join(orig_dir, '{}.mp4'.format(driver_id))
        image_path = os.path.join(image_dir, '{}.png'.format(source_id))
        if not os.path.isfile(video_path):
            print('Failed to find video sequence {}'.format(source_id),
                  file=stderr)
            continue
        if not os.path.isfile(image_path):
            print('Failed to find image for sequence {}'.format(source_id),
                  file=stderr)
            continue

        # Load and crop each frame of the video.
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        # Load image and crop around a face.
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray, model='hog')
        if len(face_locations) == 0:
            print('No face found in image for sequence {}'.format(source_id),
                  file=stderr)
            continue
        cropped_src = crop_face(image, face_locations[0], ZOOMOUT_FACTOR)

        # Compute reenactment.
        frames = reenact(model, video, cropped_src)
        video.release()

        # Write reenactment to disk.
        output_path = os.path.abspath(output_path)
        print('Writing video to "{}"'.format(output_path))
        try:
            write_video(frames, fps, (256, 256), output_path)
        except KeyboardInterrupt as e:
            # Safely handle premature termination.
            # Remove unfinished file.
            if os.exists(output_path):
                os.remove(output_path)
            raise e
        reenact_count += 1

    if reenact_count == 0:
        print('No reenactments were created')
    else:
        print('{} reenactments created'.format(reenact_count))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Generates videos with ICFace')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='Base directory for FaceForensics++ dataset')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)

        main(data_dir)
    except KeyboardInterrupt:
        print('Program terminated prematurely')

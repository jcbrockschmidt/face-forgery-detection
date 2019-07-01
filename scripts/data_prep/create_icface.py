#!/usr/bin/env python3

"""Generates videos with ICFace for the FaceForensics++ dataset."""

import argparse
import cv2
import icface.test_code_released
from icface.test_code_released.util.util import crop_face
from icface.test_code_released.options.test_options import TestOptions
from icface.test_code_released.data.data_loader import CreateDataLoader
from icface.test_code_released.models.models import create_model
import os
import shutil
import sys
from sys import stderr

from utils import get_seq_combos

# Directory for saving temporary data.  WILL BE DELETED.
TEMP_DIR = 'temp-93652'

class ICfaceModel:
    """
    Wrapper for ICface model creation and dataset loading.
    """

    def __init__(self):
        old_argv = sys.argv
        args = '--model pix2pix --which_model_netG resnet_6blocks --which_direction AtoB --dataset_mode aligned --norm batch --display_id 0 --batchSize 1 --loadSize 128 --fineSize 128 --no_flip --name gpubatch_resnet --how_many 1 --ndf 256 --ngf 128 --gpu_ids 0'.split()
        icface_dir = icface.test_code_released.__path__._path[0]
        args += ['--dataroot', icface_dir]
        args += ['--checkpoints_dir', os.path.join(icface_dir, 'checkpoints')]

        sys.argv = [sys.argv[0]] + args
        self.dataroot = icface.test_code_released.__path__._path[0]
        self.opt = TestOptions().parse()
        self.opt.nThreads = 1
        self.opt.batchSize = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True

        sys.argv = old_argv

        self.model = create_model(self.opt)

    def _crop_image(self, source_img, dest_img):
        """
        Crops an image around a face and saves it to disk.

        Args:
            source_img: Path to original image.
            dest_img: Path to save new image to.

        Returns:
            True on success.  False on failure.
        """
        img = cv2.imread(source_img)
        cropped = crop_face(img)
        if cropped is None:
            return False
        else:
            if cv2.imwrite(dest_img, cropped[0]):
                return True
            else:
                return False

    def _get_dataset(self, driver_enc, source_img):
        """
        Creates a data loader for a driver video and source image.

        Args:
            driver_enc: Path to OpenFace facial landmark encoding for the driver video.
            source_img: Path to source image.

        Returns:
            A BaseDataset containing the driving video encoding and
            source image on success.  None on failure.
        """

        # Crop source image around a face.
        img_name = os.path.splitext(os.path.basename(source_img))[0]
        cropped_img = os.path.join(TEMP_DIR, '{}_cropped.png'.format(img_name))
        res = self._crop_image(source_img, cropped_img)
        if not res:
            return None
        else:
            self.opt.csv_path = driver_enc
            self.opt.which_ref = cropped_img
            data_loader = CreateDataLoader(self.opt)
            dataset = data_loader.load_data()
            return dataset

    def reenact(self, driver_enc, source_img, output_path):
        """
        Creates a data loader for a driver video and source image.

        Args:
            driver_enc: Path to OpenFace facial landmark encoding for the driver video.
            source_img: Path to source image.
            output_path: Path to output video.

        Returns:
            True on success.  False on failure.
        """
        dataset = self._get_dataset(driver_enc, source_img)
        if dataset is None:
            return False

        self.opt.result_video = output_path
        for i, data in enumerate(dataset):
            self.model.set_input(data)
            try:
                self.model.test()
            except Exception as e:
                if os.path.exists(e):
                    os.remove(output_path)
                raise e
            break
        return True

def cleanup(data_dir):
    # Remove excess data created by our ICface model.
    checkpoints = 'checkpoints'
    results = os.path.join(
        data_dir, 'original_sequences_images/c0/images/results')
    if os.path.exists(checkpoints):
        shutil.rmtree(checkpoints)
    if os.path.exists(results):
        shutil.rmtree(results)

    # Remove our temporary data.
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def main(data_dir):
    """
    Generate videos with ICface using the same driving video and source
    video combinations used with Face2Face.

    Args:
        data_dir: Directory for FaceForensics<++ dataset.
    """

    print('Computing video reenactments with ICface...')

    # Directory containing Face2Face videos.
    face2face_dir = os.path.join(
        data_dir, 'manipulated_sequences/Face2Face/c0/videos')

    # Base directory for ICface data.
    icface_dir = os.path.join(data_dir, 'manipulated_sequences/ICface')

    # Directory containing encodings.
    enc_dir = os.path.join(icface_dir, 'encodings')

    # Directory to save videos to.
    output_dir = os.path.join(icface_dir, 'c0/videos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Directory containing images.
    image_dir = os.path.join(
        data_dir, 'original_sequences_images/c0/images')

    # Create temporary directory.
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Load model.
    model = ICfaceModel()

    pairs = get_seq_combos(face2face_dir)
    reenact_count = 0
    for source_id, driver_id in pairs:
        output_path = os.path.join(
            output_dir, '{}_{}.mp4'.format(source_id, driver_id))
        if os.path.exists(output_path):
            # Do not recreate a video if it already exists.
            # If the user wants to recreated a video
            # the existing video must be deleted first.
            continue

        print('Computing reenactment for {} onto {}...'.format(driver_id, source_id))

        # Validate that input files exist.
        encoding_path = os.path.join(enc_dir, '{}.csv'.format(driver_id))
        image_path = os.path.join(image_dir, '{}.png'.format(source_id))
        if not os.path.isfile(encoding_path):
            print('ERROR: Failed to find encoding ' \
                  'for video sequence {}'.format(driver_id),
                  file=stderr)
            continue
        if not os.path.isfile(image_path):
            print('ERROR: Failed to find image ' \
                  'for sequence {}'.format(source_id),
                  file=stderr)
            continue

        # Generate reenactment.
        res = model.reenact(encoding_path, image_path, output_path)
        if not res:
            print('ERROR: Something went wrong.  ' \
                  'Reenactment for {} onto {} failed'.format(
                      driver_id, source_id),
                  file=stderr)
        else:
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
    finally:
        cleanup(data_dir)

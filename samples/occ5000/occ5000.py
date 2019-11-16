"""

Mask R-CNN
Train on my occ5000 dataset

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Kidd Liu

"""

import os
import sys
import datetime
import numpy as np
import skimage.draw

# Root directory of the projec
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/occ5000"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################

class Occ5000Config(Config):
    """
    Configuration for training on the occ5000 dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "occ5000"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # The default image size is 1024x1024px.
    IMAGES_PER_GPU = 2
    
    # Number of classes (including background)
    # Background+hair+face+upper closes+left arm
    # +right arm+left hand +right hand+ left leg
    # +right leg+ left feet+ right feet+ accesessory
    NUM_CLASSES = 12 + 1
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1250
    
    # Class names and accessory can be ignored
    CLASS_NAMES = ['hair', 'face', 'upper_clothes', 'left_arm', 
             'right_arm', 'left_hand', 'right_hand',
             'left_leg', 'right_leg', 'left_feet', 'right_feet',
             'accessory']
    
    IMAGE_MIN_DIM = 512

    IMAGE_MAX_DIM = 1024

############################################################
#  Dataset
############################################################

class Occ5000Dataset(utils.Dataset):

    def load_occ5000(self, dataset_dir, subset):
        """
        Load a subset of the Balloon dataset.
        dataset_dir: The root directory of the COCO dataset(like /home/kidd/kidd1/occ5000)
        subset: What to load (train_all2500, val_all2500, see in occ5000/list)
        
        TODO: resize the train val to 4000(3200occ + 800unOcc) and 1000(800occ+200unocc)
              which is 80% of the whole dataset
        """
        
        # Add classes, can ignore some class here
        classNames = Occ5000Config.CLASS_NAMES
        for className, i in zip(classNames, np.arange(len(Occ5000Config.CLASS_NAMES))):
            self.add_class('occ5000', i+1, Occ5000Config.CLASS_NAMES[i])
        
        # Train or validation datset
        assert subset in ['train_all2500', 'val_all2500']
        #dateset_dir = os.path.join(dataset_dir, sub)
        
        # Get image and annotation path in the format 'impath annpath'
        lines = []
        with open(os.path.join(dataset_dir, 'list', subset+'.txt'), 'r') as list_file:
            while True:
                line = list_file.readline()
                if not line:
                    break
                line = line.strip('\n')
                lines.append(line)
                
        # Seperate image and annotation path
        im_paths = []
        ann_paths = []
        for line in lines:
            im_path = line[0:line.find('.png')+4]
            ann_path = line[line.find('.png')+5:]
            im_paths.append(im_path)
            ann_paths.append(ann_path)
        
        # Read image and annotation from png and add images
        for im_path, ann_path in zip(im_paths, ann_paths):
            im = skimage.io.imread(dataset_dir + im_path)
            height, width = im.shape[:2]
            _, image_id = os.path.split(im_path)
            path = dataset_dir + im_path
            ann = skimage.io.imread(dataset_dir + ann_path)
            
            self.add_image('occ5000',
                          image_id = image_id,
                          path = path,
                          width = width, height = height,
                          annotation = ann)
    
    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """       
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != 'occ5000':
            return super(self.__class__, self).load_mask(image_id)
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        masks = []
        classID = []
        ann = info['annotation']
        for i in np.arange(1, len(Occ5000Config.CLASS_NAMES) + 1):
            mask = np.where(ann == i, 
                            np.ones(ann.shape, dtype=np.uint8),
                           np.zeros(ann.shape, dtype=np.uint8))
            if np.sum(mask) != 0:
                masks.append(mask)
                classID.append(i)
        masks = np.stack(masks, axis = 2)
        #classID = np.arange(1,len(Occ5000Config.CLASS_NAMES) + 1)
        
        return masks, np.array(classID).astype(np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'occ5000':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
            
############################################################
#  Training
############################################################
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Occ5000Dataset()
    dataset_train.load_occ5000(args.dataset, "train_all2500")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Occ5000Dataset()
    dataset_val.load_occ5000(args.dataset, "val_all2500")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
    description='Train Mask R-CNN on occ5000.')
    
    parser.add_argument('command',
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar='/path/to/occ5000',
                        help='Directory of the occ5000 dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
     
    args = parser.parse_args()
    
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    # Configurations
    if args.command == "train":
        config = Occ5000Config()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    
    # Creat model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
        
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
        
    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))

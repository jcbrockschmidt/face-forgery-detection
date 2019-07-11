import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_data_generator(data_dir, other_classes, batch_size):
    """
    Creates a 2-class data generator for real and fake face images.

    Args:
        data_dir: Directory containing class directories.
        other_classes: Collection of classes other than "real".  The class
            "real" will be included in the generator by default.
        batch_size: Number of images to process at time.

    Returns:
        A DirectoryIterator and a dictionary of class weights.
        The DirectoryIterator has two classes, "fake" and "real", where fake
        images are labeled 0 and real images are labeled 1.  The class weights
        maps class indices (0 for fake and 1 for "real") to the inverse of their
        proportion of the samples. For instance, if their were 12 fake images
        and 8 real images, the class weights would be

        {
            0 : 0.4,
            1 : 0.6
        }

        The real images are weighted more heavily since they have fewer samples.
        These weights can help combat class imbalances during training.
    """
    # Initialize generator.
    classes = list(other_classes) + ['real']
    generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
        data_dir,
        classes=classes,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    # Modify data labels.
    real_index = generator.class_indices['real']
    new_classes = [1 if i == real_index else 0 for i in generator.classes]
    generator.classes = np.array(new_classes, dtype=np.int32)

    # Change class-to-index mapping.
    new_indices_map = {
        'fake' : 0,
        'real' : 1
    }
    generator.class_indices = new_indices_map

    # Calculate the weights.
    _, counts = np.unique(generator.classes, return_counts=True)
    fake_prop = counts[0] / generator.samples
    real_prop = 1 - fake_prop
    weights = {
        0 : real_prop,
        1 : fake_prop
    }

    return generator, weights

def tpr_pred(y_true, y_pred):
    """
    Custom Keras metrics that calculates the true positive rate / sensitivity.

    Returns:
        TP / (TP + FN)
    """
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    return tp / (tp + fn + K.epsilon())

def tnr_pred(y_true, y_pred):
    """
    Custom Keras metrics that calculates what true negative rate / specificity.

    Returns:
        TN / (TN + FP)
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    return tn / (tn + fp + K.epsilon())

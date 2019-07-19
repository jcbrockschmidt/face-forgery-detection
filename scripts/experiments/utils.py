import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

IMG_SIZE = (256, 256)

def create_data_generator(data_dir, other_classes, batch_size, class_mode):
    """
    Creates a 2-class data generator for real and fake face images.

    Args:
        data_dir: Directory containing class directories.
        other_classes: Collection of classes other than "real".  The class
            "real" will be included in the generator by default.  Order matters
            when `class_mode` is not "binary".
        batch_size: Number of images to process at time.
        class_mode: See `keras.preprocessing.image.ImageDataGenerator.flow_from_directory`.

    Returns:
        A DirectoryIterator and a dictionary of class weights.
        The class weights map class indices to the inverse of their sample
        count. For instance, if their were 100 images belonging the first fake
        class, 50 to second fake class, and 20 to the real class, the weights
        would be

        {
            0 : 0.01,
            1 : 0.02,
            2 : 0.05
        }

        The classes with fewer images are weighted more heavily.  These weights
        can help combat class sample imbalances during training.
    """
    # Initialize generator.
    classes = list(other_classes) + ['real']
    generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
        data_dir,
        classes=classes,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training')

    if class_mode == 'binary':
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
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = 1 / count

    return generator, weights

def load_single_class_generators(data_dir, classes, batch_size=16):
    """
    Creates a dictionary of data generators for a list of classes.

    Args:
        data_dir: Directory containing classes.
        classes: List of classes to make generators for.  Should have
            corresponding directories within the directory pointed to by
            `data_dir`.
        batch_size: Number of images to read at a time.

    Returns:
        Dictionary mapping class names to data generators.
    """
    generators = {}
    for c in classes:
        path = os.path.join(data_dir, c)
        if not os.path.isdir(path):
            print('ERROR: No directory for class "{}" in "{}"'.format(c, data_dir),
                  file=stderr)
            exit(0)
        gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
            data_dir,
            classes=[c],
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            subset='training')

        # Real images need to be labeled "1" and not "0".
        if c == 'real':
            # Modify data labels.
            new_classes = np.ones(gen.classes.shape, dtype=gen.classes.dtype)
            gen.classes = np.array(new_classes, dtype=np.int32)

            # Change class-to-index mapping.
            new_indices_map = {'real' : 1}
            gen.class_indices = new_indices_map

        generators[c] = gen

    return generators

def tpr_pred(y_true, y_pred):
    """
    Keras metric for the true positive rate / sensitivity for a
    binary classifier.

    Returns:
        TP / (TP + FN)
    """
    y_pred_round = K.round(y_pred)
    neg_y_pred = 1 - y_pred_round
    tp = K.sum(y_true * y_pred_round)
    fn = K.sum(y_true * neg_y_pred)
    return tp / (tp + fn + K.epsilon())

def tnr_pred(y_true, y_pred):
    """
    Keras metric for the true negative rate / specificity for a
    binary classifier.

    Returns:
        TN / (TN + FP)
    """
    y_pred_round = K.round(y_pred)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred_round
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred_round)
    return tn / (tn + fp + K.epsilon())

def tpr_cat_pred(y_true, y_pred):
    """
    Keras metrics for the true positive rate / sensitivity for a categorical
    classifier. The last category/class is considered the real/positive class.
    All other classes are grouped into the same category of fake/negative

    Returns:
        TP / (TP + FN)
    """
    class_true = K.argmax(y_true, axis=-1)
    class_pred = K.argmax(y_pred, axis=-1)

    real_class = K.int_shape(y_pred)[1] - 1
    real_true = K.cast(K.equal(class_true, real_class), K.floatx())
    real_pred = K.cast(K.equal(class_pred, real_class), K.floatx())
    neg_real_pred = 1 - real_pred

    tp = K.sum(real_true * real_pred)
    fn = K.sum(real_true * neg_real_pred)

    return tp / (tp + fn + K.epsilon())

def tnr_cat_pred(y_true, y_pred):
    """
    Keras metrics for the true negative rate / specificity for a categorical
    classifier. The last category/class is considered the real/positive class.
    All other classes are grouped into the same category of fake/negative.

    Returns:
        TN / (TN + FP)
    """
    class_true = K.argmax(y_true, axis=-1)
    class_pred = K.argmax(y_pred, axis=-1)

    real_class = K.int_shape(y_pred)[1] - 1
    real_true = K.cast(K.equal(class_true, real_class), K.floatx())
    real_pred = K.cast(K.equal(class_pred, real_class), K.floatx())
    neg_real_true = 1 - real_true
    neg_real_pred = 1 - real_pred

    tn = K.sum(neg_real_true * neg_real_pred)
    fp = K.sum(neg_real_true * real_pred)

    return tn / (tn + fp + K.epsilon())

def cat_acc_pred(y_true, y_pred):
    """
    Keras metrics for the accuracy of a categorical classifier.  The last
    category/class is considered the real/positive class.  All other classes
    are grouped into the same category of fake/negative.

    Returns:
        (TP + TN) / (TP + TN + FP + FN)
    """
    class_true = K.argmax(y_true, axis=-1)
    class_pred = K.argmax(y_pred, axis=-1)

    real_class = K.int_shape(y_pred)[1] - 1
    real_true = K.cast(K.equal(class_true, real_class), K.floatx())
    real_pred = K.cast(K.equal(class_pred, real_class), K.floatx())
    neg_real_true = 1 - real_true
    neg_real_pred = 1 - real_pred

    tp = K.sum(real_true * real_pred)
    tn = K.sum(neg_real_true * neg_real_pred)
    fp = K.sum(neg_real_true * real_pred)
    fn = K.sum(real_true * neg_real_pred)

    return (tp + tn) / (tp + tn + fp + fn)

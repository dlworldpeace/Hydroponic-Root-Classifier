from os import listdir, rename
from os.path import isdir, isfile, join

from random import shuffle
import pandas as pd



def get_image_names(path_to_dir):
    """ Get image names in the provided directory

    path: a path to a directory of images
    returns: a list of image names in the directoy
    """
    assert isdir(path_to_dir), "path not a valid directoy"
    dir_entries = listdir(path_to_dir)
    # ensure images are files and end in ".jpg"
    images = list(filter(lambda entry_name: isfile(join(path_to_dir,
                                                        entry_name)) and
                                                   entry_name.endswith(".jpg"),
                       dir_entries))
    return images

def partition_set(image_names, test_set_size, test_set_name, output_file_name):
    # ensure randomness
    shuffle(image_names)
    # build training set
    training_set = image_names[test_set_size:]
    # build test set with blank values appended to match trianing set size
    blanks_to_append = len(training_set) - test_set_size
    test_set = image_names[:test_set_size] + [""] * blanks_to_append

    # store in csv file
    data_dict = {test_set_name: test_set,
                 "training set": training_set}
    data_frame = pd.DataFrame(data=data_dict)
    data_frame.to_csv(output_file_name)



def create_validation_set(test_partition_name):
    test_set_partition = pd.read_csv(test_partition_name)
    training_set_names = list(test_set_partition["training set"])
    partition_set(training_set_names, 134, "validation set", "validation_partition.csv")


def main():

    image_names = get_image_names("allData/validation")
    print(len(image_names))
    for name in image_names:
        if name.endswith("a.jpg"):
            rename(join("allData/validation", name), join("allData/validation/hairy", name))






main()

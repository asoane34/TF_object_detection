'''
Script to produce .tfrecord files for use with Tensorflow Object Detection API
My images did have .xml annotations compatible with the scripts provided in
Tensorflow Object Detection repo, and I didn't feel like annotating 3400 images
so I wrote my own
'''
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field 
from PIL import Image
from io import BytesIO
import os 

@dataclass
class RecordWriter():
    '''
    Object to prepare TFRecord file for TF Object Detection API.
    Input is pd.DataFrame that MUST contain the following features:
        
        - filename [str]: file location (jpeg or png)
        - xmin [int]: left limit of object boundary box
        - xmax [int]: right limit of object boundary box
        - ymin [int]: lower limit of object boundary box
        - ymax[int]: upper limit of object boundary box
        - class[int]: Numeric class of object in boundary box
            IMPORTANT: class label integers MUST start at 1, not 0! 0 is reserved
            in TF object detection API.
        - class_text[str]: Text label for object in box
    
    The input DataFrame can have many other features (width, height, truncated, etc.) but these 
    are the bare minimum to write a .tfrecord. 
    '''
    df: pd.core.frame.DataFrame
    group_var: str
    img_dir: str
    output_path_train: str
    output_path_validation: str = None
    uniform_size: bool = True
    test_size: float = 0.2
    stratify: bool = True
    '''
    Args:
    
        - df [pd.DataFrame]: pandas DataFrame specified above
        - group_var [str]: Feature to group DataFrame by (i.e. unique key for each image, typically filename)
        - img_dir [str]: Path to directory with training images
        - output_path_train [str]: Location to write output .tfrecord file, training set
        - output_path_validation [str]: location to write output .tfrecord file, validation set
        - uniform_size [bool]: Boolean flag to indicate whether width/height is known and feature in df,
            or if width/height must be determined from input image. 
        - test_size [float]: size of validation set
        - stratify [bool]: flag to indicate whether to use stratified split (only option at moment)
    '''
        
    def write_records(self):
        '''
        Write .tfrecord file (Will be adding sharding option soon)
        
        Args:
            
            - None
            
        Returns:
        
            - .tfrecord file
        '''
        
        if self.stratify: #this is the only option it supports at the moment, will be adding
        
            all_train, all_validation = self.stratified_split()

        for i, df in enumerate([all_train, all_validation]):
        
            all_groups = self.split_groups(df, "filename")
            
            if i == 0:
            
                with tf.io.TFRecordWriter(self.output_path_train) as writer:
                    
                    for input_tuple in all_groups:
                        
                        tf_record = self.create_tf_record(input_tuple)
                        
                        writer.write(tf_record.SerializeToString())
                        
                print("Record successfully written")

            else:

                with tf.io.TFRecordWriter(self.output_path_validation) as writer:
                    
                    for input_tuple in all_groups:
                        
                        tf_record = self.create_tf_record(input_tuple)
                        
                        writer.write(tf_record.SerializeToString())
                        
                print("Record successfully written")
        
        
    def create_tf_record(self, input_tuple):
        '''
        Format data for writing to .tfrecord using tf.train.Example
        Args:
            
            - input_tuple [tuple]: tuple containing (filename, pd.DataFrame) for each image
                where a single row is one boundary box in that image. 
        Returns:
            
            - tf_record [tf.train.Example]
        
        '''
        with tf.io.gfile.GFile(os.path.join(self.img_dir, input_tuple[0]), "rb") as file:
            
            encoded = file.read()
            
        if self.uniform_size:
            
            width, height = self.df.width.max(), self.df.height.max()
            
        else:
            
            encoded_io = BytesIO(encoded)
            
            with Image.open(encoded_io) as image:
            
                width, height = image.size
        
        by = self.df.source.max().encode()
        
        filename = input_tuple[0].encode()
        
        source = input_tuple[0].split(".")[0].encode()
        
        img_format = b"jpeg"
        
        xmin = input_tuple[1].xmin.values / width
        
        xmax = input_tuple[1].xmax.values / width
        
        ymin = input_tuple[1].ymin.values / height
        
        ymax = input_tuple[1].ymax.values / height
        
        classes = input_tuple[1]["class"].values
        
        classes_text = [text.encode() for text in list(input_tuple[1].class_text.values)]
        
        tf_record = tf.train.Example(features = tf.train.Features(feature = {
            
            "image/height" : tf.train.Feature(int64_list = tf.train.Int64List(value = [height])),
            
            "image/width" : tf.train.Feature(int64_list = tf.train.Int64List(value = [width])),
            
            "image/filename" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename])),
            
            "image/source" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [source])),
            
            "image/by" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [by])),
            
            "image/encoded" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [encoded])),
            
            "image/format" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_format])),
            
            "image/object/bbox/xmin" : tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
            
            "image/object/bbox/xmax" : tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
            
            "image/object/bbox/ymin" : tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
            
            "image/object/bbox/ymax" : tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
            
            "image/object/class/text" : tf.train.Feature(bytes_list = tf.train.BytesList(value = classes_text)),
            
            "image/object/class/label" : tf.train.Feature(int64_list = tf.train.Int64List(value = classes))
        }))
        
        return(tf_record)
    
    def stratified_split(self):

        source_strat = self.df.groupby(self.group_var).source.max()

        source_df = pd.DataFrame({"filename" : source_strat.index, "source" : source_strat.values})

        train, validation = train_test_split(source_df, 
        stratify = source_df["source"], test_size = self.test_size, random_state = 42)

        train_files, validation_files = train.filename.values, validation.filename.values

        all_train = self.df[self.df.filename.isin(train_files)].reset_index(drop = True)

        all_validation = self.df[self.df.filename.isin(validation_files)].reset_index(drop = True)

        return(all_train, all_validation)

    @staticmethod
    def split_groups(df, group_var):
        '''
        Split DataFrame into individual image dataframes. Necessary for images with many bounding boxes
        Args:
            
            - None
            
        Returns:
        
            - [list]: Returns list of tuples containing (filename, pd.DataFrame) for each image
        '''
        
        by_picture = df.groupby(group_var)
        
        return([(img_id, by_picture.get_group(f)) for img_id, f in zip(by_picture.groups.keys(),
                                                                      by_picture.groups)])

    @staticmethod
    def write_labels(class_list, label_path):

        with open(label_path, "w+") as f:

            for i, _class in enumerate(class_list):
    
                f.write("item {{\n id: {}\n name:\'{}\'\n}}".format(i + 1, _class))


if __name__ == "__main__":

    INPUT_PATH = "./global-wheat-detection"

    GROUP_VAR = "filename"

    IMG_DIR = os.path.join(INPUT_PATH, "train")

    OUTPUT_PATH_TRAIN = os.path.join(INPUT_PATH, "train.tfrecord")

    OUTPUT_PATH_VALIDATION = os.path.join(INPUT_PATH, "validation.tfrecord")

    DF = pd.read_csv(os.path.join(INPUT_PATH, "pre_tf.csv.gz"), compression = "gzip")

    record_writer = RecordWriter(DF, GROUP_VAR, IMG_DIR, OUTPUT_PATH_TRAIN,
    OUTPUT_PATH_VALIDATION)

    record_writer.write_records()

    record_writer.write_labels(["wheat-head"], os.path.join(INPUT_PATH, "label_map.pbtxt"))
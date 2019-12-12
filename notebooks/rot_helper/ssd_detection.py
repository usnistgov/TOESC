# Object detection imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import csv
import json
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

paths = 'rot_helper/config_sensible_paths.json' #M.
#paths = 'rot_helper/config_sensible_paths_lambda.json' #L.
datastore = None
with open(paths, 'r') as f:
    datastore = json.load(f)

# This is needed since the notebook is stored elsewhere.
sys.path.append(datastore['paths']['research']['path'])

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#### Frozen and trained ML models for detection and forecasting tasks -- To be updated
# Detection frozen tf graph

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = datastore['paths']['PATH_TO_FROZEN_GRAPH']['path']

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(datastore['paths']['PATH_TO_LABELS']['path'], 'sky_images_label_map.pbtxt')

#### Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#### Loading label map

"""Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. 
Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
"""

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#### Detection

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#### Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def detection_data(sample_image_file):

	obj_detections_dict = {} # To hold the image path and its prediction output dictionary.

	image = Image.open(sample_image_file)
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = load_image_into_numpy_array(image)
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	# Actual detection.
	output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

	obj_detections_dict[sample_image_file] = output_dict

	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		output_dict['detection_boxes'],
		output_dict['detection_classes'],
		output_dict['detection_scores'],
		category_index,
		instance_masks=output_dict.get('detection_masks'),
		use_normalized_coordinates=True,
		line_thickness=8)
    
	plt.figure(figsize=IMAGE_SIZE)
	plt.imshow(image_np)

	return obj_detections_dict

def post_process_detection_data(results_detection_data):
	""" For each image in the test set, predict a list of boxes describing objects in the image. Each box is described as:

		- ImageID,PredictionString
		- ImageID,{Label Confidence XMin YMin XMax YMax},{...}
	"""

	threshold = 0.5
	label_map_dict = {1 : 'sun', 2 : 'cloud', 3 : 'occlusion'}

	imageId_PredictionString_dict = {} # {ImageID: PredictionString}

	for key, detection_dict in results_detection_data.items():

		ImageID = key
		PredictionString = [] # [{Label Confidence XMin YMin XMax YMax}, {...}]

		for index, label in enumerate(detection_dict['detection_classes']):

			confidence = float(detection_dict['detection_scores'][index])
			if confidence > threshold: 

				boxes = " ".join(map(str, detection_dict['detection_boxes'][index]))
				PredString = str(label_map_dict[label]) + " " + str(confidence) + " " + boxes

				PredictionString.append(PredString)

		imageId_PredictionString_dict[ImageID] = " ".join(PredictionString)

	return imageId_PredictionString_dict
# Forecasting Helper Functions.
import numpy as np
import tensorflow as tf

def local_padding_series(ssd_detection_data, max_bboxes_vec_size):
    """ Given an ssd_detection_data, return its padded version """
    
    padded_series = []
    
    vec = ssd_detection_data
    
    vec_size = len(vec)

    if vec_size == 1: # 'nan'
        vec[0] = 0.0
        padding_size = max_bboxes_vec_size - vec_size
        padded_series.append(list(np.pad(vec, (0,padding_size), 'constant', constant_values=(0))))
        #padded_series.append(list(np.pad(vec, (0,padding_size), 'mean')))

    elif vec_size > max_bboxes_vec_size:
        padded_series.append(vec[:max_bboxes_vec_size]) # Trunc

    elif vec_size == max_bboxes_vec_size:
        padded_series.append(vec) # Do nothing

    else:
        padding_size = max_bboxes_vec_size - vec_size
        padded_series.append(list(np.pad(vec, (0,padding_size), 'constant', constant_values=(0)))) # Padd
        #padded_series.append(list(np.pad(vec, (0,padding_size), 'mean')))
        
    return np.array(padded_series)

def post_process_ssd_detection_data_for_forecasting(ssd_detection_data, max_bboxes_vec_size):
    """ Given an output SSD detection for a given image, post-process it, and pad it and return the new string."""
    
    ssd_detection_data = ssd_detection_data.split()
    reverse_label_map_dict = {'sun':10, 'cloud': 7, 'occlusion':0} # DJ Idea.
    
    for i in range(len(ssd_detection_data)):
        if ssd_detection_data[i] == 'sun':
            ssd_detection_data[i] = reverse_label_map_dict['sun']

        if ssd_detection_data[i] == 'cloud':
            ssd_detection_data[i] = reverse_label_map_dict['cloud']

        if ssd_detection_data[i] == 'occlusion':
            ssd_detection_data[i] = reverse_label_map_dict['occlusion']
    
    for i in range(len(ssd_detection_data)):
        ssd_detection_data[i] = float(ssd_detection_data[i])
    
    return local_padding_series(ssd_detection_data, max_bboxes_vec_size) 

def return_sequence_ssd_detection_data(popped_sky_image_filename, ssd_detection_data_dict, seq_num, max_bboxes_vec_size):
    """ Given a dictionary of ssd detection data and key to search and extract seq_num detection data """

    ssd_keys = list(ssd_detection_data_dict.keys())
    ssd_keys.sort()

    matched_ssd_keys = []
    sequence_ssd_detections = []

    for i in range(len(ssd_keys)):
        if ssd_keys[i] == popped_sky_image_filename:
            matched_ssd_keys = ssd_keys[i:i+seq_num]
            break

    for key in matched_ssd_keys:
        processed_padded_ssd_data = post_process_ssd_detection_data_for_forecasting(ssd_detection_data_dict[key], max_bboxes_vec_size)
        sequence_ssd_detections.append(processed_padded_ssd_data[0])

    return sequence_ssd_detections

def forecast_inferrence(extracted_seq_num_detection_data, forecasting_model_name):
    """ Given an extracted sequence of ssd detection data, 
        feed them to trained LSTM model for forecast the next frame detection data """
    
    forecast_result = []
    model = tf.keras.models.load_model(forecasting_model_name)
    
    detection_np_array = np.array(extracted_seq_num_detection_data)
    #print(detection_np_array.shape)
    
    test_one_detection_np_array = np.array([detection_np_array,])
    #print(test_one_detection_np_array.shape, detection_np_array.shape)
    
    test_one_prediction_result = model.predict(test_one_detection_np_array)
    #print(test_one_prediction_result[0], test_one_prediction_result[0].shape)
    
    return test_one_prediction_result
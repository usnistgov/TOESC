# Forecasting Helper Functions.
import numpy as np
import tensorflow as tf

DEBUG=False

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

def n_windowed_forecast_inferrence(forecast_window, extracted_seq_num_detection_data, forecasting_model_name):
    """ Given: 
            - Forecast window (number of future frames to predict)
            - A seq_num of images detection data
            - Trained forecasting model
        Return:
            - n_future frames prediction data
    """
    
    forecast_detection_result = []
    model = tf.keras.models.load_model(forecasting_model_name)
    
    for i in range(forecast_window):
    
        detection_np_array = np.array(extracted_seq_num_detection_data)
        test_one_detection_np_array = np.array([detection_np_array,])
        test_one_prediction_result = model.predict(test_one_detection_np_array)
        
        #print("\n:: Prediction at: ", i, " iteration: \n\t", test_one_prediction_result)

        forecast_detection_result.append(test_one_prediction_result)
        
        extracted_seq_num_detection_data = np.append(extracted_seq_num_detection_data, test_one_prediction_result)
        extracted_seq_num_detection_data = np.reshape(extracted_seq_num_detection_data, (51, 30))
        
        extracted_seq_num_detection_data = extracted_seq_num_detection_data[1:]
        
    print("\n", forecast_window, "- Forecast session is done.\n")
    
    return forecast_detection_result

def forecast_report_on_occlusions_predictions_area(forecast_results):
    """ Given the output of our forecasting model, explore it with a focus on occlusion events forecast """
    
    forecast_booleanist_final = [] # Array of yes/no forecast for the occlusion event occurrences.

    if DEBUG:
        print("\n\n:: Single Forecast Report Start:")

    for forecast_data in forecast_results:
        ### Extract the first 30 predicted data points, which represent one image's object detection data

        forecast_booleanist = [] # 1 or 0
        objects_detection_list = []

        if len(forecast_data) == 30:
            for i in range(0, 30, 6):
                objects_detection_list.append(forecast_data[i:(i+6)])

        #print("\n\n:: objects_detection_list: ", objects_detection_list, "\n\n")

        ### Process objects_detection_list to determine if the object type is within the range of occlusion type density and 
        ###     the bounding box area of the predicted occlusion area to be a potential forecast.
        
        for object_data in objects_detection_list:

            object_data_list = list(object_data)
            object_type = float(object_data_list[0])

            #print("\nobject_data_list: ", object_data_list)
            #print("\nobject_type: ", object_type)

            if object_type <= 0.3 and object_type >= -0.3: # Occlusion object type values range with a mean of 0.              

                ymin =  float(object_data_list[2])
                xmin =  float(object_data_list[3])
                ymax =  float(object_data_list[4])
                xmax =  float(object_data_list[5])

                w = (xmax - xmin) # Scaled
                h = (ymax - ymin) # Scaled

                area = w * h

                #print("\nScaled area: ", area)

                if area <= 0.0150 and area >= 0.005: # Occlusion bounding box area range. +- (0.005)* over the range (0 - 0.025)
                    forecast_booleanist.append(1)
                else:
                    forecast_booleanist.append(0)
        if DEBUG:
            print("\nforecast_booleanist: ", forecast_booleanist) # Per object detection data

        if sum(forecast_booleanist) >= 1:
            forecast_booleanist_final.append("Yes")
        else:
            forecast_booleanist_final.append("No")

    if DEBUG:
        print("\n\n:: Forecast Report End.")

    return forecast_booleanist_final


def longterm_forecast_report_on_occlusions_predictions_area(forecast_results):
    """ Given the output of our forecasting model, explore it with a focus on occlusion events forecast """
    
    forecast_booleanist_final = [] # Array of yes/no forecast for the occlusion event occurrences.

    if DEBUG:
        print("\n\n:: Initial Long Term Forecast Report Start:")

    for forecast_data in forecast_results:
        ### Extract the first 30 predicted data points, which represent one image's object detection data
        if DEBUG:
            print ("\n\nN-Forecast-Data: ", forecast_data)

        forecast_booleanist = [] # 1 or 0 
        objects_detection_list = []
        
        forecast_booleanist_final.append(forecast_report_on_occlusions_predictions_area(forecast_data))
        if DEBUG:
            input("Checkmate???")

    if DEBUG:
        print("\n\n:: Long Term Forecast Report End.")

    return list((np.array(forecast_booleanist_final)).flatten())
 
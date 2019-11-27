import time
import threading

def sky_images_generator(sky_camera_images_files, queue_of_inbound_sky_images):
	""" Given images and a queue name pop the oldest image from images and insert it into the queue to be consumed later on. """

	for next_image in sky_camera_images_files: # Simple selection of next image

		queue_of_inbound_sky_images.put(next_image)

		time.sleep(5) # Simulate the 5 seconds shutter camera window

	return "Done"
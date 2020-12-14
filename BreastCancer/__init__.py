
import tensorflow as tf
my_app = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_app, device_type='CPU')
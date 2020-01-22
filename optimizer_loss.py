from denpendency import *

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def _content_loss(y, y_pred):  
  return tf.reduce_mean(tf.square(y - y_pred))
  
  
def _adversarial_loss(y_pred):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  y_discrim, y_discrim_logits = discriminator(y_pred)
  return tf.reduce_mean(loss_object(y_discrim_logits, tf.ones_like(y_discrim_logits)))
  
  
def gen_loss_function(y, y_pred):
  return _content_loss(y, y_pred) + 1e-3*(_adversarial_loss(y_pred))
  
  
def discriminator_loss_function(y_real_pred, y_fake_pred, y_real_pred_logits, y_fake_pred_logits):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_real = tf.reduce_mean(loss_object(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
  loss_fake = tf.reduce_mean(loss_object(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
  return loss_real + loss_fake
  

from model import *
from dependency import *
from optimizer_loss import *

@tf.function
def train_step(g_x, g_y , epoch):
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    
    # Generator
    g_y_pred = generator(g_x)
    
    g_loss = gen_loss_function(g_y, g_y_pred)
    
    # Discriminator      
    d_y_real_pred, d_y_real_pred_logits = discriminator(g_y)
    
    d_y_fake_pred, d_y_fake_pred_logits = discriminator(g_y_pred)
    
    d_loss = discriminator_loss_function(d_y_real_pred, d_y_fake_pred, d_y_real_pred_logits, d_y_fake_pred_logits)
    

  generator_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
  
  discriminator_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
  
  
  generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
  
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
  
  return  g_loss , d_loss

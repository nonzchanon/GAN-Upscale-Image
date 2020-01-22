
#Generator model  
class Generator(tf.keras.Model):
  def __init__(self ,learning_rate = 1e-4, num_blocks = 16, num_upsamples=2):
    super(Generator, self).__init__()
    
    self.leanring_rate = learning_rate
    self.num_upsamples = num_upsamples
    self.discriminator = discriminator
    self.num_blocks = num_blocks
    
    self.residual = [
       
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1),
    self.ResidualBlock(filters = 64, kernel_size = 3, strides =1)
        
    
    ]
    
    self.upsample1 = tf.keras.layers.Conv2DTranspose(256, 4,
                                         strides=2,
                                         padding='same',
                                         activation='linear') # (bs, 1024, 256, 3)
    
    self.upsample2 = tf.keras.layers.Conv2DTranspose(256, 4,
                                         strides=2,
                                         padding='same',
                                         activation='linear') # (bs, 1024, 256, 3)
     
    
    self.convo0 = tf.keras.layers.Conv2D(filters = 64, kernel_size=9 , strides = 1, padding='same')  
    self.convo1 = tf.keras.layers.Conv2D(filters = 64, kernel_size=3 , strides = 1, padding = 'same')
    self.convo2 = tf.keras.layers.Conv2D(filters = 3 , kernel_size=9 , strides = 1, padding = 'same')
    
    #self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
    self.relu1 =  tf.keras.layers.ReLU()
    self.batchnormal = tf.keras.layers.BatchNormalization()
    
  def call(self, x):
    
    x = self.convo0(x)
    #x = self.prelu(x)
    x = self.relu1(x)
    skip = x
  
    # ResidualBlock
    for res in self.residual:
      
      skip_x = x
      x = res(x)
      x += skip_x         
     
      
    x = self.convo1(x)
    x = self.batchnormal(x)
    x += skip
    
    # upsamples Blocks
    x = self.upsample1(x)
    x = self.upsample2(x)
      
      
    x = self.convo2(x)
    
    return x

   ###################### Residual
  def ResidualBlock(self, filters, kernel_size ,strides=1):
            
  #  skip = x    
    result = tf.keras.Sequential()            
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = 'same', use_bias = False))
    result.add(tf.keras.layers.ReLU())
    #result.add(tf.keras.layers.PReLU(shared_axes=[1,2]))
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = 'same', use_bias = False))
    result.add(tf.keras.layers.BatchNormalization())
       
  #  x += skip
                
    return result      
    
#Discriminator model  
class Discriminator(tf.keras.Model):
  def __init__(self, learning_rate=1e-4):
    super(Discriminator, self).__init__()
 
    self.learning_rate = learning_rate
  
    self.convo0 = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same')
  
    self.convoblock = [
      self.ConvolutionBlock(32,  3 ,2),
      self.ConvolutionBlock(64, 3 ,1),
      self.ConvolutionBlock(64, 3 ,2),
      self.ConvolutionBlock(128, 3 ,1),
      self.ConvolutionBlock(128, 3 ,2),
      self.ConvolutionBlock(256, 3 ,1),
      self.ConvolutionBlock(256, 3 ,2),
    ]
    
    self.flatten = tf.keras.layers.Flatten()
    self.Dense1 = tf.keras.layers.Dense(512)
    self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)
    self.Dense2 = tf.keras.layers.Dense(1)
    self.leaky2 = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    
  
  def ConvolutionBlock(self,filters, kernel_size, strides):
    # Conv2D + BN + LeakyReLU
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters=filters, kernel_size = kernel_size , strides=strides, padding='same', use_bias=False))
    result.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    return result
    
  def call(self, x):
    
    x = self.convo0(x)
    x = self.leaky1(x)
    
    for convo in self.convoblock:
      x = convo(x)
    
    x = self.flatten(x)
    x = self.Dense1(x)
    x = self.leaky2(x)
    logits = self.Dense2(x)
    x = tf.keras.activations.sigmoid(logits)
    return x, logits
  
 

import keras
from keras.preprocessing.image import load_img, img_to_arraytarget_image_path= 'target.jpg' #target image
style_image_path='style.jpg' #style image
width,height =load_img(target_image_path).size
image_height =400
image_width=int(width * image_height /height)

import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img=load_img(image_path, target_size=(image_height, image_width))
    img= img_to_array(img)
    img=np.expand_dims(img, axis=0)
    img=vgg19.preprocess_input(img)
    return img
    
    
def deprocess_image(x):
    x[:,:, 0] += 103.939
    x[:,:, 1] += 116.779
    x[:,:, 2] += 123.68
    x=x[:, :, ::-1]
    x=np.clip(x,0,255).astype('uint8')
    return x
    
from keras import backend as K

target_image = K.constant (preprocess_image(target_image_path))
style_image= K.constant(preprocess_image(style_image_path))
combination_image=K.placeholder((1, image_height, image_width, 3))
input_tensor=K.concatenate([target_image,style_image,combination_image],axis =0)


model =vgg19.VGG19(input_tensor=input_tensor, weights='imagenet',include_top=False)
print('Model successfully loaded')


def content_loss(base, combination):
    return K.sum(K.square(combination-base))
def style_loss(style,combination):
    S=gram_matrix(style)
    C=gram_matrix(combination)
    channels=3
    size=image_height * image_width
    return (K.sum(K.square(S-C)) / (4. * (channels ** 2) * (size **2)))

def total_variation_loss (x):
    a=K.square(
    x[:, : image_height-1, : image_width -1, :]- x[:, 1:, :image_width-1, :])
    
    
    b=K.square(x[:, :image_height -1, :image_width-1, :]- x[:, :image_height -1 , 1:, :])
    
    return K.sum(K.pow(a+b , 1.25))
outputs_dict = dict([(layer.name,layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers= ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

total_variation_weight=1e-2
style_weight=10.
content_weight=0.125
loss=K.variable(0.)
layer_features = outputs_dict [content_layer]
target_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2, :, :, :]
loss =  loss + content_weight * content_loss(target_image_features, combination_features)


for layer_name in style_layers:
    layer_features =outputs_dict[layer_name]
    style_features =layer_features[1, :, :, :]
    combination_features=layer_features [2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
    
loss =loss + total_variation_weight * total_variation_loss(combination_image)
grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])
class Evaluator (object):
    
    def __init__(self):
        self.loss_value = None
        self.grad_values= None
        
    def loss (self, x):
        assert self.loss_value is None
        x = x .reshape ((1, image_height, image_width, 3))
        outs= fetch_loss_and_grads([x])
        loss_value = outs [0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value =loss_value
        self.grad_values =grad_values
        return self.loss_value
    
    def grads (self,x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values= None
        return grad_values
   
   evaluator= Evaluator()
   import cv2
   from scipy.optimize import fmin_l_bfgs_b

import time


result = 'result'

iterations = 20

x =preprocess_image(target_image_path)
x = x.flatten()


for i in range(iterations):
    print('Iteration : ', i)
    start_time = time.time()
    s, min_val, info = fmin_l_bfgs_b(evaluator.loss,x,fprime=evaluator.grads,maxfun=20)
    
    print('Current LOSS Value', min_val)
    img = x.copy().reshape((image_height, image_width, 3))
    img = deprocess_image(img)
    fname= result + '_at_iteration_%d.png' % i
    cv2.imwrite(fname,img)
    print(fname)
    end_time=time.time()
    print('Iteration %d completed in %ds' % (i,end_time - start_time))

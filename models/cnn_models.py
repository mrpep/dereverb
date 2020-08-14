import tensorflow.keras.layers as tfkl
import tensorflow as tf

def cnn_1(N):
  #sdofjdsoifjodsifj
  layer_in = tfkl.Input(shape=[N,1])
  enc = tfkl.Conv1D(8,8,strides=1,padding='SAME',activation='relu',use_bias=False)(layer_in)
  enc = tfkl.MaxPooling1D(2)(enc)
  enc = tfkl.Conv1D(16,8,strides=1,padding='SAME',activation='relu',use_bias=False)(enc)
  enc = tfkl.MaxPooling1D(2)(enc)
  enc = tfkl.Conv1D(32,8,strides=1,padding='SAME',activation='relu',use_bias=False)(enc)
  enc = tfkl.MaxPooling1D(2)(enc)
  enc = tfkl.Conv1D(64,8,strides=1,padding='SAME',activation='relu',use_bias=False)(enc)
  dec = tfkl.UpSampling1D(2)(enc)
  dec = tfkl.Conv1D(32,8,strides=1,padding='SAME',activation='relu',use_bias=False)(dec)
  dec = tfkl.UpSampling1D(2)(dec)
  dec = tfkl.Conv1D(16,8,strides=1,padding='SAME',activation='relu',use_bias=False)(dec)
  dec = tfkl.UpSampling1D(2)(dec)
  dec = tfkl.Conv1D(8,8,strides=1,padding='SAME',activation='relu',use_bias=False)(dec)
  out = tfkl.Conv1D(1,8,strides=1,padding='SAME',use_bias=False)(dec)
  model = tf.keras.Model(inputs=[layer_in],outputs=[out])
  return model
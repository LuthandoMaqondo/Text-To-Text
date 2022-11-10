import tensorflow as tf

class Animator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer


print("Animator Loaded")
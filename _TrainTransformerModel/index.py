# !pip install -U tensorflow
# !pip install -U tensorflow-text
# !pip install -U "tensorflow-text==2.8.*"
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_text as tf_text

# import Transformer


model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)
[item for item in dir(tokenizers.en) if not item.startswith('_')]



num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
#     target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
#     dropout_rate=dropout_rate)
# transformer.summary()


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super().__init__()

#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)

#         self.warmup_steps = warmup_steps

#         def __call__(self, step):
#             step = tf.cast(step, dtype=tf.float32)
#             arg1 = tf.math.rsqrt(step)
#             arg2 = step * (self.warmup_steps ** -1.5)

#             return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')
#     loss = loss_object(label, pred)

#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask

#     loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#     return loss


# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred

#     mask = label != 0

#     match = match & mask

#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match)/tf.reduce_sum(mask)

# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
# transformer.fit(train_batches, epochs=2, validation_data=val_batches)
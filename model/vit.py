import tensorflow as tf
from tensorflow.keras import layers

class Transformer(tf.keras.Model):
    def __init__(self, input_shape, patch_size=16, projection_dim=128, num_heads=8, transformer_layers=12):
        super(Transformer, self).__init__()
        self.input_layer = layers.Input(shape=input_shape)

        # Patch embedding
        self.patches = Patches(patch_size)(self.input_layer)
        self.encoded_patches = PatchEncoder(
            num_patches=(input_shape[0] // patch_size) * (input_shape[1] // patch_size),
            projection_dim=projection_dim
        )(self.patches)

        # Transformer blocks
        for _ in range(transformer_layers):
            self.encoded_patches = transformer_encoder(self.encoded_patches, projection_dim, num_heads)

        # Reshape to feature map: [B, H/P, W/P, C]
        h, w = input_shape[0] // patch_size, input_shape[1] // patch_size
        x = layers.Reshape((h, w, projection_dim))(self.encoded_patches)

        # Upsampling with skip connections (optional)
#         x = layers.Conv2DTranspose(512, 3, strides=2, padding="same", activation="relu")(x)
#         x = layers.Dropout(0.1)(x)  # 添加Dropout
        x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Dropout(0.1)(x)  # 添加Dropout
        x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Dropout(0.1)(x)  # 添加Dropout
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Dropout(0.1)(x)  # 添加Dropout
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Dropout(0.1)(x)  # 添加Dropout

        # Output layer: single-channel mask
        output = layers.Conv2D(1, 1, activation="sigmoid")(x)

        # Build the final model
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=output)

    def call(self, inputs):
        return self.model(inputs)


# Patch extraction layer
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config


# Patch encoder (with position embedding)
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


# Transformer encoder block
def transformer_encoder(x, projection_dim, num_heads):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads,  # Adjust key_dim for better performance
        dropout=0.1
    )(x1, x1)
    x2 = layers.Add()([attention_output, x])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = layers.Dense(units=projection_dim * 2, activation="gelu")(x3)
    x3 = layers.Dropout(0.1)(x3)
    x3 = layers.Dense(units=projection_dim)(x3)
    return layers.Add()([x2, x3])


# Example usage
input_size = (256, 256, 3)
model = Transformer(input_size).model
model.summary()
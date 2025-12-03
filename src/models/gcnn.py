from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAvgPool1D, Input, Dropout, BatchNormalization
from typing import Optional, Callable
from . import edgeconv

def layer_name(prefix):
    return edgeconv.EdgeConv.layer_name(prefix)

custom_objects = {"EdgeConv": edgeconv.EdgeConv, "SplitLayer": edgeconv.SplitLayer, "mean": keras.backend.mean}


def create_model(
        n_points: int,  # number of nodes
        n_coords: int = 4,  # number of coordinates used for distance computation
        n_features: Optional[int] = None,  # number of features carried by a node
        n_hypothesis_tests: int = 1,  # number of hypothesis to be tested
        pretrained: str = '',
        # Model's architecture:
        n_conv: int = 2,
        kernel_layers: list[int] = 2 * [80],
        dense_layers: list = [],
        k_neighbors: int = 8,
        l2: float = 0.000493, l1: float = 0.000053, dropout_rate: float = 0.13,
        normalize: bool = True,
        activation: str = 'relu', output_activation: str = 'sigmoid',
        dynamic_conv: bool = True,
        # Training parameters:
        loss: str | Callable = 'binary_crossentropy',
        metrics: str = 'accuracy',
        lr: float = 0.001,
):
    # TODO: assert incompatible losses & modes
    if n_hypothesis_tests > 1:
        assert loss == 'categorical_crossentropy'

    # default values for the parameters above were obtained with optimization for n_points=100
    if n_features is None:
        n_features = n_coords
    assert n_coords <= n_features
    inline_activation = activation
    layer_activation = None
    if activation == 'prelu':
        inline_activation = 'linear'
        def layer_activation():
            return keras.layers.PReLU(name=layer_name('prelu'))

    weight_reg = None
    if l1 > 0 or l2 > 0:
        weight_reg = keras.regularizers.L1L2(l1=l1, l2=l2)

    if pretrained:
        model = keras.models.load_model(pretrained, compile=False, custom_objects=custom_objects)
    else:
        features = Input((n_points, n_features), name=layer_name('dgcnn_in'))
        if n_coords < n_features:
            points = keras.layers.Lambda(lambda x: x[:, :, :n_coords])(features)
        else:
            points = features

        norm_features = features
        if normalize:
            norm_features = BatchNormalization(name=layer_name('batch_norm'))(features)

        def EdgeConv(x):
            return edgeconv.EdgeConv(next_neighbors=k_neighbors,
                                     kernel_layers=kernel_layers,
                                     kernel_l1=l1,
                                     kernel_l2=l2,
                                     kernel_activation=activation)(x)

        out = EdgeConv([points, norm_features])
        for i in range(1, n_conv):
            if dynamic_conv:
                out = EdgeConv(out)
            else:
                out = EdgeConv([points, out])
        out = GlobalAvgPool1D()(out)
        for dim in dense_layers:
            out = Dense(dim, activation=inline_activation, kernel_regularizer=weight_reg, name=layer_name('dense'))(out)
            if layer_activation is not None:
                out = layer_activation()(out)
            if dropout_rate > 0:
                out = Dropout(dropout_rate, name=layer_name('dropout'))(out)

        out = Dense(n_hypothesis_tests, activation=output_activation,
                    kernel_regularizer=weight_reg, name=layer_name('dense'))(out)

        model = keras.models.Model(features, out)

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,
              optimizer=optimizer, metrics=[metrics])

    # tmp_model_file = '/tmp/save_test.h5'
    # model.save(tmp_model_file, overwrite=True)
    # model = keras.models.load_model(tmp_model_file, compile=True, custom_objects=custom_objects)

    return model

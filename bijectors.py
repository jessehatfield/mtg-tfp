import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def matchup_matrix_bijector():
    return tfp.bijectors.Chain([
        tfp.bijectors.TransformDiagonal(diag_bijector=tfp.bijectors.Sigmoid()),
        PadMatrix(left=1, bottom=1, constant=0.0),
        tfp.bijectors.FillTriangular(upper=True),
        tfp.bijectors.Sigmoid(),
    ])


class PadMatrix(tfp.bijectors.Bijector):
    def __init__(self, left=0, right=0, top=0, bottom=0,
                 constant=0.0, name="Pad", validate_args=False):
        super().__init__(
            validate_args=validate_args,
            is_constant_jacobian=True,
            forward_min_event_ndims=2,
            inverse_min_event_ndims=2,
            name=name)
        self.constant = constant
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.paddings = tf.constant([[top, bottom], [left, right]])

    def _forward(self, x):
        return tf.pad(x, self.paddings, constant_values=self.constant)

    def _inverse(self, y: tf.Tensor):
        height = y.get_shape()[0] - self.top - self.bottom
        width = y.get_shape()[1] - self.left - self.right
        return tf.slice(y, [self.top, self.left], [height, width])

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., dtype=y.dtype)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., dtype=x.dtype)


class MatchupMatrix(tfp.bijectors.Bijector):
    def __init__(self, name="MatchupMatrix", validate_args=False):
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=2,
            name=name)
        self.pad = PadMatrix(left=1, bottom=1, constant=0.0)
        self.fill_triangular = tfp.bijectors.FillTriangular(upper=True)
        self.sigmoid = tfp.bijectors.Sigmoid()

    def _forward(self, x):
        upper_half = self.pad.forward(self.fill_triangular.forward(x))
        mirrored = tf.subtract(upper_half, tf.transpose(upper_half))
        return self.sigmoid.forward(mirrored)

    def _inverse(self, y):
        n = int(y.get_shape()[0])
        mask = np.ones([n, n])
        mask[np.tril_indices(n)] = 0.0
        half = tf.multiply(y, mask)
        return self.sigmoid.inverse(self.fill_triangular.inverse(self.pad.inverse(half)))

    def _forward_log_det_jacobian(self, x):
        return self.sigmoid._forward_log_det_jacobian(x)

    def _inverse_log_det_jacobian(self, y):
        return self.sigmoid._inverse_log_det_jacobian(y)


if __name__ == "__main__":
    session = tf.Session()
    vector = tf.constant([1, -1, .1, -.1, 2, -2], dtype=tf.float32)
    print(session.run([vector]))
    matrix = MatchupMatrix().forward(vector)
    print(session.run([matrix]))
    reverse = MatchupMatrix().inverse(matrix)
    print(session.run([reverse]))


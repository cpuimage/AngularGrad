"""AngularGrad optimizer implementation."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, Callable, Dict

from tensorflow.keras import initializers
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import deserialize
from tensorflow.python.ops import array_ops, state_ops, control_flow_ops, cond_v2
from tensorflow.python.ops import math_ops



class AngularGrad(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            method_angle: str = 'cos',
            learning_rate: Union[float, Callable, Dict] = 1e-3,
            beta_1: Union[float, Callable] = 0.9,
            beta_2: Union[float, Callable] = 0.999,
            epsilon: float = 1e-8,
            theta_coeff: float = (180.0 / 3.141592653589793),
            min_angle: float = (3.141592653589793 / 2.0),
            name: str = 'AngularGrad',
            **kwargs, ):
        super(AngularGrad, self).__init__(name, **kwargs)
        self.method_angle = method_angle
        if isinstance(learning_rate, Dict):
            learning_rate = deserialize(learning_rate)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('theta_coeff', theta_coeff)
        self.epsilon = epsilon or backend_config.epsilon()
        self.min_angle = min_angle
        self.use_tan_theta = self.method_angle == 'tan'

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'prev_grad')
            self.add_slot(var, 'min_angle', initializer=initializers.Constant(value=self.min_angle))
            self.add_slot(var, 'theta')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AngularGrad, self)._prepare_local(var_device, var_dtype, apply_state)
        epsilon = ops.convert_to_tensor_v2(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        theta_coeff_t = array_ops.identity(self._get_hyper('theta_coeff', var_dtype))
        lr_t = apply_state[(var_device, var_dtype)]['lr_t']
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        one_minus_beta_1_t = 1.0 - beta_1_t
        one_minus_beta_2_t = 1.0 - beta_2_t
        lr = (lr_t * (math_ops.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=epsilon,
                lr=lr,
                theta_coeff_t=theta_coeff_t,
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                local_step=local_step,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_1_t=one_minus_beta_1_t,
                one_minus_beta_2_t=one_minus_beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(AngularGrad, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'] + m_scaled_g_values, use_locking=self._use_locking)
        v = self.get_slot(var, 'v')
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = math_ops.square(grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] + v_scaled_g_values, use_locking=self._use_locking)
        prev_grad = self.get_slot(var, 'prev_grad')
        tan_theta = math_ops.abs((prev_grad - grad) / (1.0 + prev_grad * grad))
        prev_grad_t = state_ops.assign(prev_grad, grad, use_locking=self._use_locking)
        updates = [prev_grad_t, m_t, v_t]
        angle = math_ops.atan(tan_theta) * coefficients['theta_coeff_t']
        min_angle = self.get_slot(var, 'min_angle')
        mean_angle = math_ops.reduce_mean(math_ops.cast(math_ops.greater(angle, min_angle), var_dtype))
        theta = self.get_slot(var, 'theta')

        def cos_theta_true_fn():
            min_angle_t = state_ops.assign(min_angle, angle, use_locking=self._use_locking)
            updates.append(min_angle_t)
            theta_t = state_ops.assign(theta, (1.0 / math_ops.sqrt(1.0 + math_ops.square(tan_theta))),
                                       use_locking=self._use_locking)
            updates.append(theta_t)
            return min_angle_t, theta_t

        def tan_theta_true_fn():
            min_angle_t = state_ops.assign(min_angle, angle, use_locking=self._use_locking)
            updates.append(min_angle_t)
            theta_t = state_ops.assign(theta, tan_theta, use_locking=self._use_locking)
            updates.append(theta_t)
            return min_angle_t, theta_t

        def false_fn():
            return min_angle, theta

        if self.use_tan_theta:
            min_angle, theta = cond_v2.cond_v2(math_ops.less(mean_angle, 0.5), tan_theta_true_fn, false_fn)
        else:
            min_angle, theta = cond_v2.cond_v2(math_ops.less(mean_angle, 0.5), cos_theta_true_fn, false_fn)
        angular_coeff = math_ops.tanh(theta) * 0.5 + 0.5
        var_t = angular_coeff * m_t / math_ops.maximum(coefficients['epsilon'], math_ops.sqrt(v_t))
        var_update = state_ops.assign_sub(var, coefficients['lr'] * var_t, use_locking=self._use_locking)
        updates.append(var_update)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'], use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        v = self.get_slot(var, 'v')
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = math_ops.square(grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'], use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        prev_grad = self.get_slot(var, 'prev_grad')
        tan_theta = math_ops.abs((prev_grad - grad) / (1.0 + prev_grad * grad))
        prev_grad_t = state_ops.assign(prev_grad, grad, use_locking=self._use_locking)
        updates = [prev_grad_t, m_t, v_t]
        angle = math_ops.atan(tan_theta) * coefficients['theta_coeff_t']
        min_angle = self.get_slot(var, 'min_angle')
        mean_angle = math_ops.reduce_mean(math_ops.cast(math_ops.greater(angle, min_angle), var_dtype))
        theta = self.get_slot(var, 'theta')

        def cos_theta_true_fn():
            min_angle_t = state_ops.assign(min_angle, angle, use_locking=self._use_locking)
            updates.append(min_angle_t)
            theta_t = state_ops.assign(theta, (1.0 / math_ops.sqrt(1.0 + math_ops.square(tan_theta))),
                                       use_locking=self._use_locking)
            updates.append(theta_t)
            return min_angle_t, theta_t

        def tan_theta_true_fn():
            min_angle_t = state_ops.assign(min_angle, angle, use_locking=self._use_locking)
            updates.append(min_angle_t)
            theta_t = state_ops.assign(theta, tan_theta, use_locking=self._use_locking)
            updates.append(theta_t)
            return min_angle_t, theta_t

        def false_fn():
            return min_angle, theta

        if self.use_tan_theta:
            min_angle, theta = cond_v2.cond_v2(math_ops.less(mean_angle, 0.5), tan_theta_true_fn, false_fn)
        else:
            min_angle, theta = cond_v2.cond_v2(math_ops.less(mean_angle, 0.5), cos_theta_true_fn, false_fn)

        angular_coeff = math_ops.tanh(theta) * 0.5 + 0.5
        var_t = angular_coeff * m_t / math_ops.maximum(coefficients['epsilon'], math_ops.sqrt(v_t))
        var_update = state_ops.assign_sub(var, coefficients['lr'] * var_t, use_locking=self._use_locking)
        updates.append(var_update)
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(AngularGrad, self).get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'beta_1': self._serialize_hyperparameter('beta_1'),
                'beta_2': self._serialize_hyperparameter('beta_2'),
                'theta_coeff': self._serialize_hyperparameter('theta_coeff'),
                'epsilon': self.epsilon,
            }
        )
        return config

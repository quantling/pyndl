
.. _comparison_of_algorithms:

Comparison with other Algorithms
================================

Rescorla Wagner learning rule
-----------------------------

In order to update the association strengths (weights) between cues and
outcomes we do for each event the following:

We calculate the activation (prediction) :math:`a_j` for each outcome
:math:`o_j` by using all present cues :math:`C_\text{PRESENT}`:

.. math::

    a_j = \sum_{i \text{ for } c_i \in C_\text{PRESENT}} w_{ij}

After that, we calculate the update :math:`\Delta w_{ij}` for every
cue-outcome-combination:

.. math::
    \Delta w_{ij}
    \begin{cases}
      0                                         & \text{if cue } c_i \text{ is absent}\\
      \alpha_i \beta_1 \cdot (\lambda - a_j )   & \text{if outcome } o_j \text{ and cue } c_i  \text{ is present.}\\
      \alpha_i \beta_2 \cdot (0 - a_j )         & \text{if outcome } o_j \text{ is absent and cue } c_i  \text{ is present.}
    \end{cases}

In the end, we update all weights according to :math:`w_{ij} = w_{ij} + \Delta
w_{ij}`.

Usually, we set :math:`\lambda = 1`, :math:`\beta_1 = \beta_2 = \beta = 0.01`,
and :math:`\alpha_i = \alpha = 0.01`.

If we set all the :math:`\alpha`'s and :math:`\beta`'s to a fixed value we can
replace them in the equation with a general learning parameter :math:`\eta =
\alpha \cdot \beta`.


In matrix notation
^^^^^^^^^^^^^^^^^^
We can rewrite the Rescorla-Wagner learning rule into matrix notation with a
binary cue (input) vector :math:`\vec{c}`, which is one for each cue present in
the event and zero for all other cues. Respectively, we define a binary outcome
(output) vector :math:`\vec{o}`, which is one for each outcome present in the
event and zero if the outcome is not present. In order to stick close to the
definition above we can define the activation vector as :math:`\vec{a} = W^T
\vec{c}`. Here :math:`W^T` denotes the transposed matrix of the weight matrix
:math:`W`.

For simplicity let us assume we have a fixed learning rate :math:`\eta = \alpha
\beta`. We will relax this simplification in the end. We can rewrite the above
rule as:

.. math::

   \Delta &= \eta \vec{c} \cdot (\lambda \vec{o} - \vec{a})^T \\
   &= \eta \vec{c} \cdot (\lambda \vec{o} - W^T \cdot \vec{c})^T

Let us first check the dimensionality of the matrices:

:math:`\Delta` is the update of the weight matrix :math:`W` and therefore needs
to have the same dimensions :math:`n \times m` where :math:`n` denotes the
number of cues (inputs) and :math:`m` denotes the number of outcomes (outputs).

The cue vector :math:`\vec{c}` can be seen as a matrix with dimensions :math:`n
\times 1` and the outcome vector can be seen as a matrix with dimensions
:math:`m \times 1`. Let us tabulate the dimensions:

================================================================  ====================================================
:math:`\lambda \vec{o}`                                           :math:`m \times 1`
:math:`W^T`                                                       :math:`m \times n`
:math:`\vec{c}`                                                   :math:`n \times 1`
:math:`W^T \cdot \vec{c}`                                         :math:`m \times 1 = (m \times n) \cdot (n \times 1)`
:math:`\lambda \vec{o} - W^T \cdot \vec{c}`                       :math:`m \times 1 = (m \times 1) - (m \times 1)`
:math:`(\lambda \vec{o} - W^T \cdot \vec{c})^T`                   :math:`1 \times m = (m \times 1)^T`
:math:`\eta \vec{c} \cdot (\lambda \vec{o} - W^T \cdot \vec{c})`  :math:`n \times m = (n \times 1) \cdot (1 \times n)`
================================================================  ====================================================

We therefore end with the right set of dimensions. We now can try to simplify /
rewrite the equation.

.. math::
   \Delta &= \eta \vec{c} \cdot ((\lambda \vec{o})^T - (W^T \cdot \vec{c})^T) \\
   &= \eta \vec{c} \cdot (\lambda \vec{o}^T - \vec{c}^T \cdot W) \\
   &= \eta \lambda \vec{c} \cdot \vec{o}^T - \eta \vec{c} \cdot \vec{c}^T \cdot W \\

If we now look at the full update:

.. math::

   W_{t + 1} &= W_t + \Delta_t \\
   &= W + \Delta \\
   &= W + \eta \lambda \vec{c} \cdot \vec{o}^T - \eta \vec{c} \cdot \vec{c}^T
   \cdot W \\
   &= \eta \lambda \vec{c} \cdot \vec{o}^T + W - \eta \vec{c} \cdot \vec{c}^T
   \cdot W \\
   &= \eta \lambda \vec{c} \cdot \vec{o}^T + (1 - \eta \vec{c} \cdot \vec{c}^T)
   \cdot W \\

We therefore see that the Rescorla-Wagner update is an affine (linear)
transformation [affine_transformation]_ in the weights :math:`W` with an
intercept of :math:`\eta
\lambda \vec{c} \cdot \vec{o}^T` and a slope of :math:`(1 - \eta \vec{c} \cdot
\vec{c}^T)`.

In index notation we can write:

.. math::


   W^{t + 1} &= W^{t} + \eta \vec{c} \cdot (\lambda \vec{o}^T - \vec{c}^T \cdot W) \\
   W^{t + 1}_{ij} &= W^{t}_{ij} + \eta c_i (\lambda o_j - \sum_k c_k W_{kj}) \\


.. note::

   Properties of the transpose [transpose]_ with :math:`A` and :math:`B`
   matrices and :math:`\alpha` skalar:

   .. math::
      (A^T)^T = A

   .. math::
      (A + B)^T = A^T + B^T

   .. math::
      (\alpha A)^T = \alpha A^T

   .. math::
      (A \cdot B)^T = B^T \cdot A^T



Delta rule
----------

   The delta rule is a gradient descent learning rule for updating the weights
   of the inputs to artificial neurons in a single-layer neural network. It is
   a special case of the more general backpropagation algorithm. [delta_rule]_

The delta rule can be expressed as [delta_rule]_:

.. math::

   \Delta_{ij} = \alpha (t_j - y_j) \partial_{h_j} g(h_j) x_i

In the terminology above we can identify the actual output with :math:`y_j =
g(h_j) = g\left(\sum_i w_{ij} c_i\right)`, the cues with :math:`x_i = c_i`, under the
assumption that :math:`o_j` is binary (i. e. either zero or one) we can write
:math:`t_j = \lambda o_j`, the learning rate :math:`\alpha = \eta = \alpha
\beta`.  Substituting this equalities results in:

.. math::

   \Delta_{ij} = \eta (\lambda o_j - g\left(\sum_i w_{ij} c_i\right)) \partial_{h_j} g(h_j) c_i

In order to end with the Rescorla-Wagner learning rule we need to set the
neuron's activation function :math:`g(h_j)` to the identity function, i. e.
:math:`g(h_j) = 1 \cdot h_j + 0 = h_j = \sum_i w_{ij} c_i`. The derivative in respect
to :math:`h_j` is :math:`\partial_{h_j} g(h_j) = 1` for any input :math:`h_j`.

We now have:

.. math::

   \Delta_{ij} &= \eta (\lambda o_j - \sum_i w_{ij} c_i) \cdot 1 \cdot c_i \\
   &= \eta (\lambda o_j - \sum_i w_{ij} c_i) c_i \\
   &= \eta c_i (\lambda o_j - \sum_i w_{ij} c_i)

Assuming the cue vector is binary the vector :math:`c_i` effectively filters
those updates of the present cues and sets all updates of the cues that are not
present to zero. Additionally, we can rewrite the equation above into vector
notation (without indices):

.. math::

   \Delta_{ij} &= \eta c_i (\lambda o_j - \sum_i w_{ij} c_i) \\
   &= \eta c_i (\lambda o_j - \sum_i w_{ij} c_i)

.. math::

   \Delta = \eta \vec{c} \cdot (\lambda \vec{o}^T - W^T \cdot \vec{c})^T

This is exactly the form of the Rescorla-Wagner rule rewritten in matrix
notation.

.. admonition:: Conclusion

   In conclusion, the Rescorla-Wagner learning rule, which only allows for one
   :math:`\alpha` and one :math:`\beta` and therefore one learning rate
   :math:`\eta = \alpha \beta` is exactly the same as a single layer
   backpropagation gradient decent method (the delta rule) where the neuron's
   activation function :math:`g(h_j)` is set to the identity :math:`g(h_j) =
   h_j` and the inputs :math:`x_i = c_i` and target outputs :math:`t_j =
   \lambda o_j` to be binary.


Kalman filter
-------------

.. warning::

   This section is still under construction.


According to Dayan & Kakade [dayan_explaining_away_in_weight_space]_ one can
write a simplified version of the Kalman filter as:

.. math::

   r_t = \vec{w}^T_t \cdot \vec{x}_t + \epsilon_t

Here :math:`\vec{w}^T_t` are the true weights mediating between the presented stimuli
:math:`\vec{x}_t` and the scalar reward :math:`r_t` at time :math:`t`. The last
term :math:`\epsilon_t` is zero mean Gaussian noise with variance
:math:`\tau^2`, i. e. :math:`\epsilon_t \sim N(0, \tau^2)`.

We want to allow for a change in the true weights :math:`\vec{w}^T_t` over time.
Therefore we need the additional diffusion term for the propagation of the
weights:

.. math::

   \vec{w}_{t + 1} = \vec{w}_t + \vec{\eta}_t

where :math:`\vec{\eta}_t \sim N(\vec{0}, \sigma^2 I)` is a multivariate Gaussian.

As we do not know the true values for :math:`\vec{w}_t`, we need to infer them
from observations for each trial :math:`t` of stimuli (cues) :math:`\vec{x}_t`
and the reward (outcome) :math:`r_t`. According to
[dayan_explaining_away_in_weight_space]_ one way to infer / estimate the
distribution of the association vector :math:`\Pr(\vec{w}_t | r_1, \cdot, r_{t
- 1}) \sim N(\hat{\vec{w}}, S_t)` is:

.. math::

   \hat{\vec{w}}_{t + 1} = \hat{\vec{w}}_t + \frac{ S_t \cdot
   \vec{x}_t}{\vec{x}_t \cdot S_t \vec{x}_t + \tau^2} (r_t - \hat{\vec{w}}_t
   \cdot \vec{x}_t)

.. math::

   S_{t + 1} = S_t + \sigma^2 I - \frac{S_t \cdot \vec{x}_t \cdot \vec{x}_t^T
   \cdot S_t}{\vec{x}^T_t \cdot S_t \cdot \vec{x}_t + \tau^2}


Comparison to Rescorla-Wagner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Problems:

* equations above only for one reward / outcome not for a vector of rewards /
  outcomes

We can make the following identifications:

=================  =============================================
Kalman             Rescorla-Wagner
=================  =============================================
:math:`\vec{x}_t`  :math:`\vec{c}`
:math:`r_t`        :math:`\lambda o_j` for one outcome :math:`j`
:math:`\vec{w}_t`  :math:`(w_{ij})` for one outcome :math:`j`
=================  =============================================

We can rewrite the update of :math:`\hat{\vec{w}}_{t + 1}` as:

.. math::

   W_{ij}^{t + 1} = W_{ij}^{t} + \frac{\sum_k S_{ik}^{j, t} c_k^t}{\sum_l
   \sum_k c_k^t S_{kl}^{j, t} c_l^t + \tau^2} (o_j^t - \sum_k W_{kj}^t c_k^t)

where :math:`S^{j, t}` is the covariance matrix for outcome :math:`j` at trial
/ event :math:`t`. We wrote the trial / event index as a superscript and will
omit it in the following for all events :math:`t`.

If we set the covariance matrix for all outcomes to the identity matrix we get:

.. math::

   W_{ij}^{t + 1} &= W_{ij} + \frac{\sum_k I_{ik} c_k}{\sum_l \sum_k c_k I_{kl}
   c_l + \tau^2} (\lambda o_j - \sum_k W_{kj} c_k) \\
   &= W_{ij} + \frac{c_i}{\sum_k c_k c_k + \tau^2} (\lambda o_j - \sum_k W_{kj} c_k) \\
   &= W_{ij} + \frac{1}{\sum_k c_k c_k + \tau^2} c_i (\lambda o_j - \sum_k W_{kj} c_k) \\
   &= W_{ij} + \eta^t c_i (\lambda o_j - \sum_k W_{kj} c_k) \\

where we have a variable learning rate which is smaller for events with many
cues and larger for events with few cues:

.. math::

   \eta^t = \frac{1}{\sum_k c_k^t c_k^t + \tau^2}

Note that :math:`\sum_k c_k^t c_k^t` is the number of cues in event :math:`t`.

.. admonition:: Conclusion

   Except for the variable learning rate the equation is identical to the
   Rescorla Wagner learning rule. If we set the variance-covariance matrix of
   the distribution of the association vector :math:`\vec{w}`, which is assumed
   to be multinomial, to the identity matrix. Furthermore, we need to assume
   that we have only binary stimuli / cues / inputs and a binary reward.

.. warning::

   Did I made somewhere some error? Is this sound? --Tino


Useful Links for Kalman filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* https://math.stackexchange.com/questions/840662/an-explanation-of-the-kalman-filter


References
----------

.. [affine_transformation] Affine transformation. https://en.wikipedia.org/wiki/Affine_transformation

.. [transpose] Transpose. https://en.wikipedia.org/wiki/Transpose

.. [delta_rule] Delta rule. https://en.wikipedia.org/wiki/Delta_rule

.. [dayan_explaining_away_in_weight_space] https://homes.cs.washington.edu/~sham/papers/neuro/kd_weight.pdf

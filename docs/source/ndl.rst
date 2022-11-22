Background
==========

Naive Discriminative Learning
-----------------------------

Terminology
~~~~~~~~~~~

Before explaining Naive Discriminative Learning (NDL) in detail, we want to
give you a brief overview over important notions:

cue :
    A cue is something that gives a hint on something else. The something else
    is called outcome. Examples for cues in a text corpus are trigraphs or
    preceding words for the word or meaning of the word.

outcome :
    The outcome is the result of an event. Examples are words, the meaning of
    the word, or lexomes.

event :
    An event connects cues with outcomes. In any event one or more unordered
    cues are present and one or more outcomes are present.

weights :
    The weights represent the learned experience / association between all cues
    and outcomes of interest. Usually, some meta data is stored alongside the
    learned weights.


Rescorla Wagner learning rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. note::

    If we set all the :math:`\alpha`'s and :math:`\beta`'s to a fixed value we
    can replace them in the equation with a general learning parameter
    :math:`\eta = \alpha \cdot \beta`.


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
transformation [1]_ in the weights :math:`W` with an
intercept of :math:`\eta
\lambda \vec{c} \cdot \vec{o}^T` and a slope of :math:`(1 - \eta \vec{c} \cdot
\vec{c}^T)`.

In index notation we can write:

.. math::


   W^{t + 1} &= W^{t} + \eta \vec{c} \cdot (\lambda \vec{o}^T - \vec{c}^T \cdot W) \\
   W^{t + 1}_{ij} &= W^{t}_{ij} + \eta c_i (\lambda o_j - \sum_k c_k W_{kj}) \\


.. note::

   Properties of the transpose [4]_ with :math:`A` and :math:`B`
   matrices and :math:`\alpha` skalar:

   .. math::
      (A^T)^T = A

   .. math::
      (A + B)^T = A^T + B^T

   .. math::
      (\alpha A)^T = \alpha A^T

   .. math::
      (A \cdot B)^T = B^T \cdot A^T


Other Learning Algorithms
-------------------------
.. _comparison_of_algorithms:

Delta rule
^^^^^^^^^^

The delta rule [2]_ is a gradient descent learning rule for updating the weights
of the inputs to artificial neurons in a single-layer neural network. It is
a special case of the more general backpropagation algorithm [3]_.

The delta rule can be expressed as:

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


References
----------

.. [1] https://en.wikipedia.org/wiki/Affine_transformation

.. [2] https://en.wikipedia.org/wiki/Delta_rule

.. [3] https://en.wikipedia.org/wiki/Backpropagation

.. [4] https://en.wikipedia.org/wiki/Transpose

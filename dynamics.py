from math import *
import casadi
from casadi import *

opti = casadi.Opti()

linear_states=opti.variable(6)
angular_states=opti.variable(6)
n_states=12
states=vertcat(linear_states, angular_states)


control_states=opti.variable(4)

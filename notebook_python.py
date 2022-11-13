from math import *
import casadi
import numpy as np

# %%
opti = casadi.Opti()
dt = 0.2
N = 20

linear_states = opti.variable(6)
angular_states = opti.variable(6)
n_states = 12
states = casadi.vertcat(linear_states, angular_states)

control_states = opti.variable(4)
n_controls = 4
controls = casadi.vertcat(control_states)

g = 9.81
m = 5
Ixx = 10
Iyy = 10
Izz = 5
d1x = 0.3
d2x = d1x
d3x = d1x
d4x = d1x
d1y = d1x
d2y = d1x
d3y = d1x
d4y = d1x

Fz = (control_states[0] + control_states[1] + control_states[2] + control_states[3])
L = control_states[0] * d1y - control_states[1] * d2y - control_states[2] * d3y + control_states[3] * d4y
M = control_states[0] * d1x - control_states[1] * d2x + control_states[2] * d3x - control_states[3] * d4x

l = 0.42

Nz = -control_states[0] * l - control_states[1] * l + control_states[2] * l + control_states[3] * l

# %%

dx = cos(angular_states[1]) * cos(angular_states[2]) * linear_states[3] + (
        -cos(angular_states[0]) * sin(angular_states[2]) + sin(angular_states[0]) * sin(angular_states[1]) * cos(
    angular_states[2])) * linear_states[4] + (
             sin(angular_states[0]) * sin(angular_states[2]) + cos(angular_states[0]) * sin(
         angular_states[1]) * cos(angular_states[2])) * linear_states[5]
dy = cos(angular_states[1]) * sin(angular_states[2]) * linear_states[3] + (
        cos(angular_states[0]) * cos(angular_states[2]) + sin(angular_states[0]) * sin(angular_states[1]) * sin(
    angular_states[2])) * linear_states[4] + (
             -sin(angular_states[0]) * cos(angular_states[2]) + cos(angular_states[0]) * sin(
         angular_states[1]) * sin(angular_states[2])) * linear_states[5]
dz = -1 * (-sin(angular_states[1]) * linear_states[3] + sin(angular_states[0]) * cos(angular_states[1]) * linear_states[
    4] + cos(angular_states[0]) * cos(angular_states[1]) * linear_states[5])
du = -g * sin(angular_states[1]) - angular_states[5] * linear_states[4] - angular_states[4] * linear_states[5]
dv = g * sin(angular_states[0]) * cos(angular_states[1]) - angular_states[5] * linear_states[3] + angular_states[3] * \
     linear_states[5]
dw = (1 / m) * (-Fz) + g * cos(angular_states[0]) * cos(angular_states[1]) + angular_states[4] * linear_states[3] - \
     angular_states[3] * linear_states[4]
dphi = angular_states[3] + (
        angular_states[4] * sin(angular_states[0]) + angular_states[5] * cos(angular_states[0])) * tan(
    angular_states[1])
dtheta = angular_states[4] * cos(0) - angular_states[5] * sin(angular_states[0])
dpsi = (angular_states[4] * sin(angular_states[0]) + angular_states[5] * cos(angular_states[0])) * 1 / (
    cos(angular_states[1]))
dp = (1 / Ixx) * (L + (Iyy - Izz) * angular_states[4] * angular_states[5])
dq = (1 / Iyy) * (M + (Izz - Ixx) * angular_states[3] * angular_states[5])
dr = (1 / Izz) * (Nz + (Ixx - Iyy) * angular_states[3] * angular_states[4])

rhs = casadi.vertcat(dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi, dp, dq, dr)

f = casadi.Function('f', [states, controls], [rhs])  # Non-linear mapping function

# %%
# Define state, control and parameter matrices
X = opti.variable(n_states, N + 1)
U = opti.variable(n_controls, N)
P = opti.parameter(2 * n_states)
# States Integration
X[:, 0] = P[0:n_states]  # Initial State
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f(st, con)
    st_next = st + dt * f_value
    # X[:,k+1]=st_next
    opti.subject_to(X[:, k + 1] == st_next)

# %%
obj = 0

# Defining weighing matrices
Q = np.eye(n_states, dtype=float)
Q = Q * 0.5

R = np.eye(n_controls, dtype=float)
R = R * 0.5

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + casadi.mtimes(casadi.mtimes((st - P[n_states:2 * n_states]).T, Q), (st - P[n_states:2 * n_states])) + casadi.mtimes(
        casadi.mtimes(con.T, R), con)

opti.minimize(obj)

# %%
max_vel = 2
opti.subject_to(X[3, :] <= max_vel)
opti.subject_to(X[4, :] <= max_vel)
opti.subject_to(X[5, :] <= max_vel)

opti.subject_to(X[3, :] >= -max_vel)
opti.subject_to(X[4, :] >= -max_vel)
opti.subject_to(X[5, :] >= -max_vel)


opti.solver('ipopt')

# %%
x0 = casadi.vertcat(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
xs = casadi.vertcat(10, 10, 10, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)

# %%
count = 0
u0 = np.zeros((n_controls, N))
while np.linalg.norm((xs - x0)) > 1e-2 and count < 10:
    opti.set_value(P, casadi.vertcat(x0, xs))
    opti.set_initial(U, u0)
    sol = opti.solve()
    u = sol.value(U)
    # print(u[:,0])
    x = sol.value(X)
    # print(x)

    st = x0
    con = u[:, 0]
    f_value = f(st, con)
    x0 = st + dt * f_value
    print(x0)

    u0 = u

    count = count + 1

# %%

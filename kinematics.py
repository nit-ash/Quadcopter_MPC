from casadi import *
import numpy as np

opti = Opti()
dt = 0.2
N = 20

linear_states = opti.variable(3)
angular_states = opti.variable(3)
n_states = 6
states = vertcat(linear_states, angular_states)

control_states = opti.variable(6)
n_controls = 6
controls = vertcat(control_states)

dx = cos(angular_states[1]) * cos(angular_states[2]) * control_states[0] + (
            -cos(angular_states[0]) * sin(angular_states[2]) + sin(angular_states[0]) * sin(angular_states[1]) * cos(
        angular_states[2])) * control_states[1] + (
                 sin(angular_states[0]) * sin(angular_states[2]) + cos(angular_states[0]) * sin(
             angular_states[1]) * cos(angular_states[2])) * control_states[2]
dy = cos(angular_states[1]) * sin(angular_states[2]) * control_states[0] + (
            cos(angular_states[0]) * cos(angular_states[2]) + sin(angular_states[0]) * sin(angular_states[1]) * sin(
        angular_states[2])) * control_states[1] + (
                 -sin(angular_states[0]) * cos(angular_states[2]) + cos(angular_states[0]) * sin(
             angular_states[1]) * sin(angular_states[2])) * control_states[2]
dz = -1 * (-sin(angular_states[1]) * control_states[0] + sin(angular_states[0]) * cos(angular_states[1]) *
           control_states[1] + cos(angular_states[0]) * cos(angular_states[1]) * control_states[2])
dphi = control_states[3] + (
            control_states[4] * sin(angular_states[0]) + control_states[5] * cos(angular_states[0])) * tan(
    angular_states[1])
dtheta = control_states[4] * cos(0) - control_states[5] * sin(angular_states[0])
dpsi = (control_states[4] * sin(angular_states[0]) + control_states[5] * cos(angular_states[0])) * 1 / (
    cos(angular_states[1]))

rhs = vertcat(dx, dy, dz, dphi, dtheta, dpsi)

f = Function('f', [states, controls], [rhs])  # Non-linear mapping function

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
    obj = obj + mtimes(mtimes((st - P[n_states:2 * n_states]).T, Q), (st - P[n_states:2 * n_states])) + mtimes(
        mtimes(con.T, R), con)

opti.minimize(obj)

# %%
max_vel = 2
opti.subject_to(U[0, :] <= max_vel)
opti.subject_to(U[1, :] <= max_vel)
opti.subject_to(U[1, :] <= max_vel)

opti.subject_to(U[0, :] >= -max_vel)
opti.subject_to(U[1, :] >= -max_vel)
opti.subject_to(U[2, :] >= -max_vel)

max_ang = 0.5
opti.subject_to(U[3, :] <= max_ang)
opti.subject_to(U[4, :] <= max_ang)
opti.subject_to(U[5, :] <= max_ang)

opti.subject_to(U[3, :] >= -max_ang)
opti.subject_to(U[4, :] >= -max_ang)
opti.subject_to(U[5, :] >= -max_ang)

# for k in range(N+1):
#     opti.subject_to(sqrt((X[0,k]-5)**2+(X[1,k]-5)**2)>2)

opti.solver('ipopt')

# %%
x0 = vertcat(0, 0, 0, 0, 0, 0)
xs = vertcat(10, 10, 10, 0, 0, 0)

# %%
count = 0
u0 = np.zeros((n_controls, N))
while np.linalg.norm((xs - x0)) > 1e-2 and count < 5:
    opti.set_value(P, vertcat(x0, xs))
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
    print(np.shape(u))

    u0 = u

    count = count + 1

# %%

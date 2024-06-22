import numpy as np
import matplotlib.pyplot as plt

L = 1
m = 200
# m = 100
m1 = m + 1
Re = 1000
# Re = 400
delta_x = L / m
a = 0.1
# a = 0.5
delta_t = a * delta_x

p = np.zeros((m1, m1)) 
u = np.zeros((m1, m1 + 1))
v = np.zeros((m1 + 1, m1))

u[m1 - 1, :] = np.ones(m1 + 1)

iterlimit_all = 100000
iterlimit_p = 10000
errlimit_p = 1e-3
errlimit_d = 1e-9
err_upbound = 1

u_star = u.copy()
v_star = v.copy()

for iter in range(iterlimit_all):
    for i in range(1, m):
        for j in range(1, m1):
            u_star[i, j] = u[i, j] - delta_t * (
                (u[i, j + 1] ** 2 - u[i, j - 1] ** 2) / (2 * delta_x) +
                (u[i + 1, j] * (v[i + 1, j - 1] + v[i + 1, j]) - u[i - 1, j] * (v[i, j - 1] + v[i, j])) / (4 * delta_x)
            ) + delta_t / Re * (
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / (delta_x ** 2) +
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (delta_x ** 2)
            ) - (p[i, j] - p[i, j - 1]) * (delta_t / delta_x)
            
            v_star[j, i] = v[j, i] - delta_t * (
                (v[j + 1, i] ** 2 - v[j - 1, i] ** 2) / (2 * delta_x) +
                (v[j, i + 1] * (u[j - 1, i + 1] + u[j, i + 1]) - v[j, i - 1] * (u[j - 1, i] + u[j, i])) / (4 * delta_x)
            ) + delta_t / Re * (
                (v[j, i + 1] - 2 * v[j, i] + v[j, i - 1]) / (delta_x ** 2) +
                (v[j + 1, i] - 2 * v[j, i] + v[j - 1, i]) / (delta_x ** 2)
            ) - (p[j, i] - p[j - 1, i]) * (delta_t / delta_x)

    # 边界条件
    u_star[:, 0] = -u_star[:, 1]
    u_star[:, m1] = -u_star[:, m1 - 1]
    v_star[0, :] = -v_star[1, :]
    v_star[m1, :] = -v_star[m1 - 1, :]

    p_iter = np.zeros((m1, m1))
    p_iter1 = p_iter.copy()
    err_p = 0

    for iter_p in range(iterlimit_p):
        for i in range(1, m):
            for j in range(1, m):
                p_iter1[i, j] = 1 / 4 * (
                    p_iter[i + 1, j] + p_iter[i, j + 1] + p_iter1[i - 1, j] + p_iter1[i, j - 1] -
                    delta_x / delta_t * (u_star[i, j + 1] - u_star[i, j] + v_star[i + 1, j] - v_star[i, j])
                )

        p_iter1[0, :] = p_iter1[1, :]
        p_iter1[:, 0] = p_iter1[:, 1]
        p_iter1[m1 - 1, :] = p_iter1[m1 - 2, :]
        p_iter1[:, m1 - 1] = p_iter1[:, m1 - 2]
        err_p = np.max(np.abs(p_iter1 - p_iter))
        p_iter = p_iter1.copy()
        if err_p < errlimit_p:
            break

    p = p + 0.1 * p_iter
    p[0, :] = p[1, :]
    p[:, 0] = p[:, 1]
    p[m1 - 1, :] = p[m1 - 2, :]
    p[:, m1 - 1] = p[:, m1 - 2]

    for i in range(1, m1 - 1):
        for j in range(1, m1):
            u[i, j] = u_star[i, j] - delta_t / delta_x * (p_iter[i, j] - p_iter[i, j - 1])
            v[j, i] = v_star[j, i] - delta_t / delta_x * (p_iter[j, i] - p_iter[j - 1, i])

    u[:, 0] = -u[:, 1]
    u[:, m1] = -u[:, m1 - 1]
    v[0, :] = -v[1, :]
    v[m1, :] = -v[m1 - 1, :]

    for i in range(1, m):
        for j in range(1, m1):
            u_star[i, j] = u[i, j] - delta_t * (
                (u[i, j + 1] ** 2 - u[i, j - 1] ** 2) / (2 * delta_x) +
                (u[i + 1, j] * (v[i + 1, j - 1] + v[i + 1, j]) - u[i - 1, j] * (v[i, j - 1] + v[i, j])) / (4 * delta_x)
            ) + delta_t / Re * (
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / (delta_x ** 2) +
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (delta_x ** 2)
            ) - (p[i, j] - p[i, j - 1]) * (delta_t / delta_x)
            
            v_star[j, i] = v[j, i] - delta_t * (
                (v[j + 1, i] ** 2 - v[j - 1, i] ** 2) / (2 * delta_x) +
                (v[j, i + 1] * (u[j - 1, i + 1] + u[j, i + 1]) - v[j, i - 1] * (u[j - 1, i] + u[j, i])) / (4 * delta_x)
            ) + delta_t / Re * (
                (v[j, i + 1] - 2 * v[j, i] + v[j, i - 1]) / (delta_x ** 2) +
                (v[j + 1, i] - 2 * v[j, i] + v[j - 1, i]) / (delta_x ** 2)
            ) - (p[j, i] - p[j - 1, i]) * (delta_t / delta_x)

    # 边界条件
    u_star[:, 0] = -u_star[:, 1]
    u_star[:, m1] = -u_star[:, m1 - 1]
    v_star[0, :] = -v_star[1, :]
    v_star[m1, :] = -v_star[m1 - 1, :]

    err_d = np.max(np.abs(u - u_star))
    print(f'iter = {iter};  err_u = {err_d};  err_p = {err_p}')
    if err_d > err_upbound:
        break
    if err_d < errlimit_d:
        break

    u = u_star.copy()
    v = v_star.copy()

u_plot = (u_star[:, :m1] + u_star[:, 1:m1 + 1]) / 2
v_plot = (v_star[:m1, :] + v_star[1:m1 + 1, :]) / 2

xaxis = np.arange(0, 1 + delta_x, delta_x)
x, y = np.meshgrid(xaxis, xaxis)

plt.figure(1)
plt.plot(x[0, :], y[0, :], 'k')
plt.plot(x[:, 0], y[:, 0], 'k')
plt.plot(x[:, m1 - 1], y[:, m1 - 1], 'k')
plt.plot(x[m1 - 1, :], y[m1 - 1, :], 'k')
plt.streamplot(x, y, u_plot, v_plot, density=3)
plt.title('Streamlines')
plt.axis('equal')
plt.savefig('Problem3/streamlines.png')

plt.figure(2)
plt.contourf(x, y, np.sqrt(u_plot ** 2 + v_plot ** 2), 10)
plt.colorbar()
plt.contour(x, y, np.sqrt(u_plot ** 2 + v_plot ** 2), 10, colors='black', linewidths=0.5)
plt.axis('equal')
plt.title('Velocity Contour')
plt.savefig('Problem3/velocity_contour.png')

plt.figure(3)
plt.pcolor(x, y, np.sqrt(u_plot ** 2 + v_plot ** 2), cmap='jet', shading='auto')
plt.colorbar()
plt.contour(x, y, np.sqrt(u_plot ** 2 + v_plot ** 2), colors='black', linewidths=0.5)
plt.axis('equal')
plt.title('Velocity Pcolor')
plt.savefig('Problem3/velocity_pcolor.png')

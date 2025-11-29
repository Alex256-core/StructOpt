import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock(x):
    a, b = 1.0, 100.0
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    a, b = 1.0, 100.0
    g1 = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    g2 =  2*b*(x[1] - x[0]**2)
    return np.array([g1, g2])

def run_structopt(theta0, steps=600, lr=1e-3, eps=1e-8,
                  S_ref=1e-3, a0=0.0, a1=6.0):
    theta = theta0.copy()
    theta_prev = theta.copy()
    g_prev = grad_rosenbrock(theta)

    m = np.zeros_like(theta)
    losses = []
    S_vals = []
    traj = []

    for _ in range(steps):
        g = grad_rosenbrock(theta)

        # simple diagonal normalizer
        m = 0.99*m + 0.01*(g*g)
        d = g / (np.sqrt(m) + eps)

        # structural signal
        S_t = np.linalg.norm(g - g_prev) / \
              (np.linalg.norm(theta - theta_prev) + eps)
        S_vals.append(S_t)

        # adaptive mixing
        z = a0 + a1 * ((S_t - S_ref) / (S_ref + eps))
        alpha = 1 / (1 + np.exp(-z))

        # update
        step = -lr * (alpha * d + (1 - alpha) * g)

        theta_prev = theta
        g_prev = g
        theta = theta + step

        losses.append(rosenbrock(theta))
        traj.append(theta.copy())

    return np.array(losses), np.array(S_vals), np.array(traj)


if __name__ == "__main__":
    theta0 = np.array([-1.5, 2.0])
    losses, S_vals, traj = run_structopt(theta0)

    # LOSS CURVE
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png", dpi=200)

    # SIGNAL CURVE
    plt.figure()
    plt.plot(S_vals)
    plt.title("Structural Signal S(t)")
    plt.savefig("signal_curve.png", dpi=200)

    # TRAJECTORY
    plt.figure()
    plt.plot(traj[:,0], traj[:,1])
    plt.title("Trajectory")
    plt.savefig("trajectory.png", dpi=200)

#/usr/bin/env python 

import numpy as np 
import matplotlib.pyplot as plt 

import deepxde as dde 
import scipy
import imageio.v2 as imageio

EPS = 1e-4 

class Isotropic: 

    @staticmethod
    def pde_functor(alpha): 
        def pde(x, u):
            ut = dde.grad.jacobian(u, x, i=0, j=3)
            uxx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            uyy = dde.grad.hessian(u, x, component=0, i=1, j=1)
            uzz = dde.grad.hessian(u, x, component=0, i=2, j=2)
            return ut - alpha * (uxx + uyy + uzz)  
        return pde 

    @staticmethod 
    def boundary_value(x): 
        return np.zeros((len(x), 1))

    @staticmethod 
    def initial_value(x): 
        return np.sin(x[:, 0:1]) * np.sin(x[:, 1:2]) * np.sin(x[:, 2:3]) 


    def __init__(self, layers, interior_nodes, pinn_nodes, alpha, load: str | None =None, dt=1e-3, opt="adam", lr=1e-3, metrics=["l2 relative error"]): 
        self.interior_nodes = interior_nodes
        self.pinn_nodes     = pinn_nodes 
        self.alpha = alpha 
        self.dt = dt

        if len(self.pinn_nodes) != 3: 
            raise ValueError("incorrect shape for PINN interior, boundary, and initial node counts")
        # This needs to be a list of these 
        self.numeric_solution = None

        # Initialize PINN, Guess tfinal = 10.0 
        geom       = dde.geometry.Cuboid([0, 0, 0], [np.pi, np.pi, np.pi])
        timedomain = dde.geometry.TimeDomain(0, 10)
        geomtime   = dde.geometry.GeometryXTime(geom, timedomain)  
        initials   = dde.IC(
            geomtime, 
            self.initial_value, 
            lambda x, on_initial: on_initial, 
            component=0
        ) 
        boundaries = dde.DirichletBC(
            geomtime, 
            self.boundary_value, 
            lambda x, on_boundary: on_boundary, 
            component=0
        )

        def exact(X): 
            x, y, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
            return self.analytic(t, x, y, z)

        pde = Isotropic.pde_functor(self.alpha)
        data = dde.data.TimePDE(
            geomtime, 
            pde, 
            [boundaries, initials], 
            num_domain=self.pinn_nodes[0], 
            num_boundary=self.pinn_nodes[1], 
            num_initial=self.pinn_nodes[2],
            solution=exact
        )

        # Apparently the LSP shouldn't be throwing an error at this? 
        network = dde.nn.pytorch.FNN(layers, "tanh", "Glorot normal")
        self.pinn = dde.Model(data, network)
        self.pinn.compile(opt, lr=lr, metrics=metrics)

        if load is not None: 
            self.pinn.restore(load, device="cuda:0", verbose=1)

    def train(self, epochs=2500, prefix="iso"):
        losshistory, train_state = self.pinn.train(iterations=epochs)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        path = self.pinn.save(prefix, verbose=1)
        print(f"saved model to {path}")
        return path


    def analytic(self, t, x, y, z):
        return np.exp(-3.0 * self.alpha * t) * np.sin(x) * np.sin(y) * np.sin(z) 

    
    def mask_diagonals_(self, N, M, r, main): 
        """
        M is N**3, N is the number of interior nodes along one axis, all three axis
        have the same number of nodes. 
        main is the main diagonal which gets packaged into the list 
        """
        rx, ry, rz = r

        xright = np.zeros(M, dtype=np.float64)
        xleft  = np.zeros(M, dtype=np.float64)
        yup   = np.zeros(M, dtype=np.float64)
        ydown = np.zeros(M, dtype=np.float64)
        zup   = np.zeros(M, dtype=np.float64)
        zdown = np.zeros(M, dtype=np.float64) 

        idx = np.arange(M)
        i = idx % N 

        mask_right = (i < N - 1)
        mask_left  = (i > 0)

        
        xright[mask_right] = rx 
        xleft[mask_left]   = rx

        j = (idx // N) % N 

        mask_up_y   = (j < N - 1)
        mask_down_y = (j > 0)


        yup[mask_up_y]     = ry 
        ydown[mask_down_y] = ry 

        k = idx // (N**2)

        mask_up_z   = (k < N - 1)
        mask_down_z = (k > 0)


        zup[mask_up_z]     = rz 
        zdown[mask_down_z] = rz


        return [main, xright, xleft, yup, ydown, zup, zdown]


    def crank_step_(self, u, B, solve_callback, bn=None): 
        #!! This is a sparse matrix multiplication, efficient
        b = B @ u 

        if bn is not None:
            b += bn 

        return solve_callback(b)
        

    def crank(self):
        N = self.interior_nodes
        M = N**3

        dz = dy = dx = np.pi / (N + 1)
        rx = 1.0 / (dx**2) 
        ry = 1.0 / (dy**2)
        rz = 1.0 / (dz**2)

        # Creating main system matrices as 7 band sparse matrix 
        main_diag = -2.0 * (rx + ry + rz) * np.ones(M, dtype=np.float64)
        diagonals = self.mask_diagonals_(N, M, (rx, ry, rz), main_diag) 
        offsets   = [0, 1, -1, N, -N, N**2, -N**2]

        L = scipy.sparse.diags(diagonals, offsets, shape=(M, M), format="csc")
        I = scipy.sparse.eye(M, format="csc")
        
        k = 0.5 * self.alpha * self.dt 
        A = I - k * L 
        B = I + k * L 

        solve_callback = scipy.sparse.linalg.factorized(A)
        
        # interior collocation points 
        xi = np.linspace(dx, np.pi - dx, N) 
        yi = np.linspace(dy, np.pi - dy, N)
        zi = np.linspace(dz, np.pi - dz, N)
        X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")

        ics   = self.analytic(0.0, X, Y, Z).reshape(M) 
        uprev = ics
        u     = self.crank_step_(uprev, B, solve_callback) 
        u_norm_prev = np.linalg.norm(uprev, ord=2)
        u_norm = np.linalg.norm(u, ord=2)

        u_hist = [ics, u]

        while np.abs(u_norm - u_norm_prev) > EPS: 
            u_norm_prev = u_norm 
            uprev = u 
            
            u = self.crank_step_(u, B, solve_callback)
            u_norm = np.linalg.norm(u, ord=2)
            u_hist.append(u)
        print(f"Computed {len(u_hist)} steps")

        self.numeric_solution = np.array(u_hist, dtype=np.float64) 

    def plot(self, skip=10):
        if self.numeric_solution is None:
            raise RuntimeError("must call Isotropic.crank() prior to plotting")

        N = self.interior_nodes
        z_index = N // 2

        # numeric_solution has shape (n_steps, N**3)
        n_steps = self.numeric_solution.shape[0]
        u_hist_3d = self.numeric_solution.reshape(n_steps, N, N, N)
        midplane_cn = u_hist_3d[:, :, :, z_index]  # (n_steps, N, N)

        # Fixed color scale for the entire animation so the decay is visible
        vmin, vmax = 0.0, 1.0

        # Grid for PINN evaluation
        xs = np.linspace(0.0, np.pi, N + 2)
        ys = np.linspace(0.0, np.pi, N + 2)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.full_like(X, np.pi / 2.0)

        frames = []

        for n in range(0, n_steps, skip):
            t = n * self.dt

            # CN slice
            u_cn = midplane_cn[n]  # shape (N, N)

            # PINN slice at the same time
            T = np.full_like(X, t)
            X_in = np.stack(
                [X.ravel(), Y.ravel(), Z.ravel(), T.ravel()],
                axis=1,
            )
            u_pinn = np.asarray(self.pinn.predict(X_in)[:, 0]).reshape(N + 2, N + 2)

            fig, (ax_cn, ax_pinn) = plt.subplots(
                1, 2, figsize=(8, 4), dpi=120, constrained_layout=True
            )

            im_cn = ax_cn.imshow(
                u_cn.T,
                origin="lower",
                extent=(0, np.pi, 0, np.pi),
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                aspect="equal",
            )
            ax_cn.set_title(f"CN  t = {t:.4f}")
            ax_cn.set_xlabel("x")
            ax_cn.set_ylabel("y")

            im_pinn = ax_pinn.imshow(
                u_pinn.T,
                origin="lower",
                extent=(0, np.pi, 0, np.pi),
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                aspect="equal",
            )
            ax_pinn.set_title(f"PINN t = {t:.4f}")
            ax_pinn.set_xlabel("x")
            ax_pinn.set_yticklabels([])

            # One shared colorbar for both plots
            cbar = fig.colorbar(im_pinn, ax=[ax_cn, ax_pinn], fraction=0.046, pad=0.04)
            cbar.set_label(r"$u(x, y, z=\pi/2, t)$")

            fig.canvas.draw()

            buf = np.asarray(fig.canvas.buffer_rgba())  # (h, w, 4)
            frame = buf[:, :, :3].copy()                # drop alpha
            frames.append(frame) 

            plt.close(fig)

        # Single GIF with CN vs PINN evolution on the midplane
        imageio.mimsave("cn_pinn_midplane.gif", frames, fps=10)

        
def main(): 
    alpha  = 1.0 
    dt = 1e-3 
    layers = [4] + [64] * 3 + [1]
    model  = Isotropic(
        layers=layers,
        interior_nodes=30,
        pinn_nodes=[16384, 1024, 1024],
        alpha=alpha,
        dt=dt,
        load="iso-2500.pt"
    )
    print("Starting PINN Model Training")
    model.train()
    print("Starting Crank-Nicolson Scheme")
    model.crank() 
    print("Starting Animated Plots")
    model.plot()


if __name__ == "__main__": 
    main() 

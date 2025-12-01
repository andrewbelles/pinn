#/usr/bin/env python 

import numpy as np
import numpy.typing as nptype 
import matplotlib.pyplot as plt 
from scipy.linalg import solve_banded 

import keras
import tensorflow as tf

# User-Implementation Imports 
from pinn import PINN, new_mlp 

class SteadyHeat: 

    def __init__(self, interior_nodes: int, input_dim: int, layers, path: str | None = None): 
        self.interior_nodes = interior_nodes
        self.finite_difference_solution: nptype.NDArray[np.float64] = np.zeros(
            (interior_nodes + 2),dtype=np.float64) 

        # Instantiate PINN 
        x, l, r = self.get_discrete_inputs()
        self.pinn = self.pinn_instantiation_(input_dim, layers, x, l, r, path)
        self.pinn_solution: nptype.NDArray[np.float64] = np.zeros(
            (interior_nodes + 2),dtype=np.float64) 


    def finite_difference(self): 
        N = self.interior_nodes 
        if N <= 0: 
            return 

        h = np.float64(1.0 / (N + 1.0))

        # Initialize Solution Vector and System's Matrices
        self.finite_difference_solution = np.zeros((N,), dtype=np.float64)
        boundary = np.zeros((N,), dtype=np.float64)  
        
        # Get boundary vector 
        xs = np.linspace(h, 1.0 - h, N, dtype=np.float64) 
        boundary = -(h**2) * np.float64(np.pi**2) * np.sin(np.pi * xs)

        main_diagonal = -2.0 * np.ones((N,), dtype=np.float64)
        off_diagonal  = 1.0 * np.ones((N-1), dtype=np.float64)

        system = np.zeros((3,N), dtype=np.float64)
        system[0, 1:] = system[2, :-1] = off_diagonal 
        system[1, :]  = main_diagonal
        
        interior_solutions = np.asarray(solve_banded((1,1), system, boundary), dtype=np.float64)
        self.finite_difference_solution = np.empty(N + 2, dtype=np.float64)
        self.finite_difference_solution[0] = self.finite_difference_solution[-1] = 0.0
        self.finite_difference_solution[1:-1] = interior_solutions


    def get_discrete_inputs(self):
        N = self.interior_nodes 
        h = 1.0 / (N + 1.0)
        xs = np.linspace(h, 1.0 - h, N, dtype=np.float64)[:, None]
        xtf = tf.convert_to_tensor(xs, dtype=tf.float32)
        ltf = tf.convert_to_tensor([0.0], dtype=tf.float32)
        rtf = tf.convert_to_tensor([1.0], dtype=tf.float32)
        return xtf, ltf, rtf


    def pinn_instantiation_(self, input_dim, layers, x, l, r, path: str | None = None):
        """
        Helper function to __init__ specific to this PDE which defines the 
        model architecture and residual terms that define the system 
        """
        def pde_term(model: keras.Model):
            with tf.GradientTape(persistent=True) as t2: 
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x) 
                    u = model(x)
                ux = t1.gradient(u, x)
            uxx = t2.gradient(ux, x)
            del t2 
            return uxx + (np.pi**2) * tf.sin(np.pi * x)

        def left_bc(model: keras.Model):
            return model(l)[:, 0]
        
        def right_bc(model: keras.Model):
            return model(r)[:, 0]

        residual_terms = [pde_term, left_bc, right_bc]
        if path:
            model = self.load(residual_terms, path)
        else: 
            model = new_mlp(input_dim, layers=layers)

        return PINN(model=model, residual_terms=residual_terms)
        

    def pinn_eval(self, lr, epochs):
        self.pinn.train_adam_(lr, epochs, epochs // 10)
        x = np.linspace(0.0, 1.0, self.interior_nodes + 2, dtype=np.float64)
        upred = self.pinn.predict(x).flatten()
        self.pinn_solution = upred.astype(np.float64)


    def save(self, path: str = "pinn.keras"):
        self.pinn.model.save(path)


    def load(self, residual_terms, path: str = "pinn.keras"): 
        model = keras.models.load_model(path)
        return PINN(model=model, residual_terms=residual_terms)


    def error_analysis_(self, arr: nptype.NDArray[np.float64]):
        # Need analytic solution array of same size of arr 
        N = len(arr) 
        xs = np.linspace(0.0, 1.0, N)
        analytic = np.sin(np.pi * xs)

        err = np.abs(analytic - arr)
        return err, analytic  


    def plot(self): 
        finite_difference_error, analytic = self.error_analysis_(self.finite_difference_solution)
        pinn_error, _ = self.error_analysis_(self.pinn_solution)
        
        x = np.linspace(0.0, 1.0, self.interior_nodes + 2)

        f, axes = plt.subplots(1, 2, figsize=(12,12))
        axes[0].plot(x, self.finite_difference_solution, label="Finite Difference Solution", lw=1.0, color="g")
        axes[0].plot(x, self.pinn_solution, label="PINN Solution", lw=1.0, color="b")
        axes[0].plot(x, analytic, label="Analytic Solution", lw = 0.8, color="k", alpha=0.65)
        axes[0].set_xlabel("x [u]")
        axes[0].set_ylabel("y [u]")
        axes[0].legend()
        
        axes[1].semilogy(x, finite_difference_error, label="Finite Difference Error", color="g", alpha=0.8)
        axes[1].semilogy(x, pinn_error, label="PINN Error", color="b", alpha=0.8)
        axes[1].set_xlabel("x [u]")
        axes[1].set_ylabel(R"$|\hat{u} - u^a| [log]$")
        axes[1].legend()
        f.suptitle("Comparison of Finite Difference and PINN model to Analytic Solution")
        f.savefig("comparison.png")
        plt.close() 


def main():
    model = SteadyHeat(320, 1, (32, 32))
    model.finite_difference() 
    model.pinn_eval(1e-3, 1000)
    model.plot()
    model.save("pinn.keras") 

if __name__ == "__main__":
    main() 

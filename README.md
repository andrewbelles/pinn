# PINN 

Projects involving Physics informed Neural Networks to gain better understanding of their use in numeric methods, specifically in the field of partial differential equations 

### Simple Steady State 

The following ordinary differential equation given by the boundary value problem: 

$u''(x)=-\pi^2\sin(\pi x), x\in(0,1)$, boundary conditions, $u(0) = 0$ and $u(1) = 0$

We can solve using a finite difference scheme with truncation error $\sim O(h^2)$ to solve the system, as well as by training a PINN using the ode. We can see a comparison of the two in the given plot below,  
![Finite Difference vs. PINN](./simple/comparison.png)

### Isotropic Heat Equation 3D 

$u_t = \alpha \nabla^2 u$, for $(x,y,z)\in(0,\pi)^3, t>0$ 

Given the Dirichlet boundary conditions 

$u(t,x,y,z) = 0$ for $(x,y,z) \in \delta(0,\pi)^3, t>0$

For the second order Crank-Nicolson scheme, we have a main diagonal with entries $1 + r_x + r_y + r_z$ where $r_x = \frac{\alpha\Delta t}{\Delta x^2}, r_y = \frac{\alpha\Delta t}{\Delta y^2}$, and $r_z = \frac{\alpha\Delta t}{\Delta z^2}. 

Off diagonals in the $x$ direction, $-r_x/2$, and likewise for the $y$ and $z$ directions. The RHS collapses to the following,

b^n_{i,j,k} = (1-r_x-r_y-r_z)u^n_{i,j,k} + \frac{r_x}2 (u^n_{i+1,j,k} + u^n_{i-1,j,k}) + \frac{r_y}2 (u^n_{i,j+1,k} + u^n_{i,j-1,k}) - \frac{r_z}2 (u^n_{i,j,k+1} + u^n_{i,j,k-1})


Which for boundaries terms, $u$ evaluates to zero given by the boundary conditions. 

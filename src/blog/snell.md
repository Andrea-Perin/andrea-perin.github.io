---
title: "Deriving Snell's law from differential geometry"
date: 2026-03-04
---

A colleague showed me an extract of a book where Snell's law was "derived" by
using differential geometry arguments.
Quoting Wikipedia's page for [Snell's law](https://en.wikipedia.org/wiki/Snell's_law),

> The law states that, for a given pair of media, the ratio of the sines of
> angle of incidence $\theta_1$ and angle of refraction $\theta_2$ is equal to
> the refractive index of the second medium with regard to the first $n_{2,1}$
> which is equal to the ratio of the refractive indices $\frac{n_2}{n_1}$ of
> the two media, or equivalently, to the ratio of the phase velocities
> $\frac{v_1}{v_2}$ in the two media:
> $$ \frac{\sin \theta_1}{\sin \theta_2} = \frac{n_2}{n_1} $$ 
> The law follows from Fermat's principle of least time, which in turn follows from the propagation of light as waves. 

Let's start with some nice interactive result first, and motivate it afterwards.
Note that this applet is powered only by differential geometry!
The implementation here is mostly due to Claude, since I know no Javascript.
Implementing an applet like this can be easily done in Python as well, however.

<iframe src="geodesic_applet.html" width="100%" height="520px"
        style="border:none; border-radius:4px;"></iframe>

### Setup

Consider the 2D plane, and imagine it is "split" into two halves about the $y$ axis.
These two halves constitute two different media in which light propagates.
To represent the different refractive indexes, we use a metric tensor that changes as we move along the $x$ coordinate.
In the following, we use the differential geometry convention of referring to coordinates as $x^\mu$, where $\mu$ is an index that ranges over the number of dimensions (in this case, $\mu = 1, 2$).
Basically, $x^1$ will be the usual $x$, and $x^2$ will be the usual $y$.
We define the metric tensor as
$$
g^{\mu\nu}(x^1, x^2) = (1 + \frac{\alpha}{2}(1 + \tanh(\beta x^1)))\delta^{\mu\nu}.
$$
For large $\beta$, this function becomes "almost the step function", representing the sharp interface between the two media.
We can't make this truly discontinuous, but it's good enough.
To the left, when $x^1 \ll 0$, the metric tensor is the identity.
To the right, when $x^1 \gg 0$, the metric tensor acquires a scaling factor of $1+\alpha$.
This is actually an example of a [conformally flat manifold](https://en.wikipedia.org/wiki/Conformally_flat_manifold).
The conformal factor is then
\begin{align}
\Omega(x^1, x^2) = \sqrt{1 + \frac{\alpha}{2}(1 + \tanh(\beta x^1))}.
\end{align}
Going from a medium to the other, this factor can be interpreted as a change in how "costly" it is to move through space.

### Derivation
Let us then try to derive Snell's law starting from this "purely geometric" difference.
The first thing we can do is to write down the geodesic equation.
In general, it reads
$$
\frac{d^2x^\lambda }{dt^2} + \Gamma^{\lambda}_{\mu \nu }\frac{dx^\mu }{dt}\frac{dx^\nu }{dt} = 0,
$$
where $\Gamma$ is the [Christoffel symbol](https://en.wikipedia.org/wiki/Christoffel_symbols) of the metric.
These are
\begin{align}
\Gamma^1 &= \left[\begin{matrix}\frac{\alpha \beta}{2 \left(\alpha \tanh{\left(\beta x^{1} \right)} + \alpha + 2\right) \cosh^{2}{\left(\beta x^{1} \right)}} & 0\\0 & - \frac{\alpha \beta}{2 \left(\alpha \tanh{\left(\beta x^{1} \right)} + \alpha + 2\right) \cosh^{2}{\left(\beta x^{1} \right)}}\end{matrix}\right], \\
\Gamma^2 &= \left[\begin{matrix}0 & \frac{\alpha \beta}{2 \left(\alpha \tanh{\left(\beta x^{1} \right)} + \alpha + 2\right) \cosh^{2}{\left(\beta x^{1} \right)}}\\\frac{\alpha \beta}{2 \left(\alpha \tanh{\left(\beta x^{1} \right)} + \alpha + 2\right) \cosh^{2}{\left(\beta x^{1} \right)}} & 0\end{matrix}\right].
\end{align}

Of course no one computes these by hand, and I am no exception.
In the collapsible section below you can find a `sympy` code snippet that did it for me (courtesy of Claude, because `sympy` docs don't really allow to learn from examples).

<details>

<summary>Code for computing Christoffel symbols</summary>

```python
import sympy as sp

# Coordinates and parameters
x1, x2 = sp.symbols('x^1 x^2')
alpha, beta = sp.symbols('alpha beta', real=True)

# Conformal factor
def Omega(x):
    return 1 + (alpha / 2) * (sp.tanh(beta * x) + 1)

# Metric tensor g_ij = Omega(x1) * delta_ij
def metric(i, j):
    return Omega(x1) * sp.KroneckerDelta(i, j)   


g = sp.Matrix([[metric(i, j) for j in range(2)] for i in range(2)])
g_inv = g.inv()

coords = [x1, x2]

# Christoffel symbols Gamma^k_ij (second kind)
# Gamma^k_ij = (1/2) * g^{kl} * (d_i g_lj + d_j g_li - d_l g_ij)
def christoffel(k, i, j):
    return sp.Rational(1, 2) * sum(
        g_inv[k, l] * (
            sp.diff(g[l, j], coords[i]) +
            sp.diff(g[l, i], coords[j]) -
            sp.diff(g[i, j], coords[l])
        )
        for l in range(2)
    )

# Compute and simplify all non-zero components
print("Non-zero Christoffel symbols Γ^k_ij:\n")
gammas = []
for k in range(2):
    gammas.append([])
    for i in range(2):
        gammas[k].append([])
        for j in range(2):
            new_gamma = sp.simplify(christoffel(k, i, j))
            gammas[k][i].append(new_gamma)
    # print the latex string
    print(f'\\Gamma^{k+1} &= {sp.latex(sp.Matrix(gammas[k]))} \\\\')
```
</details>


### Solving the geodesics

Now the geodesics can be written out a bit more explicitly:
\begin{align}
\frac{d^2x^1 }{dt^2} + \Gamma^{1}_{1 1}\left(\frac{dx^1}{dt}\right)^2 + \Gamma^{1}_{2 2}\left(\frac{dx^2}{dt}\right)^2= 0,\\
\frac{d^2x^2 }{dt^2} + \Gamma^{2}_{1 2}\frac{dx^1}{dt}\frac{dx^2}{dt} + \Gamma^{2}_{2 1}\frac{dx^1}{dt}\frac{dx^2}{dt} = 0.
\end{align}

If we call
\begin{align}
\Gamma(x^1) = \Gamma^1_{11} = \Gamma^2_{12} = \Gamma^2_{21} = -\Gamma^1_{22} =  \frac{\alpha \beta}{2 \left(\alpha \tanh{\left(\beta x^{1} \right)} + \alpha + 2\right) \cosh^{2}{\left(\beta x^{1} \right)}},
\end{align}

and sub those in the geodesic equations, we get
\begin{align}
&\frac{d^2x^1 }{dt^2} + \Gamma(x^1) \left(\frac{dx^1}{dt}\right)^2 - \Gamma(x^1) \left(\frac{dx^2}{dt}\right)^2= 0,\\
&\frac{d^2x^2 }{dt^2} + 2 \Gamma(x^1) \frac{dx^1}{dt}\frac{dx^2}{dt} = 0.
\end{align}

These ugly ODEs can only be solved numerically.
It is not too hard to do so using a package like `scipy`.
However, you can also just play around with the applet above.

### Interpreting the result

Here I am going out on a limb a little bit.
In Snell's law, the ratio between the refractive indices is taken to represent the ratio between the "ease" of traversing a medium.
In the same spirit, then, I took the ratio between the conformal factor $\Omega$ at the beginning and at the end of the trajectory.
It turns out, then, that
\begin{align}
\frac{\sin\theta_i}{\sin\theta_f} = \frac{\Omega_f}{\Omega_i},
\end{align}
where the conformal factor is taken to measure the "cost" of moving through a medium.
You can check the match between the two ratios in the applet above, at the bottom of the right hand panel.

What is really quite interesting is what happens when you set $\alpha < -0.5$ and the initial velocity to have both components positive and equal.
It almost looks like reflection!
Why is that?
For $\alpha = -1/2$, in those conditions, we have that to the right of the interface (in the approximation where $\beta\gg 1$) the conformal factor becomes $\Omega = \sqrt{1/2}$.
So we have $\sin\theta_i = \sqrt{1/2}$ and $\Omega_f/\Omega_i = \sqrt{1/2}$, and so it must be that $\sin\theta_f = 1$.
This means that the light just moves parallel to the interface.
*We have just rediscovered [total internal refraction](https://en.wikipedia.org/wiki/Total_internal_reflection)*, the mechanism behind, among other things, fiber optics.
In this case, in fact, the left hand side has higher refractive index than the right hand side, and it would be like the glass of fiber optics.

For $\alpha < -2$, the conformal factor breaks down completely, becoming imaginary.
Maybe this also has a physical interpretation after all, but it feels like it is something a bit more exotic.
A bit of creativity may allow an interesting characterization of this regime.
Hopefully an idea will pop up in the future.

Please contact me if you have a deeper understanding of this, beyond my handwavy reasoning here!

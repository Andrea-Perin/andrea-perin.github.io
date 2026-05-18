---
title: "Entropy production in an Ornstein-Uhlenbeck process"
date: 2026-05-11
---

**Question:** what is *entropy production*?
I honestly had no idea before doing this exercise, so I just compiled the work I did and added some visualization.

The widget below lets you explore the forward and reverse paths of an
[Ornstein-Uhlenbeck (OU)
process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), the
stationary covariance ellipse, and the probability current — all as a function
of the drift matrix $A$ and noise level $\sigma$.
Forward paths (blue) and time-reversed paths (red) are indistinguishable when
$e_p = 0$; the more they differ, the larger the entropy production.
Play around with the widget and explore the various presets; you can directly modify the drift matrix too[^hurwitz].
Can you guess what is the singular factor driving entropy production?

```{=html}
<div style="display:flex; justify-content:center; margin: 1.5em 0;">
  <iframe
    id="ou-widget"
    src="ou-widget.html"
    width="602"
    height="500"
    style="border:none; display:block; overflow:hidden;"
    scrolling="no"
    title="Ornstein-Uhlenbeck Explorer"
  ></iframe>
</div>
<script>
window.addEventListener('message', function(e) {
  if (e.data && e.data.ouWidgetHeight) {
    var f = document.getElementById('ou-widget');
    if (f) f.height = e.data.ouWidgetHeight + 2;
  }
});
</script>
```

## Derivation

Consider an Ornstein-Uhlenbeck process,

$$
dX_t = AX_t dt + \sigma dW_t,
$$

and the associated Fokker-Planck equation:

$$
\partial_t p = -\nabla \cdot (Ax p) + D_{ij} \partial_i\partial_j p.
$$

Crucially, we require $A$ to be [Hurwitz](https://en.wikipedia.org/wiki/Hurwitz-stable_matrix), i.e., to have all eigenvalues with negative real part.
Intuitively, this means that the deterministic portion of the SDE is "contractive" and there are no exploding trajectories.

## Step 1: find the stationary distribution
We want to find $p_\infty$, the stationary distribution.
This is computed by equating the right hand side of the FP equation to 0:

$$
0 = -\nabla \cdot (Ax p_\infty) + D_{ij} \partial_i\partial_j p_\infty.
$$

In practice, we do a bit of guesswork and go for a Gaussian *ansatz*,
$$
p_\infty = Z \exp\left(-\frac{1}{2}x^T\Sigma^{-1}x\right).
$$

Direct computations of the pieces:
$$
\begin{align}
\nabla \cdot (Axp_\infty) =& p_\infty \left(\mathrm{Tr}(A) - (Ax)\cdot(\Sigma^{-1}x)\right);\\
\partial_i\partial_j p_\infty =& p_\infty \left((\Sigma^{-1}x)_i(\Sigma^{-1}x)_j - \Sigma^{-1}_{ij}\right).
\end{align}
$$

Putting all together:
$$
0 = -p_\infty \left(\mathrm{Tr}(A) - (Ax)\cdot(\Sigma^{-1}x)\right) + p_\infty D_{ij} \left((\Sigma^{-1}x)_i(\Sigma^{-1}x)_j - \Sigma^{-1}_{ij}\right).
$$

Removing the common $p_\infty$ factor, we can notice that there are only terms independent of $x$, and terms that are squares of $x$.
Thus we obtain two equations:
$$
\begin{align}
0 =& \mathrm{Tr}(A)  + D_{ij} \Sigma^{-1}_{ij};\\
0 =& (Ax)\cdot(\Sigma^{-1}x) + D_{ij}(\Sigma^{-1}x)_i(\Sigma^{-1}x)_j.
\end{align}
$$

We can rearrange the second term in order to express it as a quadratic form in $x$.
We get that:
$$
\begin{align}
(Ax)\cdot(\Sigma^{-1}x) &= x^T A^T \Sigma^{-1}x,\\
D_{ij}(\Sigma^{-1}x)_i(\Sigma^{-1}x)_j &= (\Sigma^{-1}x)^T D(\Sigma^{-1}x) = x^T(\Sigma^{-T}D \Sigma^{-1})x.
\end{align}
$$

Then, for $C = (A^T \Sigma^{-1} + \Sigma^{-1}D \Sigma^{-1})$,
$$
x^T C x = 0.
$$
This must hold regardless of $x$, and so we get that *the symmetric part* of $C$ must be 0[^antisymm]:
$$
A^T \Sigma^{-1} + \Sigma^{-1}D \Sigma^{-1} + (A^T \Sigma^{-1} + \Sigma^{-1}D \Sigma^{-1})^T = 0.
$$
Collecting terms, and multiplying by $\Sigma$ both on the left and on the right, we get the **Lyapunov equation**:
$$
\Sigma A^T + A \Sigma + 2 D=0.
$$
This last condition gives us a constraint on $\Sigma$.
Since $A=S+B$ with $S=S^T$ and $B=-B^T$, we have
$$
\begin{align}
\Sigma S + S \Sigma + 2 D=0,\\
[B, \Sigma]=0.
\end{align}
$$
The second condition means that $B$ and $\Sigma$ are simultaneously diagonalizable.
In other words, the choice of $B$ constrains the shape of the equilibrium distribution $\Sigma$.
So now we know something more about the equilibrium distribution $p_\infty$.

## Step 2: probability current
We now write the probability current $j$, defined as
$$
j(x)=f(x)p_\infty(x)−\nabla \cdot(D p_\infty(x)).
$$
In the Ornstein-Uhlenbeck case, $f(x) = Ax$, and so
$$
j(x)=Axp_\infty(x)−\nabla \cdot(D p_\infty(x)).
$$
Since $D$ is independent of $x$, we get
$$
j(x) = (A+D\Sigma^{-1})xp_\infty.
$$
We can rewrite the term in the parenthesis:
$$
\begin{align}
(A + D\Sigma^{-1}) &= (A \Sigma + D)\Sigma^{-1}\\
&= (S\Sigma + B\Sigma - \frac{1}{2}(\Sigma S + S\Sigma))\Sigma^{-1}\\
&= B + \frac{1}{2}(S - \Sigma S \Sigma^{-1})\\
&= B + \frac{1}{2}[S, \Sigma]\Sigma^{-1},
\end{align}
$$
where we used the symmetric part of the Lyapunov equation to write
$$
D = -\frac{1}{2} (\Sigma S + S \Sigma).
$$
Notice that the term $(1/2)[S,\Sigma]\Sigma^{-1}$ measures the "relative misalignment" between $S$ and $\Sigma$ (and by extension $B$, which shares the eigenbasis with $\Sigma$).
I will call this term $S_\mathrm{off}$ in the following.
So
$$
j(x) = (B + S_\mathrm{off})xp_\infty.
$$
Importantly, $S_\mathrm{off}$ is a function of only $S, B, D$.

## Step 3: time-reversed drift
We now use the formula for the time-reversed drift, which for a general $f$ reads
$$
f_\mathrm{rev}(x) = -f(x) +2D(x) \nabla \log p_\infty(x) + \nabla \cdot (2D(x)).
$$
In our Ornstein-Uhlenbeck case, $f(x)=Ax$ and $D$ is fixed, so
$$
f_\mathrm{rev}(x) = -Ax +2D \nabla \log p_\infty(x).
$$
Direct computation:
$$
\log p_\infty = \log (Z\exp(-\frac{1}{2} x^T\Sigma^{-1}x)) = \log Z - \frac{x^T\Sigma^{-1}x}{2}.
$$
Thus
$$
f_\mathrm{rev}(x) = -Ax - 2D \nabla \frac{x^T\Sigma^{-1}x}{2} = -(A+2D\Sigma^{-1})x.
$$
We can see that
$$
f - f_\mathrm{rev} = A + (A + 2D\Sigma^{-1}) = \frac{2j}{p_\infty}.
$$

## Step 4: entropy production formula
The actual formula (courtesy of Girsanov) for the entropy production $e_p$ is
$$
e_p = \frac{1}{2}\int (f - f_\mathrm{rev})^T D^{-1} (f - f_\mathrm{rev}) p_\infty dx;
$$
subbing in the last equality above,
$$
\begin{align}
e_p &= \frac{1}{2}\int \frac{2j^T}{p_\infty} D^{-1} \frac{2j}{p_\infty} p_\infty dx\\
&= \frac{1}{2}\int 2x^T(B+S_\mathrm{off})^T D^{-1} 2(B+S_\mathrm{off})x  p_\infty dx\\
&= 2\int x^TMx p_\infty dx \\
&= 2\mathbb{E}[x^TMx]\\
&= 2\mathbb{E}[x_iM_{ij}x_j]\\
&= 2M_{ij}\mathbb{E}[x_ix_j]\\
&= 2M_{ij}\Sigma_{ij}\\
&= 2\mathrm{Tr}(M\Sigma),
\end{align}
$$
where $M=(B+S_\mathrm{off})^T D^{-1} (B+S_\mathrm{off})$.

### Some special cases

#### $S$ and $B$ commute
An especially nice case is that of $[S, B]=0$.
First of all, $S_\mathrm{off}=0$ here.
We also get an almost clean expression for $\Sigma$.
We can study the symmetric portion of the Lyapunov equation in the eigenbasis of $S$ (and $B$).
Here, the equation simplifies:
$$
S\Sigma + \Sigma S = -2D \rightarrow s_i\Sigma_{ij} + \Sigma_{ij}s_j = -2D_{ij},
$$
so we get the elementwise expression for $\Sigma$:
$$
\Sigma_{ij} = -\frac{2D_{ij}}{s_i+s_j},
$$
with $s_i, s_j$ being the eigenvalues of $S$.
Recall that these eigenvalues are smaller than 0, due to the Hurwitz property of $A$.
There is not much more that can be said of this case without assuming further structure.

#### $S$ and $B$ commute, and $D\propto I$
Here is the fun stuff.
Remember that we want to find an expression for
$$
e_p = 2\mathrm{Tr}(B^T D^{-1} B \Sigma).
$$
Let's work in the eigenbasis of $B$ and $S$.
Let us also assume that $D_{ij} = 2\sigma^2 \delta_{ij}$.
This is the condition of *isotropic noise*, which is very natural.

In its eigenbasis, $B=\bigoplus_{\ell=1}^{\lfloor N/2 \rfloor} B_\ell$, with blocks $B_\ell$ that look like
$$
B_\ell = \begin{pmatrix}
0 & \lambda_\ell \\
-\lambda_\ell & 0 
\end{pmatrix}.
$$
In the same eigenbasis, $D$ is diagonal and so is $\Sigma$.
Thus, the expression $B^T D^{-1} B \Sigma$ is composed of 2x2 blocks, too.
This is due to the fact that a block-wise matrix, when multiplied with a diagonal one, maintains its block-wise structure.
The same holds when two matrices sharing their block-wise structure are multiplied.

Thus, the trace becomes a sum of the traces of the $\lfloor N/2 \rfloor$ blocks.
As a consequence, **entropy production "factorizes" into separate, independent blocks**!
The $\ell$-th block has the following entropy production:
$$
\begin{align}
e_p(\ell) &=\mathrm{Tr}\left[B_\ell^T D_\ell^{-1} B_\ell \Sigma_\ell\right] \\
&= \mathrm{Tr}\left[\begin{pmatrix}
0 & \lambda_\ell \\
-\lambda_\ell & 0 
\end{pmatrix}^T
\frac{1}{2\sigma^2}
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
\begin{pmatrix}
0 & \lambda_\ell \\
-\lambda_\ell & 0 
\end{pmatrix}
2\sigma^2
\begin{pmatrix}
\frac{1}{s_{2\ell}} & 0 \\
0 & \frac{1}{s_{2\ell+1}} 
\end{pmatrix}\right] \\
&= \mathrm{Tr}\left[\begin{pmatrix}
0 & -\lambda_\ell \\
\lambda_\ell & 0 
\end{pmatrix}
\begin{pmatrix}
0 & \lambda_\ell \\
-\lambda_\ell & 0 
\end{pmatrix}
\begin{pmatrix}
\frac{1}{s_{2\ell}} & 0 \\
0 & \frac{1}{s_{2\ell+1}} 
\end{pmatrix}\right] \\
&= \mathrm{Tr}\left[\begin{pmatrix}
-\lambda_\ell^2 & 0 \\
0 & -\lambda_\ell^2
\end{pmatrix}
\begin{pmatrix}
\frac{1}{s_{2\ell}} & 0 \\
0 & \frac{1}{s_{2\ell+1}} 
\end{pmatrix}\right]\\
&= -\lambda_\ell^2\left(\frac{1}{s_{2\ell}} + \frac{1}{s_{2\ell+1}}\right) \\
&= \lambda_\ell^2\left(\frac{1}{|s_{2\ell}|} + \frac{1}{|s_{2\ell+1}|}\right).
\end{align}
$$
The last step, with the appearance of the absolute value, is due to the fact that the eigenvalues $s_i$ are negative (again, the Hurwitz condition).

The final result for the entropy production is thus
$$
e_p = \sum_{\ell=1}^{\lfloor N/2 \rfloor}\lambda_\ell^2\left(\frac{1}{|s_{2\ell}|} + \frac{1}{|s_{2\ell+1}|}\right).
$$
This is very nice: the dynamics decomposes into independent subspaces in which:

* the faster the rotation, the larger the entropy production;
* the larger the eigenvalues, the smaller the entropy production.
The last point can be understood in terms of the "steepness" of the potential in those dimensions.
The steeper the potential, the harder the trajectory is pulled towards the center.
This in turn means that the noise can not "inject" too much energy in the system, and thus there is less that the rotation component can do.

## Bonus paragraph: equivalent marginals
In the above, we have looked at how the choice of the system parameters (the drift $A$ and the noise covariance $D$) determine the equilibrium distribution $\Sigma$.
This is the **Lyapunov equation** we mentioned above.
Now, an interesting question is:

> How many different ways can we set $A$ and $D$, and find *the same* equilibrium distribution, i.e., $\Sigma$?

The task is: fix $\Sigma$ and $D$, and determine all $A$ such that
$$
    A\Sigma + \Sigma A^T + 2D =0.
$$
If we left- and right-multiply by $\Sigma^{-1}$, we find
$$
    \Sigma^{-1}A + A^T\Sigma^{-1} + 2\Sigma^{-1}D\Sigma^{-1} =0,
$$
and splitting the term with $D$ equally, we can rewrite this as
$$
    \Sigma^{-1}(A + D\Sigma^{-1}) + (A^T+ \Sigma^{-1}D)\Sigma^{-1} =0.
$$
Let's define
$$
    Q = A + D\Sigma^{-1} \quad \mathrm{implying}\quad A=-D\Sigma^{-1}+Q,
$$
and the equation reads
$$
    \Sigma^{-1}Q + Q^T\Sigma^{-1}=0;
$$
another matmul sandwich, this time with $\Sigma$, and we get
$$
    Q\Sigma + \Sigma Q^T =0.
$$
This means that $\Sigma$ is determined by $A$, *modulo the choice of a matrix $Q$ such that $Q\Sigma$ is antisymmetric*.
Define then the antisymmetric matrix $\Omega$ as
$$
    \Omega=Q\Sigma \quad \mathrm{implying}\quad Q=\Omega\Sigma^{-1},
$$
and plug this expression for $Q$ inside the definition of $A$ above:
$$
    A = -D\Sigma^{-1} + \Omega\Sigma^{-1} = (-D+ \Omega)\Sigma^{-1}.
$$
In other words: *adding a rotation component to the field will not change the equilibrium distribution*.
But remember that it is precisely that rotation component to induce entropy production in the first place.
And so we get that *we can't determine whether entropy is being produced by just looking at the stationary distribution*.
In other words, you can not get the nonequilibrium structure just by looking at the time-averaged paths.
I guess the sharper readers will be entirely unimpressed by this statement, but I like to find obvious results as consequences of maths.


### Acks
A big thanks to [Victor Yeom Song](https://vyeoms.github.io/) who proofread and found a few mistakes here and there!
A healthy mix of the big three AI models provided the JS visualization, in which I coded a portion of the actual maths for the simulations.


[^hurwitz]: If you select "bad" values, your matrix may not be Hurwitz (as defined in the derivation). If this happens, the simulation will simply stop.
[^antisymm]: Indeed, for an antisymmetric matrix $B$, it holds true that $x^TBx=0$ for all $x$.

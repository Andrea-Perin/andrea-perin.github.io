---
title: "A derivation of Tweedie's formulas"
date: 2026-05-21
---

These notes are just a personal reminder of how one could derive the Tweedie's formulas for mean and covariance that so frequently appear in diffusion-related literature.

## Setting

Consider a vector valued observation $y$ which is assumed to be a noise-corrupted version of a true signal $x$.
The noise is supposed to be gaussian with mean 0 and covariance $\Sigma$.
In formulas,
$$
y= x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Sigma).
$$
Tweedie's formulas give an expression for the expectation and the covariance of $x$, *conditional on the value of* $y$:
$$
\begin{align}
\mathbb{E}[x|y] &= y + \Sigma \nabla_y \log p(y),\\
\mathrm{Cov}(x|y) &= \Sigma\nabla_y\nabla_y^T \log p(y) \Sigma + \Sigma.
\end{align}
$$
The big idea behind these expressions is:

> We only have access to the observation $y$ and $x$ is unknown; thus, we should expect these formulas to only involve what we know:
>
> * the noise distribution (i.e., $\Sigma$)
> * the value of the observation, $y$.

## Tweedie's formula --- mean
The formula pops up on its own when computing a seemingly unrelated quantity[^robbins].
I will work in coordinate notation because I don't know vector calculus; $\partial_i$ is the derivative wrt the $i$-th entry of vector $y$.
Under the setup described above, consider
$$
\partial_i \log p(y) = \frac{\partial_i p(y)}{p(y)}.
$$
Since $p(y) = \int p(y|x)p(x)dx$, it follows that
$$
\frac{\partial_i p(y)}{p(y)} = \frac{\partial_i\int p(y|x)p(x)dx }{p(y)} = \frac{\int \left[\partial_i p(y|x)\right]p(x)dx }{p(y)}.
$$
Now, since $y=x+\varepsilon$, we have that $p(y|x)$ is the noise distribution, $\mathcal{N}(0, \Sigma)$.
Then,
$$
\begin{align}
\partial_i p(y|x) &= 
\partial_i \left[(2\pi)^{-n/2}|\Sigma|^{-1/2} \exp\left(-\frac{1}{2}(y-x)^T\Sigma^{-1}(y-x)\right) \right] \\
&=(2\pi)^{-n/2}|\Sigma|^{-1/2}\exp\left(-\frac{1}{2}(y-x)^T\Sigma^{-1}(y-x)\right) \partial_i \left(-\frac{1}{2}(y-x)^T\Sigma^{-1}(y-x)\right)  \\
&=p(y|x) (-\Sigma^{-1}(y-x))_i.
\end{align}
$$
We plug this inside the numerator in the initial equation:
$$
\begin{align}
\frac{\int (-\Sigma^{-1}(y-x))_i p(y|x)p(x)dx}{p(y)} 
&= \frac{\int (\Sigma^{-1}x)_i p(y|x)p(x)dx}{p(y)} -\frac{\int (\Sigma^{-1}y)_i p(y|x)p(x)dx}{p(y)}\\
&= \frac{\int (\Sigma^{-1}x)_i p(x|y)p(y)dx}{p(y)} -\frac{(\Sigma^{-1}y)_i \int p(y|x)p(x)dx}{p(y)}\\
&= \int (\Sigma^{-1}x)_i p(x|y)dx -\frac{(\Sigma^{-1}y)_i p(y)}{p(y)}\\
&= \mathbb{E}[(\Sigma^{-1}x)_i|y] - (\Sigma^{-1}y)_i\\
&= \mathbb{E}[\sum_j \Sigma^{-1}_{ij}x_j|y] - (\Sigma^{-1}y)_i\\
&= \sum_j \Sigma^{-1}_{ij}\mathbb{E}[x_j|y] - \sum_j \Sigma^{-1}_{ij}y_j\\
&= \sum_j \Sigma^{-1}_{ij}(\mathbb{E}[x_j|y] - y_j).
\end{align}
$$
Then, the formula reads
$$
\partial_i \log p(y) = \sum_j \Sigma^{-1}_{ij}(\mathbb{E}[x_j|y] - y_j).
$$
This is basically a linear system (if you squint, you get something of the form $b_i = A_{ij}x_j$), so the solution will be obtaind by inverting the matrix, leading to
$$
\sum_i \Sigma_{ji}\partial_i \log p(y) = (\mathbb{E}[x_j|y] - y_j),
$$
and moving things around, we get
$$
\mathbb{E}[x_j|y] = y_j + \sum_i \Sigma_{ji}\partial_i \log p(y).
$$
Note that the component-wise formula is for $j$ now instead of $i$, but we can just swap the names and get
$$
\mathbb{E}[x_i|y] = y_i + \sum_j \Sigma_{ij}\partial_j \log p(y).
$$
In vector form,
$$
\mathbb{E}[x|y] = y + \Sigma \nabla_y \log p(y).
$$


## Tweedie's formula --- covariance
For covariance, we follow a similar path.
Let us compute the second derivative
$$
\begin{align}
\partial_j \partial_i \log p(y) &= \partial_j \left(\sum_k \Sigma^{-1}_{ik}(\mathbb{E}[x_k|y] - y_k) \right)\\
&= \sum_k \Sigma^{-1}_{ik}\partial_j (\mathbb{E}[x_k|y] - y_k)\\
&= \sum_k \Sigma^{-1}_{ik}\partial_j (\mathbb{E}[x_k|y] - \delta_{jk})\\
&= \sum_k \Sigma^{-1}_{ik}\partial_j \mathbb{E}[x_k|y] - \Sigma^{-1}_{ij}.
\end{align}
$$
We now need to compute $\partial_j \mathbb{E}[x_k|y]$.
We express this as an integral, and the use Bayes' rule:
$$
\mathbb{E}[x_k|y] = \int x_k p(x|y)dx = \int x_k \frac{p(y|x)p(x)}{p(y)}dx.
$$
The derivative enters the integral, and operates on the ratio $p(y|x)/p(y)$:
$$
\begin{align}
\partial_j \frac{p(y|x)}{p(y)} 
&= \frac{ \partial_j p(y|x)p(y) - p(y|x)\partial_j p(y)}{p(y)^2}\\
&= \frac{ \partial_j p(y|x) }{p(y)} - \frac{p(y|x)}{p(y)}\frac{\partial_j p(y)}{p(y)}\\
&= \frac{ \left(\Sigma^{-1}(x-y)\right)_j p(y|x) }{p(y)} - \frac{p(y|x)}{p(y)}\partial_j \log p(y)\\
&= \frac{p(y|x)}{p(y)} \left( \sum_k \Sigma^{-1}_{jk}(x-y)_k - \partial_j \log p(y)\right)\\
&= \frac{p(y|x)}{p(y)} \left( \sum_k \Sigma^{-1}_{jk}(x-y)_k - \sum_k \Sigma^{-1}_{jk}(\mathbb{E}[x_k|y] - y_k)\right)\\
&= \frac{p(y|x)}{p(y)} \left( \sum_k \Sigma^{-1}_{jk}(x-y-\mathbb{E}[x|y]+y)_k\right)\\
&= \frac{p(y|x)}{p(y)} \left( \sum_k \Sigma^{-1}_{jk}(x-\mathbb{E}[x|y])_k\right).
\end{align}
$$
Plugging back in the integral,
$$
\partial_j \mathbb{E}[x_k|y] = \int \frac{p(y|x)p(x)}{p(y)} x_k \left( \sum_\ell \Sigma^{-1}_{j\ell}(x-\mathbb{E}[x|y])_\ell\right) dx.
$$
Bayes again on the horrible ratio of probabilities, plus taking out of the integral all that can be taken out, and interpreting integrals as expectations,
$$
\begin{align}
\partial_j \mathbb{E}[x_k|y] 
&= \sum_\ell \Sigma^{-1}_{j\ell} \int p(x|y) x_k \left( x_\ell-\mathbb{E}[x_\ell|y] \right) dx\\
&= \sum_\ell \Sigma^{-1}_{j\ell} \left(\mathbb{E}[x_k x_\ell | y] -\mathbb{E}[x_k \mathbb{E}[x_\ell|y] | y] \right)\\
&= \sum_\ell \Sigma^{-1}_{j\ell} \left(\mathbb{E}[x_k x_\ell | y] -\mathbb{E}[x_k | y] \mathbb{E}[x_\ell|y] \right)\\
&= \sum_\ell \Sigma^{-1}_{j\ell} \mathrm{Cov}(x|y)_{\ell k}.
\end{align}
$$
In the last step, I extract the expectation of $x_\ell$ because it is a constant.
And just like that, we have the conditional covariance inside the parentheses.
Plugging this back in the initial formula, and moving the indices around a little bit, plus using the symmetry in both $\Sigma$ and $\mathrm{Cov}(x|y)$,
$$
\begin{align}
\partial_j \partial_i \log p(y)
&= \sum_k \Sigma^{-1}_{ik} \left( \sum_\ell \Sigma^{-1}_{j\ell} \mathrm{Cov}(x|y)_{\ell k} \right) - \Sigma^{-1}_{ij}\\
&= \sum_{k, \ell} \Sigma^{-1}_{ik} \sum_\ell \Sigma^{-1}_{j\ell} \mathrm{Cov}(x|y)_{\ell k}  - \Sigma^{-1}_{ij}\\
&= \left[\Sigma^{-1}\mathrm{Cov}(x|y)\Sigma^{-1}\right]_{ij} - \Sigma^{-1}_{ij}.
\end{align}
$$
Now, rearranging, we can express the conditional covariance directly:
$$
\mathrm{Cov}(x|y)_{ij} = \Sigma_{ik}\partial_k\partial_\ell \log p(y) \Sigma_{\ell j} + \Sigma_{ij}.
$$
In matrix notation,
$$
\mathrm{Cov}(x|y) = \Sigma\nabla_y\nabla_y^T \log p(y) \Sigma + \Sigma.
$$


[^robbins]: Claude mentions that [Herbert Robbins](https://en.wikipedia.org/wiki/Herbert_Robbins) first noticed that the expectation $\mathbb{E}[x|y]$ emerged out of the computation of $\nabla_y \log p(y)$; so the interesting quantity just popped up "by chance".

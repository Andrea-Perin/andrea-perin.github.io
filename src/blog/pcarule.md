---
title: "PCA update rule"
date: 2025-09-02
---

Is it possible to find [Oja's rule](https://en.wikipedia.org/wiki/Oja%27s_rule) from a "linear autoencoder" type loss?
The (single sample) loss has the following functional form:

$$
\mathcal{L}^{(AE)} = \lVert x^{(n)} - W^T W x^{(n)} \rVert^2,
$$

where $x^{(n)} \in \mathbb{R}^d$, $n$ denotes the $n$-th sample, and $W \in \mathbb{R}^{k\times d}$.
The optimal $W$ is composed of columns which span the *principal subspace*, and constitute a basis.
This is effectively a *family* of solutions: among them, we may find the usual PCA or its close relative, the ZCA.
Needless to say, finding a specific solution is exceedingly rare, as they are all perfectly equivalent under the loss we are using.

If we are *extremely* lucky, we may just so initialize $W$ so that, under gradient descent for the linear autoencoder loss, we end up in the PCA solution.
By extremely, I think there is some kind of exponentially small probability of this happening.
Basically, try to optimize for this objective and you will simply find some solution that has little to do with PCA.
And by "little", I mean that it is a *rotated* and *scaled* version of PCA.
Indeed, any solution for the linear autoencoder loss will span[^1] the same subspace, but PCA has the additional requirement of orthonormality for the columns of $W$.
As such, a "linear autoencoder" flavored objective for PCA could look like this:

$$
\mathcal{L}^{(PCA)} = (1-\alpha)\lVert x^{(n)} - W^T W x^{(n)} \rVert^2 + \alpha \lVert \mathbb{I}_k - WW^T\rVert^2,
$$

where the additional term (opportunely weighted by $\alpha$) promotes orthogonality of the columns of $W$.
For the time being, I will drop superscripts and simply use $\mathcal{L}^{(AE)}$ as the objective to minimize.

Now, I want to compute the gradient that comes out of this loss, and do so with some index notation[^2].
The updates for $W$ are a function of the gradient $\frac{\partial \mathcal{L}}{\partial W}$.
I will use the shorthand $\partial_{ab} = \partial/\partial W_{ab}$, and avoid the superscript $(n)$, as the derivation simply distributes over the sum.

$$
\begin{align}
    \partial_{ab} \mathcal{L} &= \partial_{ab} \lVert x - W^T W x\rVert^2 \\
    &= \partial_{ab} \sum_k (x_k - (W^T W x)_k)^2 \\
    &= \partial_{ab} \sum_k (x_k - \sum_{ij} W^T_{ki} W_{ij} x_j)(x_k - \sum_{lm} W^T_{kl} W_{lm} x_m) \\
    &= \partial_{ab} \sum_k (x_k^2 - 2\sum_{ij} W^T_{ki} W_{ij} x_j x_k - \sum_{ijlm} W^T_{ki} W_{ij} W^T_{kl} W_{lm} x_j x_m) \\
    &= \partial_{ab} \sum_k (x_k^2 - 2\sum_{ij} W_{ik} W_{ij} x_j x_k - \sum_{ijlm} W_{ik} W_{ij} W_{lk} W_{lm} x_j x_m).
\end{align}
$$

Now the usual trick with derivatives in index notation.
Since a tensor entry is just "a number with a label", derivatives are only nonzero if the indices match.
In other words, consider a rank-2 tensor $T$.
Its $(i,j)$-th entry is a "labeled number", and it has "nothing to do" with its $(l, m)$-th entry, in the sense that their only thing in common is that they belong to $T$.
The only thing, that is, *unless* the indices are *all* the same!

Thus,
$$
\frac{\partial T_{ij}}{\partial T_{lm}} = \begin{cases}
1 \quad \mathrm{if} \quad i=l \quad \mathrm{and}\quad j=m,\\
0 \quad \mathrm{otherwise.}
\end{cases}
$$
The same information can be conveyed in a much more compact form using the [Kronecker Deltas](https://en.wikipedia.org/wiki/Kronecker_delta):
$$
\frac{\partial T_{ij}}{\partial T_{lm}} = \delta_{il}\delta_{jm}.
$$

With this classic trick, and remembering the differentiation rules for the product, we get the following expression:
$$
\begin{align}
\partial_{ab} \mathcal{L} = [& -2 \sum_{ijk} x_j x_k \delta_{ai}\delta_{bk} W_{ij} \\
& -2 \sum_{ijk}x_j x_k W_{ik} \delta_{ai}\delta_{bj}\\
& + \sum_{ijklm} \delta_{ai}\delta_{bk}W_{ij}W_{lk}W_{lm}x_j x_m \\
& + \sum_{ijklm} W_{ik}\delta_{ai}\delta_{bj}W_{lk}W_{lm}x_j x_m \\
& + \sum_{ijklm} W_{ij}W_{ij}\delta_{al}\delta_{bk}W_{lm}x_j x_m \\
& + \sum_{ijklm} W_{ik}W_{ij}W_{lk}\delta_{al}\delta_{bm}x_j x_m]
\end{align}
$$

Now, consider the following "renaming scheme" of some of the terms that appear in the expressions:
$$
y_i = \sum_j W_{ij}x_j, \quad x^{rec}_i = \sum_j W_{ji}y_i, \quad y^{rec}_i = \sum_j W_{ij} x^{rec}_i.
$$
In this notation, $y$ denotes the "encoded" version of $x$, with the "rec" superscript indicating the "reconstructed" versions of the variables.
Using these names, and simplifying the Kronecker Deltas, we get
$$
\partial_{ab} \mathcal{L} = -4 y_a x_b + 2 y_a x^{rec}_b + 2 y^{rec}_a x_b,
$$
which contains the Hebbian rule in the first summand, and some additional terms afterwards.
We can reframe the rigth hand side in a more symmetric, and insightful, form:
$$
\begin{align}
\partial_{ab} \mathcal{L} &= 2 \left(y_a (x^{rec}_b-x_b) + (y^{rec}_a-y_a) x_b\right)\\
&= 2 (y_a \Delta x^{rec}_b + \Delta y^{rec}_a x_b).
\end{align}
$$
So the weights are updated by "correcting" the reconstruction errors $\Delta x^{rec}$ and $\Delta y^{rec}$ **in both directions**!
I don't think I had this insight before doing this (honestly pretty basic) exercise.

Now, what about comparing this to Oja's rule?

### Oja's rule

Looking around the web, Oja's rule is formulated in a slightly different way, with the most common weight update formula looking like this:
$$
\Delta w_i = \eta (x_i y - y^2 w_i),
$$
where $i=1,\dots, N$ denotes the sample, and $W$ has now a single column.
This is consistent with the fact that Oja's rule recovers *the first principal component*.

It is easy to see that this rule is quite different from the one we found via gradient descent on the linear autoencoder loss.
Maybe if we look at the "augmented" loss, which includes an orthogonality term, things will get closer to our case.

### Gradient updates for the PCA loss

The computation is pretty much the same as in the previous section.
We simply need to add the gradients of the orthogonality portion of the loss.
Run the computations and you get that the additional orthogonality term (which I will now call $\mathcal{L}^{(O)}$), gives the following gradients:
$$
\partial_{ab} \mathcal{L}^{(O)} = -4 \left( W_{ab} - \sum_{ij} W_{ai}W_{ji}W_{jb} \right).
$$

I am inclined to call the second term $W^{rec}_{ab}$, since it kinda uses a weighted sum of the rows of $W$ to construct a $W$-compatible object.

It gets a bit more understandable if we look at a row-wise update (which means treating the rows of $W$ as if they were vectors[^3]).

TODO: do a drawing to get more intuition, add idea about repulsive forces of charges on a sphere (for the case in which they are orthonormal)

[^1]: The language here is a bit sloppy, but when I say that "the solution spans" I really mean *its columns*.
[^2]: I have never really learned how to do matrix derivatives; instead, I resort to using indices.
[^3]: Which they are, but I guess in the dual space. Still, let's think of them as arrows in some space.

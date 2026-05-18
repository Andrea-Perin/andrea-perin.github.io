---
title: "Derivatives with Penrose notation"
date: 2025-09-22
---

Recently, I have faced a relatively harmless task involving differentiating a loss function that includes a few tensor contractions.
My approach was the usual: treat tensors as numbers with labels, use some Kronecker delta, and that's it.
It is what I did [in this post](/blog/pcarule.html).

But what if there was a better way, using the [Penrose graphical notation](https://en.wikipedia.org/wiki/Penrose_graphical_notation)?

## The better way, using the Penrose notation

Let us consider the loss function I was dealing with.
It is the usual *principal subspace* loss, also known as the "linear autoencoder" loss:
$$
\mathcal{L} = \lVert x - W^T W x \rVert^2,
$$
for a single sample $x\in\mathbb{R}^d$, and a learnable weight matrix $W\in\in\mathbb{R}^{k\times d}$.
We want to compute the derivative of this scalar number with respect to the matrix $W$.
Let's try to frame this problem in the Penrose way.

### Step 1: the building blocks

First of all, let's write down what $x$ and $W$ are in the notation:
```{.ascii-art}
 ┼       │ 
┌┴┐     ┌┴┐
│W│     │x│
└┬┘     └─┘
 │         
```
Notice how the lower leg of $W$ and the (only) leg of $x$ look the same, while the upper leg of $W$ has a sideways segment.
This is to graphically denote that the *indices* have a specific *range*: we can only contract indices "of the same length".
In this sense, the notation gives us no chance of messing up the contractions, which I find pretty neat: no need to remember transpositions and shapes: just set the drawings up the right way, and the notation does the heavy lifting for you.

Indeed, let us also write what the encoding of $x$ with $W$ (which we will call $y$ in the following) looks like:
```{.ascii-art}
 ┼ 
┌┴┐
│W│
└┬┘
┌┴┐
│x│
└─┘
```
That is, $y=Wx$.

### Step 2: the L2 norm
     
The loss is the L2 norm of a vector (namely, the difference between $x$ and its reconstruction $W^TWx$).
For a vector $v$, it can be written as the following sum:
$$
\lVert v \rVert^2 = \sum_k v_k^2,
$$
which is basically the contraction of $v$ with itself.
Graphically,
```{.ascii-art}
        2     ┌─┐
││ │ ││       │v│
││┌┴┐││       └┬┘
│││v│││   =    │ 
││└─┘││       ┌┴┐
││   ││       │v│
              └─┘
```
In our case, $v$ is a difference between two tensors; as above, we connect the free legs, and this time, *in all possible ways*:

* $x$ with itself;
* $x$ with $W^TWx$;
* $W^TWx$ with $x$;
* $W^TWx$ with itself.

Graphically,
```{.ascii-art}
                                           ┌─┐
                                           │x│
                                           └┬┘
                                            │ 
              2            ┌─┐     ┌─┐     ┌┴┐
││       │ ││              │x│     │x│     │W│
││      ┌┴┐││              └┬┘     └┬┘     └┬┘
││      │W│││               │       │       ┼ 
││      └┬┘││      ┌─┐     ┌┴┐     ┌┴┐     ┌┴┐
││ │     ┼ ││      │x│     │W│     │W│     │W│
││┌┴┐   ┌┴┐││      └┬┘     └┬┘     └┬┘     └┬┘
│││x│ - │W│││   =   │   -   ┼   -   ┼   +   │ 
││└─┘   └┬┘││      ┌┴┐     ┌┴┐     ┌┴┐     ┌┴┐
││       │ ││      │x│     │W│     │W│     │W│
││      ┌┴┐││      └─┘     └┬┘     └┬┘     └┬┘
││      │x│││               │       │       ┼ 
││      └─┘││              ┌┴┐     ┌┴┐     ┌┴┐
││         ││              │x│     │x│     │W│
                           └─┘     └─┘     └┬┘
                                            │ 
                                           ┌┴┐
                                           │x│
                                           └─┘
```
Take a look at the terms on the right hand side: if you write down the (painfully long) summations with all the proper indices, you will find that your formula maps exactly to this graphical representation.
For instance, note the two terms in the middle: these two are the same, and they are the double product that is usually found in squares.
Even more interestingly: notice how there are now *no free legs*: by taking the L2 norm, we are left with a scalar, which is a rank-0 tensor (i.e., a leg-less box).

### Step 3 (the real kicker): the derivation

Now, let us take the derivative of the stuff above with respect to $W$:
```{.ascii-art}
    ┌─                 ─┐
    │                ┌─┐│
    │                │x││
    │                └┬┘│
    │                 │ │
    │        ┌─┐     ┌┴┐│
    │        │x│     │W││
    │        └┬┘     └┬┘│
    │         │       ┼ │
    │┌─┐     ┌┴┐     ┌┴┐│
  ┼ ││x│     │W│     │W││
 ┌┴┐│└┬┘     └┬┘     └┬┘│
∂│W││ │  - 2  ┼   +   │ │
 └┬┘│┌┴┐     ┌┴┐     ┌┴┐│
  │ ││x│     │W│     │W││
    │└─┘     └┬┘     └┬┘│
    │         │       ┼ │
    │        ┌┴┐     ┌┴┐│
    │        │x│     │W││
    │        └─┘     └┬┘│
    │                 │ │
    │                ┌┴┐│
    │                │x││
    │                └─┘│
    └─                 ─┘
```
Undoubtedly one of the most cursed things I have ever written.
Anyways, the rule now is:

> Whenever you see $W$: remove **just the block** from the graph, leaving the legs where they are.
> Whenever you see $W$, *but with differently oriented legs* (i.e., upside down): first twist the graph until you get the "proper" block, then remove it, **leaving the remaining legs as they are**.
> For graphs that contain $W$ multiple times, take away only one copy at a time, so that a graph that contains, say, 3 copies of $W$, will result in 3 graphs[^2].
> Do this separately for every graph in the expression you need to differentiate[^3].

Let's go over all the terms.

#### First term
Obviously,
```{.ascii-art}
    ┌─┐     
  ┼ │x│     
 ┌┴┐└┬┘     
∂│W│ │  = 0 
 └┬┘┌┴┐     
  │ │x│     
    └─┘     
```
since there is no $W$ in the graph.

#### Second term
Here we do have a $W$, and since it appears twice, we will get two terms:
```{.ascii-art}
       ┌─┐                                                      
       │x│                                                      
       └┬┘           ┌───┐       ┌───┐       ┌───┐              
       ┌┴┐           ┼   ┼       ┼   ┼       ┼   ┼        ┌─┐   
  ┼    │W│     ┼    ┌┴┐ ┌┴┐         ┌┴┐     ┌┴┐           │x│ │ 
 ┌┴┐   └┬┘    ┌┴┐   │W│ │W│         │W│     │W│           └┬┘┌┴┐
∂│W│ 2  ┼  = ∂│W│ 2 └┬┘ └┬┘ = 2  │  └┬┘ + 2 └┬┘  │   =  4 ┌┴┐│x│
 └┬┘   ┌┴┐    └┬┘   ┌┴┐ ┌┴┐     ┌┴┐ ┌┴┐     ┌┴┐ ┌┴┐       │W│└─┘
  │    │W│     │    │x│ │x│     │x│ │x│     │x│ │x│       └┬┘   
       └┬┘          └─┘ └─┘     └─┘ └─┘     └─┘ └─┘        ┼    
       ┌┴┐                                                      
       │x│                                                      
       └─┘                                                      
```
Let's recap the steps:

1. Rewrite the graph by twisting it around the $W$s whenever necessary;
2. Every time a (now correctly oriented) copy of $W$ appears, remove it and write the graph with the corresponding hole. **IMPORTANT**: leave the legs in the resulting orientation!
3. (optional, for the looks) rewrite the branches so that the legs do not have to "do a curve";
4. Sum together equivalent graphs.

*But Andrea* (I hear you ask), *why are output legs **inverted**? Could we be doing something wrong?*
An attempt at an explanation is given in the paragraph ahead, but feel free to skip.
The answer, anyways, is that for this specific application, *and this application only*, we can confuse up and down.

> The leg orientation being swapped is a very interesting consequence of the notation, and it has to do with the *true geometric nature of a gradient*.
> A proper answer requires a bit of background on differential geometry which I myself lack, and thus can not really explain.
> However, the gist is the following.
> We can frame the loss as a function $\mathcal{L}: \mathbb{R}^n \to\mathbb{R}$, which takes a parameter vector as input and returns a scalar.
> The gradient of this function *with respect to the parameter vector* gives a *direction* which, *if followed*, results in maximum loss change.
>
> In a sense, it is a "measuring device" that, given as input a certain "parameter displacement", tells you how much the loss changes.
> So, it is a (linear) function that, given a vector of parameter changes (which we may just as well call $\Delta W$ for $W$ being the parameter), returns a scalar.
> If you are familiar with dual spaces, that is pretty much the definition of a *covector*!
> 
> **Does this mean that when we use the gradients (a covector) to update the parameters (a vector) we are mixing things up?**
>
> In principle, **yes**.
> The maths says that the two types of objects are in general different, and cannot be simply added.
> A notable exception however is Euclidean space: here, vectors and covectors *can* be used interchangeably!
> I convinced myself of this by the following reasoning.
>
> The gradient defines a direction of maximum change.
> In Euclidean space, the result of "measuring the impact on the loss $\mathcal{L}(W)$ of a change vector $w$ by the gradient $\nabla_W \mathcal{L}$" is expressed as the usual dot product between the gradient (interpreted as a vector) and the change in parameter:
> $$\Delta\mathcal{L} = \nabla_W\mathcal{L} \cdot w.$$
> How would you maximize a dot product $a\cdot b$, if you could only change $b$?
> Easy: by *aligning it to $a$*!
> And **that** is why Euclidean space is special: in it, the dot product is maximized by alignment!
> And that is also why we drop the distinction between the gradient and the weight updates.
> This reasoning can be extended to mixed type parameters, as is done for the weight matrix we derive against.
> The result is that, in general, for a loss that depends on a $(p, q)$-type tensor, the gradient will be a $(q, p)$-type tensor: up and down are swapped!


#### Last term
Here, $W$ appears 4 times, so we will get 4 terms as output (I group them together already due to space constraints).
You can try your hand at deriving this using the rules above, and check whether it works!
```{.ascii-art}
    ┌─┐                                                                       
    │x│                                                                       
    └┬┘                                                                       
     │                                                                        
    ┌┴┐                                                          ┌─┐          
    │W│                                                          │x│     │    
    └┬┘                                                          └┬┘    ┌┴┐   
     ┼         ┌──┐  ┌──┐      ┌──┐  ┌──┐      ┌──┐  ┌──┐         │     │W│   
    ┌┴┐        ┼  ┼  ┼  ┼      ┼  ┼  ┼  ┼      ┼  ┼  ┼  ┼        ┌┴┐    └┬┘┌─┐
  ┼ │W│     ┼ ┌┴┐┌┴┐┌┴┐┌┴┐       ┌┴┐┌┴┐┌┴┐    ┌┴┐   ┌┴┐┌┴┐     │ │W│     ┼ │x│
 ┌┴┐└┬┘    ┌┴┐│W││W││W││W│       │W││W││W│    │W│   │W││W│    ┌┴┐└┬┘    ┌┴┐└┬┘
∂│W│ │  = ∂│W│└┬┘└┬┘└┬┘└┬┘ = 2   └┬┘└┬┘└┬┘ + 2└┬┘   └┬┘└┬┘ = 2│x│ ┼  + 2│W│ │ 
 └┬┘┌┴┐    └┬┘ │  │  │  │      │  │  │  │      │  │  │  │     └─┘┌┴┐    └┬┘┌┴┐
  │ │W│     │ ┌┴┐ └──┘ ┌┴┐    ┌┴┐ └──┘ ┌┴┐    ┌┴┐ └──┘ ┌┴┐       │W│     │ │W│
    └┬┘       │x│      │x│    │x│      │x│    │x│      │x│       └┬┘    ┌┴┐└┬┘
     ┼        └─┘      └─┘    └─┘      └─┘    └─┘      └─┘        │     │x│ ┼ 
    ┌┴┐                                                          ┌┴┐    └─┘   
    │W│                                                          │W│          
    └┬┘                                                          └┬┘          
     │                                                            ┼           
    ┌┴┐                                                                       
    │x│                                                                       
    └─┘                                                                       
```
### All together
Let us put all together:
```{.ascii-art}
                  2                ┌─┐          
    ││       │ ││                  │x│     │    
    ││      ┌┴┐││                  └┬┘    ┌┴┐   
    ││      │W│││                   │     │W│   
    ││      └┬┘││     ┌─┐          ┌┴┐    └┬┘┌─┐
  ┼ ││ │     ┼ ││     │x│ │      │ │W│     ┼ │x│
 ┌┴┐││┌┴┐   ┌┴┐││     └┬┘┌┴┐    ┌┴┐└┬┘    ┌┴┐└┬┘
∂│W││││x│ - │W│││ = -4 │ │x│ + 2│x│ ┼  + 2│W│ │ 
 └┬┘││└─┘   └┬┘││     ┌┴┐└─┘    └─┘┌┴┐    └┬┘┌┴┐
  │ ││       │ ││     │W│          │W│     │ │W│
    ││      ┌┴┐││     └┬┘          └┬┘    ┌┴┐└┬┘
    ││      │x│││      ┼            │     │x│ ┼ 
    ││      └─┘││                  ┌┴┐    └─┘   
    ││         ││                  │W│          
                                   └┬┘          
                                    ┼           
```
Now this may seem mysterious.
Some would even accuse this notation of proving the "maths is hieroglyphs" crowd correct.
But what if we collect a few terms?
Then we may get something like this:
```{.ascii-art}
                                           ┌─       ─┐
                  2                        │┌─┐      │
    ││       │ ││        ┌─       ─┐       ││x│      │
    ││      ┌┴┐││        │ │       │       │└┬┘      │
    ││      │W│││        │┌┴┐      │       │┌┴┐      │
    ││      └┬┘││     ┌─┐││W│      │       ││W│   ┌─┐│
  ┼ ││ │     ┼ ││     │x││└┬┘      │     │ │└┬┘   │x││
 ┌┴┐││┌┴┐   ┌┴┐││     └┬┘│ ┼     │ │    ┌┴┐│ ┼  - └┬┘│
∂│W││││x│ - │W│││ = 2 ┌┴┐│┌┴┐   ┌┴┐│ + 2│x││┌┴┐   ┌┴┐│
 └┬┘││└─┘   └┬┘││     │W│││W│ - │x││    └─┘││W│   │W││
  │ ││       │ ││     └┬┘│└┬┘   └─┘│       │└┬┘   └┬┘│
    ││      ┌┴┐││      ┼ │┌┴┐      │       │┌┴┐    ┼ │
    ││      │x│││        ││x│      │       ││W│      │
    ││      └─┘││        │└─┘      │       │└┬┘      │
    ││         ││        └─       ─┘       │ ┼       │
                                           └─       ─┘
```
Let's look at the various terms:

* the first term is just $Wx$, which we already called $y$;
* the stuff inside the first parentheses has the shape of $x$: indeed, the second term is literally just $x$.  
The first one, instead, is $W^TWx$, which we may just as well call $x^{rec}$, as it is the reconstructed version of $x$;
* the second term start with a simple $x$;
* the stuff inside the second parentheses has the shape of $y$: indeed, the second term is literally just $y$.  
The first one, instead, is $WW^TWx=Wx^{rec}$, which we may just as well call $y^{rec}$, as it is the reconstructed version of $y$.

We have just obtained [the same result as in the manual computations](/blog/pcarule.html)!
But we did it by drawing boxes, which is a nice way to avoid the monotony of algebraic manipulation.
Additionally, we got the correct variance property of the result for free.
Pretty neat!

### Bonus round: higher order derivatives

In an excess of enthusiasm, I annoyed my colleague [Victor](https://vyeoms.github.io/) by showing him some of this stuff on the whiteboard in the coffee room.
Instead of just nodding and smiling while slowly walking away (as any sane person would do), he asked me what would happen if one were to take higher order derivatives.
For instance, what about the gradient of the gradient (i.e., the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix))?

I think the notation can be applied to this case as well.
Let us consider an easy case: computing the Hessian of the scalar function $f(x) = \lVert x\rVert^2$ where $x \in \mathbb{R}^n$ is some vector.
We know the result to be $H(f)=2 \mathbb{I}_n$.
Let's try to find it with the notation we used so far.

The gradient is simply:
```{.ascii-art}
    ┌─┐        ┌───┐         
  │ │x│     │  │   │         
 ┌┴┐└┬┘    ┌┴┐┌┴┐ ┌┴┐     ┌─┐
∂│x│ │  = ∂│x││x│ │x│ = 2 │x│
 └─┘┌┴┐    └─┘└─┘ └─┘     └┬┘
    │x│                    │ 
    └─┘                      
```
And, following the methodology above, we get for the second derivative:
```{.ascii-art}
  │              │  ┌──┐     ┌──┐
 ┌┴┐   ┌─┐      ┌┴┐┌┴┐ │     │  │
∂│x│ 2 │x│ = 2 ∂│x││x│ │ = 2 │  │
 └─┘   └┬┘      └─┘└─┘ │     │  │
        │              │     │  │
```
Now, this is admittedly a bit mysterious.
We end up with a "boxless" pair of covariant legs: a 2-covariant tensor, as expected for our Hessian!
But what about the glaring lack of "tensor-ness" (that is to say, the lack of boxes in the diagram)?
I think it is safe to "rationalize" this notation *ex-post*:

> I argue that a boxless tensor is equivalent to a Kronecker delta.

Indeed, if this equation[^4] is correct,
$$
x^i = \delta^i_j x^j,
$$
which it undoubtedly is, then one should accept that
```{.ascii-art}
 │       
┌┴┐      
│δ│    │ 
└┬┘   ┌┴┐
 │  = │x│
┌┴┐   └─┘
│x│      
└─┘      
```
where the $\delta$ box is the (1,1)-type tensor denoting a Kronecker delta.
This means that we took a leg (admittedly, a box-attached leg) and added a Kronecker-box to it for free.
Thus I argue that a "floating leg" (like the one resulting from the Hessian computations above) is really just a Kronecker delta in disguise.

Of course, this is an egregious example of *handwaviness* and *sloppy thinking*.
If you are as annoyed by this as I am, but have a better solution, please let me know at [my email](mailto:andrea.perin@aalto.fi).

[^1]: But *only* of linear operations.
[^2]: This is basically the product rule for the derivative.
[^3]: And this is linearity of the derivative.
[^4]: Using proper Einstein notation, with covariance and contravariance meanings associated to sub/superscripts respectively.

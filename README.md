# phase-accumulation

This is a project investigating the potential of phase-restricted masks to accumulate to full
range phase masks, or even arbitrary unitary transforms.

A recent technology that this project is based off of is Multi-Plane Light Conversion (MPLC),
which uses a series of sucessive phase masks to try and implement arbitrary unitary transforms.
Between any pair of phase masks, the free space propagation of light results in the mixing of
various modes, thus with a sufficient number of phase masks we are able to design MPLC systems 
which implement complex unitary transformations.

We are interested in investigating whether this technique can be used with phase masks that do
not have the full [0, 2π] phase range that you would typically desire when working with phase 
masks.  The motivation behind this investigation is the fact a spatial light modulator's (SLM)
speed is dependent on the range of phases that it is capable of imparting.  In particular, for 
the case of Liquid Crystal SLMs, the speed is inversely proportional to the phase range squared.
Consequently, if you designed a SLM with only a [0, 2π/10] phase range, it would be 100x faster
than its full [0, 2π] phase range counterpart.  However, this is only useful if we can combine
several of these [0, 2π/10] phase range SLMs to build up to arbitrary phase masks.
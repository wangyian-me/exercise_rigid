# exercise_rigid
it's a course project for "Physics Simulation for Computer Graphics"

## Env

I use python3.8 environment with taichi 1.4.1.

## Results

I implemented the dynamics of 3D rigid object in taichi, and simple collision detection and impulse based collision response method for 3D boxes. I haven't implemented demo 4 for multi-body system with ground (still have some confusion with it).

In order to enable edge-edge collision detaction, I simple interpolate some point on each edges and use them as vertex to compute impulse. For the impulse between 2 rigid objects, if there are multiple collision pair, I take the average of the resulting impulses. (I think this will not work when it comes to demo 4....)

### Demo 1

The output is

```
linear vel: [0.99950027 0.99950027 0.        ]
angle vel: [-2.39880085  4.9155755  -1.76382422]
vel of point (-0.3, -0.5, -0.25): [-4.63719131  4.45483292 17.29551543]
```

### Demo 2

This is the result when I apply a force equal to its gravity on the bottom-front-left point.
<p align="center">
    <img src="imgs/GIF_1_gravity.gif", height=480>
</p>

This is the result when I apply a force equal to its gravity on the bottom-front-left point, but there is no gravity.
<p align="center">
    <img src="imgs/GIF_1_nogravity.gif", height=480>
</p>

### Demo 3

This is the result when I apply a force equal to its gravity on the bottom-front-left point for each object.
<p align="center">
    <img src="imgs/GIF_3_0.gif.gif", height=480>
</p>

This is the result when I apply another attraction to the point (0, 0, 0.3) on the bottom-front-right point.
<p align="center">
    <img src="imgs/GIF_3_1.gif.gif", height=480>
</p>

Noticed that the edge-edge collision is well modeled.


### Demo 4

To be continue...

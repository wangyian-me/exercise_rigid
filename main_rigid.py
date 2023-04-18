import taichi as ti
import argparse
import os
from rigid_box import Box
import imageio
import numpy as np

ti.init(arch=ti.cpu, default_ip=ti.i32)

dt = 0.01
Len_x = 1
Len_y = 0.6
Len_z = 0.5
c = 0.5
n_verts = 8

pos = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
pos1 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

box = Box(dt, Len_x, Len_y, Len_z)
box1 = Box(dt, Len_x, Len_y, Len_z)
indices = ti.field(ti.i32, shape=36)
# elastic1 = Box(n_verts, dt, n_verts, 0.5)

# manipulate_force = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

def get_indices():
    list_face = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    for i in range(6):
        a, b, c, d = list_face[i]
        indices[i * 2 * 3] = a
        indices[i * 2 * 3 + 1] = b
        indices[i * 2 * 3 + 2] = c

        indices[(i * 2 + 1) * 3] = a
        indices[(i * 2 + 1) * 3 + 1] = c
        indices[(i * 2 + 1) * 3 + 2] = d

@ti.kernel
def update_pos():
    for i in range(n_verts):
        pos[i] = box.F_x_world[i]
        pos1[i] = box1.F_x_world[i]

@ti.kernel
def get_contact(cnt1: ti.template(), cnt2: ti.template(), impulse1: ti.template(), impulse2: ti.template(), ang_impulse1: ti.template(), ang_impulse2: ti.template(),
                       pos1: ti.template(), center1: ti.template(),center2: ti.template(), xl: ti.f32, yl: ti.f32, zl: ti.f32, F_x1: ti.template(), rotmat2: ti.template(), vel1: ti.template(),
                vel2: ti.template(), w1: ti.template(), w2: ti.template(), mass1: ti.f32, mass2: ti.f32, iner_inv1: ti.template(), iner_inv2: ti.template()):
    for i in pos1:
        dist = rotmat2[None].transpose() @ (pos1[i] - center2[None])
        if ti.abs(dist[0]) < xl and ti.abs(dist[1]) < yl and ti.abs(dist[2]) < zl:

            now_dist = xl - ti.abs(dist[0])
            normal = ti.Vector([1.0, 0., 0.])
            if dist[0] < 0:
                normal = ti.Vector([-1.0, 0., 0.])
            if yl - ti.abs(dist[1]) < now_dist:
                normal = ti.Vector([0., 1.0, 0.])
                now_dist = yl - ti.abs(dist[1])
                if dist[1] < 0:
                    normal = ti.Vector([0., -1.0, 0.])
            if zl - ti.abs(dist[2]) < now_dist:
                normal = ti.Vector([0., 0., 1.0])
                now_dist = zl - ti.abs(dist[2])
                if dist[2] < 0:
                    normal = ti.Vector([0., 0., -1.0])

            normal = rotmat2[None] @ normal
            v1 = vel1[None] + w1[None].cross(pos1[i] - center1[None])
            v2 = vel2[None] + w2[None].cross(pos1[i] - center2[None])
            v_rel = normal.dot(v1 - v2)

            if v_rel < 0:
                cnt1[None] += 1.0
                cnt2[None] += 1.0
                x1 = pos1[i] - center1[None]
                x2 = pos1[i] - center2[None]
                J = - (1 + c) * v_rel / (1.0 / mass1 + 1.0 / mass2 + ((iner_inv1[None] @ x1.cross(normal)).cross(x1) + (iner_inv2[None] @ x2.cross(normal)).cross(x2)).dot(normal))
                impulse1[None] += J * normal
                impulse2[None] -= J * normal
                ang_impulse1[None] += x1.cross(J * normal)
                ang_impulse2[None] -= x2.cross(J * normal)


substeps = 100


# p_contact[0].deactivate()
# p_contact[0].append(pair(-1, -1))
def main(args):

    window = ti.ui.Window("Taichi Paper Simulation on GGUI", (480, 480),
                          vsync=True, show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    center = ti.Vector([0., 0., 0.])
    quat = ti.Vector([1.0, 0., 0., 0.])
    box.init(center, quat)
    center = ti.Vector([0., 0., 0.7])
    quat = ti.Vector([1.0, 0., 0., 0.])
    box1.init(center, quat)

    get_indices()
    update_pos()

    # box.force[None] = ti.Vector([1., 1., 0.])
    # xi = ti.Vector([0.3, 0.5, 0.25])
    # box.torque[None] = xi.cross(box.force[None])
    # box.step_simple()
    # print("vel:", box.vel[None])
    # print("angle_vel:", box.angle_vel[None])
    # yi = ti.Vector([-0.3, -0.5, -0.25])
    # vi = box.vel[None] + box.angle_vel[None].cross(yi - box.pos[None])
    # print("vel of point:", vi)

    tot_step = 0
    save_path = f"imgs/box_collision"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    step_cnt = 0
    frames = []

    while window.running:

        tot_step += 1

        if tot_step > 300:
            break

        box.F_f[4] = ti.Vector([0, 0, 10])
        box1.F_f[4] = ti.Vector([0, 0, 10])
        box.F_f[5] = (ti.Vector([0, 0, 0.3]) - box.F_x_world[5]) * 10
        box1.F_f[5] = (ti.Vector([0, 0, 0.3]) - box1.F_x_world[5]) * 10
        box.clear_impulse()
        box1.clear_impulse()
        box.get_rotmat()
        box1.get_rotmat()

        get_contact(box.collision_count, box1.collision_count, box.collision_impulse, box1.collision_impulse,
                    box.collision_torque_impulse, box1.collision_torque_impulse,
                    box.F_x_world, box.pos, box1.pos, box1.lenx / 2, box1.leny / 2, box1.lenz / 2, box.F_x, box1.rotmat,
                    box.vel, box1.vel, box.angle_vel, box1.angle_vel,
                    box.mass, box1.mass, box.inert_inv, box1.inert_inv)
        get_contact(box1.collision_count, box.collision_count, box1.collision_impulse, box.collision_impulse,
                    box1.collision_torque_impulse, box.collision_torque_impulse,
                    box1.F_x_world, box1.pos, box.pos, box.lenx / 2, box.leny / 2, box.lenz / 2, box1.F_x, box.rotmat,
                    box1.vel, box.vel, box1.angle_vel, box.angle_vel,
                    box1.mass, box.mass, box1.inert_inv, box.inert_inv)

        box.step()
        box1.step()

        update_pos()
        # print(pos[0], vertices[0])
        # print(tot_step, ":", vertices[0])
        # for i, j in ti.ndrange(N + 1, N + 1):
        #     if(vertices[i * (N + 1) + j][2] > 0.001):
        #         print("i:", i, "j:", j, "xyz:", vertices[i * (N + 1) + j])

        camera.position(0, -2, 3)
        camera.lookat(0, 0, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 0, 3), color=(1, 1, 1))
        scene.ambient_light((0, -2, 2))

        scene.mesh(pos, indices, color=(0.73, 0.33, 0.23))
        scene.mesh(pos1, indices, color=(0.73, 0.33, 0.23))
        # print("!!!")
        # Draw a smaller ball to avoid visual penetration
        canvas.scene(scene)

        window.save_image(os.path.join(save_path, f"{tot_step}.png"))

        # window.show()

    for i in range(1, tot_step):
        filename = os.path.join(save_path, f"{i}.png")
        frames.append(imageio.imread(filename))

    gif_name = os.path.join(save_path, f"GIF_{args.exp_id}.gif")
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=0)
    main(parser.parse_args())

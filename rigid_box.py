import argparse

import numpy as np

import taichi as ti


@ti.data_oriented
class Box:
    def __init__(self, dt, Len_x, Len_y, Len_z):

        self.density = 6.67
        self.dt = dt
        self.mass = Len_x * Len_y * Len_z * self.density
        self.lenx = Len_x
        self.leny = Len_y
        self.lenz = Len_z
        n_verts = 8
        self.n_verts = n_verts
        n_faces = 12
        self.n_faces = 12
        self.F_vertices = ti.Vector.field(3, dtype=ti.i32, shape=n_faces)
        n_edges = 12
        self.n_edges = 12
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=n_edges)

        #   3- 2
        #  /  / |
        # 0- 1  6
        # |  | /
        # 4- 5
        self.n_sep = 9

        self.F_x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts + 12 * self.n_sep)
        self.F_v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.F_f = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.F_x_world = ti.Vector.field(3, dtype=ti.f32, shape=n_verts + 12 * self.n_sep)

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.rot = ti.Vector.field(4, dtype=ti.f32, shape=())
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.angle_vel = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.inert = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.inert_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())


        self.rotmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

        self.collision_count = ti.field(dtype=ti.f32, shape=())
        self.collision_impulse = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.collision_torque_impulse = ti.Vector.field(3, dtype=ti.f32, shape=())

    # @ti.kernel
    # def init_faces(self):
    #     faces = [[0, 1, 5], [0, 4, 5], [3, 2, 6], [3, 7, 6], [0, 3, 2], [0, 1, 2], [4, 7, 6], [4, 5, 6], [1, 2, 6], [1, 5, 6], [0, 4, 7], [0, 3, 7]]
    #     for i in range(self.n_faces):
    #         self.F_vertices[i] = ti.Vector(faces[i])

    @ti.kernel
    def init_vertices(self):
        self.F_v.fill(0)
        self.F_f.fill(0)
        x = self.lenx * 0.5
        y = self.leny * 0.5
        z = self.lenz * 0.5
        self.F_x[0] = ti.Vector([-x, -y, z])
        self.F_x[1] = ti.Vector([x, -y, z])
        self.F_x[2] = ti.Vector([x, y, z])
        self.F_x[3] = ti.Vector([-x, y, z])
        self.F_x[4] = ti.Vector([-x, -y, -z])
        self.F_x[5] = ti.Vector([x, -y, -z])
        self.F_x[6] = ti.Vector([x, y, -z])
        self.F_x[7] = ti.Vector([-x, y, -z])
        for i in range(4):
            self.edges[i] = ti.Vector([i, i + 4])
            self.edges[i + 4] = ti.Vector([i, (i + 1) % 4])
            self.edges[i + 8] = ti.Vector([i + 4, (i + 4 + 1) % 8])

        for i, j in ti.ndrange(self.n_sep, self.n_edges):
            self.F_x[8 + i * self.n_edges + j] = self.F_x[self.edges[i][0]] + (self.F_x[self.edges[i][1]] - self.F_x[self.edges[i][0]]) * (i + 1) / (self.n_sep + 1)

    @ti.kernel
    def init_pos(self, center: ti.template(), quat: ti.template()):
        self.pos[None] = center
        self.rot[None] = quat

    def init(self, center: ti.template(), quat: ti.template()):
        # self.init_faces()
        self.init_pos(center, quat)
        self.init_vertices()
        Len_x = self.lenx
        Len_y = self.leny
        Len_z = self.lenz
        self.inert[None] = ti.Matrix([[self.mass * (Len_y * Len_y + Len_z * Len_z) / 12, 0, 0],
                                      [0, self.mass * (Len_x * Len_x + Len_z * Len_z) / 12, 0],
                                      [0, 0, self.mass * (Len_x * Len_x + Len_y * Len_y) / 12]])
        self.inert_inv[None] = self.inert[None].inverse()
        self.get_rotmat()
        self.get_vert_pos()

    @ti.kernel
    def get_vert_pos(self):
        for i in self.F_x_world:
            self.F_x_world[i] = self.pos[None] + self.rotmat[None] @ self.F_x[i]

    def get_rotmat(self):
        s = self.rot[None][0]
        x = self.rot[None][1]
        y = self.rot[None][2]
        z = self.rot[None][3]
        self.rotmat[None] = ti.Matrix([[s*s+x*x-y*y-z*z, 2*(x*y-s*z), 2*(x*z+s*y)],
                                 [2*(x*y+s*z), s*s-x*x+y*y-z*z, 2*(y*z-s*x)],
                                 [2*(x*z-s*y), 2*(y*z+s*x), s*s-x*x-y*y+z*z]])

    def step_simple(self):
        self.vel[None] += self.dt * self.force[None] / self.mass
        self.pos[None] += self.vel[None] * self.dt
        self.get_rotmat()
        inertia_inv = (self.rotmat[None] @ self.inert[None] @ self.rotmat[None].transpose()).inverse()
        self.angle_vel[None] += self.dt * inertia_inv @ self.torque[None]
        v2 = ti.Vector([self.rot[None][1], self.rot[None][2], self.rot[None][3]])
        real = -self.angle_vel[None].dot(v2)
        res = self.rot[None][0] * self.angle_vel[None] + self.angle_vel[None].cross(v2)
        self.rot[None][0] += real * self.dt / 2
        self.rot[None][1] += res[0] * self.dt / 2
        self.rot[None][2] += res[1] * self.dt / 2
        self.rot[None][3] += res[2] * self.dt / 2
        self.rot[None] = self.rot[None].normalized()
        self.get_rotmat()
        self.get_vert_pos()

    @ti.kernel
    def gather_force_and_torque(self):
        self.force.fill(0)
        self.torque.fill(0)
        for i in self.F_f:
            self.force[None] += self.F_f[i]
            self.torque[None] += (self.rotmat[None] @ self.F_x[i]).cross(self.F_f[i])

    def step_no_collision(self):
        self.get_rotmat()
        self.gather_force_and_torque()
        self.force[None][2] -= 10
        self.step_simple()

    def step(self):
        self.get_rotmat()
        self.gather_force_and_torque()
        self.force[None][2] -= 10
        self.vel[None] += self.dt * self.force[None] / self.mass

        inertia_inv = (self.rotmat[None] @ self.inert[None] @ self.rotmat[None].transpose()).inverse()
        self.angle_vel[None] += self.dt * inertia_inv @ self.torque[None]

        if self.collision_count[None] > 0.5:
            self.vel[None] += self.collision_impulse[None] / self.mass / self.collision_count[None]
            self.angle_vel[None] += inertia_inv @ self.collision_torque_impulse[None] / self.collision_count[None]

        self.pos[None] += self.vel[None] * self.dt
        v2 = ti.Vector([self.rot[None][1], self.rot[None][2], self.rot[None][3]])
        real = -self.angle_vel[None].dot(v2)
        res = self.rot[None][0] * self.angle_vel[None] + self.angle_vel[None].cross(v2)
        self.rot[None][0] += real * self.dt / 2
        self.rot[None][1] += res[0] * self.dt / 2
        self.rot[None][2] += res[1] * self.dt / 2
        self.rot[None][3] += res[2] * self.dt / 2
        self.rot[None] = self.rot[None].normalized()
        self.get_rotmat()
        self.get_vert_pos()

    def contact_with_ground(self):
        for i in range(self.n_verts):
            if self.F_x_world[i][2] < 0:
                normal = ti.Vector([0., 0., 1.0])


    def clear_impulse(self):
        self.collision_count.fill(0)
        self.collision_impulse.fill(0)
        self.collision_torque_impulse.fill(0)



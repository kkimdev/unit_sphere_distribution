#![crate_name = "unit_sphere_distribution"]
#![crate_type = "rlib"]

#![feature(core)]
#![feature(convert)]
#![feature(std_misc)]
#![feature(alloc)]
#![feature(thread_local)]
#![feature(libc)]
#![feature(plugin, custom_attribute)]

#![plugin(gfx_macros)]

extern crate piston;
extern crate piston_window;
extern crate vecmath;
extern crate camera_controllers;
extern crate gfx;
extern crate gfx_device_gl;
extern crate sdl2_window;
extern crate libc;
#[macro_use]
extern crate autograd;

mod ipopt;
pub mod optimization;
pub mod display;

pub fn normalize<T>(p: &[T; 3]) -> [T; 3] where T: std::num::Float {
    let norm = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
    [p[0] / norm, p[1] / norm, p[2] / norm]
}

pub fn zip<T>(x: &[T]) -> Box<[[T; 2]]> where T: std::num::Float {
    let mut result = Vec::new();
    for i in 0..x.len()/2 {
        result.push([x[i*2], x[i*2+1]]);
    }
    result.into_boxed_slice()
}

pub fn extract<T>(x: &[[T; 2]]) -> Box<[T]> where T: std::num::Float {
    let mut result = Vec::new();
    for i in x {
        result.push(i[0]);
        result.push(i[1]);
    }
    result.into_boxed_slice()
}

pub fn to_cartesian_coordinate<T>(p: &[T; 2]) -> [T; 3] where T: std::num::Float {
    [
        p[0].sin() * p[1].cos(),
        p[0].sin() * p[1].sin(),
        p[0].cos(),
    ]
}

pub fn to_spherical_coordinate<T>(p: &[T; 3]) -> [T; 2] where T: std::num::Float {
    [
        (p[2] / (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt()).acos(),
        (p[1] / p[0]).atan(),
    ]
}

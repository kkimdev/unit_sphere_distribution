#![feature(collections)]

extern crate unit_sphere_distribution;
extern crate piston;
extern crate piston_window;
extern crate rand;
extern crate vecmath;
extern crate camera_controllers;
extern crate gfx;
extern crate sdl2_window;
extern crate time;

use unit_sphere_distribution::*;
use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::mpsc::channel;
use std::rc::Rc;
use std::thread;
use std::sync::Arc;
use piston::window::{ AdvancedWindow, WindowSettings };
use piston_window::*;
use piston::input::keyboard::Key;
use piston::input::Button::Keyboard;
use piston::event::PressEvent;
use sdl2_window::{ Sdl2Window, OpenGL };


struct OptimizationData {
    request_id: usize,
    points: Box<[[f64;3]]>
}

impl std::clone::Clone for OptimizationData {
    fn clone(&self) -> Self {
        OptimizationData {
            request_id: self.request_id,
            points: (*self.points).to_vec().into_boxed_slice(),
        }
    }
}

struct OptimizationRunner {
    tx: std::sync::mpsc::Sender<OptimizationData>,
    last_request_id: Arc<Mutex<usize>>,
    thread_handle: std::thread::JoinHandle<()>,
    result : Arc<Mutex<Option<OptimizationData>>>,
}

impl OptimizationRunner {
    pub fn new() -> OptimizationRunner {
        let (tx, rx) = channel::<OptimizationData>();

        let last_request_id = Arc::new(Mutex::new(0));
        let last_request_id_cloned = last_request_id.clone();
        let result = Arc::new(Mutex::new(None));
        let result_cloned = result.clone();

        let handle = thread::spawn(move || {
            for request in rx.iter() {
                if request.request_id != *last_request_id.lock().unwrap() {
                    continue;
                }
                *result.lock().unwrap() = Some(OptimizationData {
                    request_id: request.request_id,
                    points: optimization::optimize(&*request.points)
                })
            }
        });

        OptimizationRunner {
            tx: tx,
            last_request_id: last_request_id_cloned,
            thread_handle: handle,
            result: result_cloned,
        }
    }

    pub fn request_optimization(&mut self, points: Box<[[f64; 3]]>) {
        *self.last_request_id.lock().unwrap() += 1;
        self.tx.send(OptimizationData {
            points: points,
            request_id: *self.last_request_id.lock().unwrap()});
    }

    pub fn get_latest_result(&mut self) -> Option<Box<[[f64;3]]>> {
        let result = (*self.result.lock().unwrap()).clone();
        match result {
            Some(data) => {
                if data.request_id == *self.last_request_id.lock().unwrap() {
                    (*self.result.lock().unwrap()) = None;
                    return Some(data.points);
                } else {
                    return None;
                }
            },
            None => {return None},
        }
    }
}

fn main () {
    let window = Rc::new(RefCell::new(Sdl2Window::new(
        OpenGL::_3_2,
        WindowSettings::new("", [1280, 960]).exit_on_esc(true).samples(4)
        ).capture_cursor(false)));
    let events = PistonWindow::new(window, empty_app());
    let mut display = display::Display::new(&events);

    let mut points: Box<[[f64; 3]]> = Box::new([[1.0, 0.0, 0.0]]);
    let mut target_points: Box<[[f64; 3]]> = Box::new([[0.0, 1.0, 0.0]]);
    let mut optimization_runner = OptimizationRunner::new();
    let mut last_time_ns = time::precise_time_ns();

    for event in events {
        let now = time::precise_time_ns();
        let time_diff_ns = now - last_time_ns;
        last_time_ns = now;

        match optimization_runner.get_latest_result() {
            Some(data) => {
                target_points = data;
            }
            _ => {}
        }

        let diff = (target_points.len() as isize) - (points.len() as isize);
        if diff > 0 {
            let mut vec = points.to_vec();
            for i in (0..diff as usize) {
                vec.push(target_points[target_points.len() - diff as usize + i]);
            }
            points = vec.into_boxed_slice();
        } else if diff < 0 {
            let mut vec = points.to_vec();
            vec.resize(target_points.len(), [0.0, 0.0, 0.0]);
            points = vec.into_boxed_slice();
        }

        let mut step_size = time_diff_ns as f64 * 7e-9;
        if step_size > 0.03 {
            step_size = 0.03
        }

        // interpolate to target smoothly.
        for i in (0..points.len()) {
            let point = [
                points[i][0] + (target_points[i][0] - points[i][0]) * step_size,
                points[i][1] + (target_points[i][1] - points[i][1]) * step_size,
                points[i][2] + (target_points[i][2] - points[i][2]) * step_size,
            ];
            points[i] = normalize(&point);
        }

        event.press(|button| {
            match button {
                x if x ==  Keyboard(Key::Q) => {
                    // Add a point.
                    let mut new_points = points.to_vec();
                    let new_point = normalize(&[rand::random(), rand::random(), rand::random()]);
                    new_points.push(new_point);
                    optimization_runner.request_optimization(new_points.into_boxed_slice());
                },
                x if x ==  Keyboard(Key::W) => {
                    // Remove a point.
                    if points.len() > 1 {
                        let mut new_points = points.to_vec();
                        new_points.pop();
                        optimization_runner.request_optimization(new_points.into_boxed_slice());
                    }
                },
                _ => {}
            }
        });

        display.draw(&event, &points);
    }
}

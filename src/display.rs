use gfx;
use std::cell::RefCell;
use std;
use std::rc::Rc;
use vecmath;
use piston_window::*;
use piston::event::*;
use piston::window::{ AdvancedWindow, WindowSettings, Window };
use camera_controllers::{
    OrbitZoomCameraSettings,
    OrbitZoomCamera,
    CameraPerspective,
    model_view_projection
};
use gfx_device_gl;
use gfx::Resources;
use gfx::traits::*;
use sdl2_window::{ Sdl2Window, OpenGL };

#[vertex_format]
#[derive(Copy, Clone)]
struct Vertex {
    #[as_float]
    a_pos: [f32; 3],
    color: [f32; 4],
}

impl Vertex {
    fn new(pos: [f32; 3], color: [f32; 4]) -> Vertex {
        Vertex {
            a_pos: pos,
            color: color,
        }
    }
}

#[shader_param]
struct Params<R: gfx::Resources> {
    u_model_view_proj: [[f32; 4]; 4],
    phantom_resource: std::marker::PhantomData<R>,
}

pub struct Display {
    unit_sphere_mesh: gfx::render::mesh::Mesh<gfx_device_gl::Resources>,
    point_sphere_mesh: gfx::render::mesh::Mesh<gfx_device_gl::Resources>,
    slice: gfx::render::mesh::Slice<gfx_device_gl::Resources>,
    program: gfx::device::handle::Program<gfx_device_gl::Resources>,
    orbit_zoom_camera: OrbitZoomCamera,
    state: gfx::DrawState,
}

impl Display {
    fn avg(p1: &[f32; 3], p2: &[f32; 3]) -> [f32; 3] {
        return [
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        ]
    }

    pub fn new<W>(window : &PistonWindow<W>) -> Display where W: Window {

        // octahedron
        let mut sphere = vec![
            [[0.,  1., 0.], [ 0., 0.,  1.], [ 1., 0.,  0.]],
            [[0.,  1., 0.], [ 1., 0.,  0.], [ 0., 0., -1.]],
            [[0.,  1., 0.], [ 0., 0., -1.], [-1., 0.,  0.]],
            [[0.,  1., 0.], [-1., 0.,  0.], [ 0., 0.,  1.]],
            [[0., -1., 0.], [ 1., 0.,  0.], [ 0., 0.,  1.]],
            [[0., -1., 0.], [ 0., 0., -1.], [ 1., 0.,  0.]],
            [[0., -1., 0.], [-1., 0.,  0.], [ 0., 0., -1.]],
            [[0., -1., 0.], [ 0., 0.,  1.], [-1., 0.,  0.]],
        ];

        // Triangle subdivision
        for _ in 0..5 {
            let mut sphere_refined = Vec::new();
            for j in sphere {
                let m0 = Self::avg(&j[0], &j[1]);
                let m1 = Self::avg(&j[1], &j[2]);
                let m2 = Self::avg(&j[2], &j[0]);
                sphere_refined.push([j[0], m0, m2]);
                sphere_refined.push([m0, j[1], m1]);
                sphere_refined.push([m2, m1, j[2]]);
                sphere_refined.push([m0, m1, m2]);
            }
            sphere = sphere_refined;
        }

        let mut unit_sphere_vertex_data = Vec::new();
        let mut point_sphere_vertex_data = Vec::new();
        for i in sphere.iter() {
            for j in i.iter() {
                unit_sphere_vertex_data.push(Vertex::new(
                    ::normalize(j), [0.2, 1.0, 0.8, 0.5]));
                point_sphere_vertex_data.push(Vertex::new(vecmath::vec3_scale(
                    ::normalize(j), 0.07), [1.0, 0.0, 0.0, 0.8]));
            }
        }
        let mut index_data = (0..(sphere.len()*3) as u32).collect::<Vec<_>>();

        let unit_sphere_mesh = window.canvas.borrow_mut().factory.create_mesh(&unit_sphere_vertex_data);
        let point_sphere_mesh = window.canvas.borrow_mut().factory.create_mesh(&point_sphere_vertex_data);
        let slice = window.canvas.borrow_mut().factory.create_buffer_index(&index_data.as_slice())
                           .to_slice(gfx::PrimitiveType::TriangleList);

        let program = {
            let canvas = &mut *window.canvas.borrow_mut();
            let vertex = gfx::ShaderSource {
                glsl_150: Some(include_bytes!("cube_150.glslv")),
                .. gfx::ShaderSource::empty()
            };
            let fragment = gfx::ShaderSource {
                glsl_150: Some(include_bytes!("cube_150.glslf")),
                .. gfx::ShaderSource::empty()
            };
            canvas.factory.link_program_source(vertex, fragment,
                &canvas.device.get_capabilities()).unwrap()
        };

        let mut camera_settings = OrbitZoomCameraSettings::default();
        camera_settings.orbit_speed = 0.005;
        camera_settings.pan_speed = 0.005;
        camera_settings.pitch_speed = -1.0;
        camera_settings.zoom_speed = 0.005;
        let mut orbit_zoom_camera = OrbitZoomCamera::new(
            [0.0, 0.0, 0.0],
            camera_settings
        );
        orbit_zoom_camera.distance = 3.0;
        let state = gfx::DrawState::new().blend(gfx::BlendPreset::Alpha);

        return Display{ unit_sphere_mesh: unit_sphere_mesh,
                        point_sphere_mesh: point_sphere_mesh,
                        slice: slice,
                        program: program,
                        orbit_zoom_camera: orbit_zoom_camera,
                        state: state, }
    }

    pub fn draw<W, T>(&mut self, e : &PistonWindow<W, T>,
                      points: &[[f64; 3]])
            where W: Window, W::Event: GenericEvent {
        self.orbit_zoom_camera.event(e);

        e.draw_3d(|canvas| {
            let args = e.render_args().unwrap();

            canvas.renderer.clear(
                gfx::ClearData {
                    color: [0.1, 0.1, 0.1, 1.0],
                    depth: 1.0,
                    stencil: 0,
                },
                gfx::COLOR | gfx::DEPTH,
                &canvas.output
            );

            let (width, height) = canvas.output.get_size();
            let projection = CameraPerspective {
                fov: 45.0,
                near_clip: 0.1,
                far_clip: 1000.0,
                aspect_ratio: (width as f32) / (height as f32),
            }.projection();
            let mvp = model_view_projection(
                vecmath::mat4_id(),
                self.orbit_zoom_camera.camera(args.ext_dt).orthogonal(),
                projection,
            );
            let state = gfx::DrawState::new()
                .blend(gfx::BlendPreset::Alpha)
                .depth(gfx::state::Comparison::LessEqual, true);

            for p in points {
                let mut translation_mat = vecmath::mat4_id();
                translation_mat[3][0] = p[0] as f32;
                translation_mat[3][1] = p[1] as f32;
                translation_mat[3][2] = p[2] as f32;
                let data = Params {
                    u_model_view_proj: vecmath::col_mat4_mul(mvp, translation_mat),
                    phantom_resource: std::marker::PhantomData,
                };

                canvas.renderer.draw(&(&self.point_sphere_mesh,
                                        self.slice.clone(),
                                        &self.program,
                                        &data,
                                        &state),
                                     &canvas.output).unwrap();
            }

            let data = Params {
                u_model_view_proj: mvp,
                phantom_resource: std::marker::PhantomData,
            };

            canvas.renderer.draw(&(&self.unit_sphere_mesh,
                                    self.slice.clone(),
                                    &self.program,
                                    &data,
                                    &state),
                                 &canvas.output).unwrap();
        });
    }
}

use ipopt;
use autograd::Context;
use std;
use std::raw::Repr;

// Unused
extern "C" fn g(
    n: ipopt::Index,
    x: *mut ipopt::Number,
    new_x: ipopt::Bool,
    m: ipopt::Index,
    g: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    false as ipopt::Bool
}

// Unused
extern "C" fn jac_g(
    n: ipopt::Index,
    x: *mut ipopt::Number,
    new_x: ipopt::Bool,
    m: ipopt::Index,
    nele_jac: ipopt::Index,
    iRow: *mut ipopt::Index,
    jCol: *mut ipopt::Index,
    values: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    false as ipopt::Bool
}

// Unused
extern "C" fn h(
    n: ipopt::Index,
    x: *mut ipopt::Number,
    new_x: ipopt::Bool,
    obj_factor: ipopt::Number,
    m: ipopt::Index,
    lambda: *mut ipopt::Number,
    new_lambda: ipopt::Bool,
    nele_hess: ipopt::Index,
    iRow: *mut ipopt::Index,
    jCol: *mut ipopt::Index,
    values: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    false as ipopt::Bool
}

extern "C" fn f(
    n: ipopt::Index,
    x: *mut ipopt::Number,
    new_x: ipopt::Bool,
    obj_value: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr) -> ipopt::Bool {

    unsafe{
        let array = ::zip(std::slice::from_raw_parts(x, n as usize));
        *obj_value = func(&*array).unwrap();
    }

    return true as ipopt::Bool;
}

extern "C" fn grad(
    n: ipopt::Index,
    x: *mut ipopt::Number,
    new_x: ipopt::Bool,
    grad_f: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr) -> ipopt::Bool {

    unsafe{
        let mut x_slice = std::slice::from_raw_parts_mut(x, n as usize);
        let context = new_autograd_context!(f64, (10 * n * n) as usize);
        let mut x_autograd = Vec::new();
        for i in x_slice.iter() {
            x_autograd.push(context.new_variable(*i));
        }
        let array = ::zip(x_autograd.as_slice());
        let y = func(&*array).unwrap();
        context.differentiate(y);

        let mut i = 0;
        let mut grad_f_slice = std::slice::from_raw_parts_mut(grad_f, n as usize);
        for x in x_autograd {
            grad_f_slice[i] = context.get_derivative(x);
            i += 1;
        }
    }
    return true as ipopt::Bool;
}

fn func<T>(x : &[[T; 2]]) -> Option<T> where T :
        std::num::Float +
        std::ops::Add<f64, Output = T> +
        std::ops::Sub<f64, Output = T> +
        std::ops::Mul {
    // A hack to get 0 when T is Autograd.
    let mut sum = x[0][0] - x[0][0];

    for i in x {
        for j in x {
            let p = ::to_cartesian_coordinate(i);
            let q = ::to_cartesian_coordinate(j);
            let c = p[0]*q[0] + p[1]*q[1] +p[2]*q[2] + 1.0;
            let cc = c*c;
            sum = sum + cc*cc*cc;
        }
    }

    return Some(sum);
}

pub fn optimize(x_input : &[[f64; 3]]) -> Box<[[f64; 3]]> {
    unsafe {
        let mut x_spherical = Vec::new();
        for p in x_input.iter() {
            x_spherical.push(::to_spherical_coordinate(p));
        }
        let mut x = ::extract(x_spherical.as_slice());
        let mut x_L = vec![-std::f64::INFINITY; x.len()];
        let mut x_U = vec![ std::f64::INFINITY; x.len()];
        let mut obj_val: ipopt::Number = 0.0;
        let problem = ipopt::CreateIpoptProblem(
            x.len() as i32, // n
            x_L.as_mut_slice().as_mut_ptr(),
            x_U.as_mut_slice().as_mut_ptr(),
            0, // m
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0, // nele_jac
            0,
            0,
            Some(f), // eval_f
            Some(g), // eval_g
            Some(grad), // eval_grad_f
            Some(jac_g),
            Some(h)
            );
        assert!(problem != std::ptr::null_mut());

        ipopt::AddIpoptStrOption(problem,
            "hessian_approximation\0".as_ptr() as *mut i8,
            "limited-memory\0".as_ptr() as *mut i8);

        let status = ipopt::IpoptSolve(
            problem,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj_val,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            );

        ipopt::FreeIpoptProblem(problem);

        let mut result = Vec::new();
        for p in ::zip(&*x).iter() {
            result.push(::to_cartesian_coordinate(p));
        }

        result.into_boxed_slice()
    }
}

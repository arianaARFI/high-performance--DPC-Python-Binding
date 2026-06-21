#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"

namespace py = pybind11;

std::string get_device_name(const sycl::device &dev) {
    return dev.get_info<sycl::info::device::name>();
}

int opposite_direction(int k) {
    if (k == 1) return 3;
    if (k == 2) return 4;
    if (k == 3) return 1;
    if (k == 4) return 2;
    if (k == 5) return 7;
    if (k == 6) return 8;
    if (k == 7) return 5;
    if (k == 8) return 6;
    return 0;
}

py::dict run_minilb_step(const sycl::device &dev, int nx, int ny) {
    if (nx <= 2 || ny <= 2) {
        throw std::runtime_error("nx and ny must be greater than 2");
    }

    sycl::queue q(dev);

    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const std::size_t center =
        static_cast<std::size_t>(ny / 2) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(nx / 2);

    const float omega = 1.5f;
    const float u_lid = 0.1f;

    float *f = sycl::malloc_shared<float>(9 * n, q);
    float *f_new = sycl::malloc_shared<float>(9 * n, q);
    float *f_stream = sycl::malloc_shared<float>(9 * n, q);

    float *rho = sycl::malloc_shared<float>(n, q);
    float *ux = sycl::malloc_shared<float>(n, q);
    float *uy = sycl::malloc_shared<float>(n, q);

    if (!f || !f_new || !f_stream || !rho || !ux || !uy) {
        throw std::runtime_error("USM allocation failed");
    }

    for (std::size_t i = 0; i < n; ++i) {
        f[0 * n + i] = 4.0f / 9.0f;

        f[1 * n + i] = 1.0f / 9.0f;
        f[2 * n + i] = 1.0f / 9.0f;
        f[3 * n + i] = 1.0f / 9.0f;
        f[4 * n + i] = 1.0f / 9.0f;

        f[5 * n + i] = 1.0f / 36.0f;
        f[6 * n + i] = 1.0f / 36.0f;
        f[7 * n + i] = 1.0f / 36.0f;
        f[8 * n + i] = 1.0f / 36.0f;

        for (int k = 0; k < 9; ++k) {
            f_new[k * n + i] = 0.0f;
            f_stream[k * n + i] = 0.0f;
        }

        rho[i] = 0.0f;
        ux[i] = 0.0f;
        uy[i] = 0.0f;
    }

    // Small perturbation in the center cell.
    f[1 * n + center] += 0.05f;

    // Kernel 1: compute rho, ux, uy.
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        std::size_t i = idx[0];

        float f0 = f[0 * n + i];
        float f1 = f[1 * n + i];
        float f2 = f[2 * n + i];
        float f3 = f[3 * n + i];
        float f4 = f[4 * n + i];
        float f5 = f[5 * n + i];
        float f6 = f[6 * n + i];
        float f7 = f[7 * n + i];
        float f8 = f[8 * n + i];

        float local_rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

        rho[i] = local_rho;
        ux[i] = (f1 + f5 + f8 - f3 - f6 - f7) / local_rho;
        uy[i] = (f2 + f5 + f6 - f4 - f7 - f8) / local_rho;
    }).wait();

    // Kernel 2: BGK collision for all D2Q9 populations.
    q.parallel_for(sycl::range<1>(9 * n), [=](sycl::id<1> idx) {
        std::size_t gid = idx[0];

        int k = static_cast<int>(gid / n);
        std::size_t i = gid % n;

        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;

        if (k == 0) {
            cx = 0.0f;  cy = 0.0f;  w = 4.0f / 9.0f;
        } else if (k == 1) {
            cx = 1.0f;  cy = 0.0f;  w = 1.0f / 9.0f;
        } else if (k == 2) {
            cx = 0.0f;  cy = 1.0f;  w = 1.0f / 9.0f;
        } else if (k == 3) {
            cx = -1.0f; cy = 0.0f;  w = 1.0f / 9.0f;
        } else if (k == 4) {
            cx = 0.0f;  cy = -1.0f; w = 1.0f / 9.0f;
        } else if (k == 5) {
            cx = 1.0f;  cy = 1.0f;  w = 1.0f / 36.0f;
        } else if (k == 6) {
            cx = -1.0f; cy = 1.0f;  w = 1.0f / 36.0f;
        } else if (k == 7) {
            cx = -1.0f; cy = -1.0f; w = 1.0f / 36.0f;
        } else {
            cx = 1.0f;  cy = -1.0f; w = 1.0f / 36.0f;
        }

        float local_rho = rho[i];
        float local_ux = ux[i];
        float local_uy = uy[i];

        float cu = cx * local_ux + cy * local_uy;
        float u2 = local_ux * local_ux + local_uy * local_uy;

        float feq =
            w * local_rho *
            (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);

        f_new[k * n + i] =
            f[k * n + i] - omega * (f[k * n + i] - feq);
    }).wait();

    // Kernel 3: streaming with bounce-back walls.
    //
    // This is no longer periodic streaming.
    // If a population would leave the domain, it is reflected into the opposite direction.
    q.parallel_for(sycl::range<1>(9 * n), [=](sycl::id<1> idx) {
        std::size_t gid = idx[0];

        int k = static_cast<int>(gid / n);
        std::size_t i = gid % n;

        int x = static_cast<int>(i % static_cast<std::size_t>(nx));
        int y = static_cast<int>(i / static_cast<std::size_t>(nx));

        int dx = 0;
        int dy = 0;

        if (k == 0) {
            dx = 0;  dy = 0;
        } else if (k == 1) {
            dx = 1;  dy = 0;
        } else if (k == 2) {
            dx = 0;  dy = 1;
        } else if (k == 3) {
            dx = -1; dy = 0;
        } else if (k == 4) {
            dx = 0;  dy = -1;
        } else if (k == 5) {
            dx = 1;  dy = 1;
        } else if (k == 6) {
            dx = -1; dy = 1;
        } else if (k == 7) {
            dx = -1; dy = -1;
        } else {
            dx = 1;  dy = -1;
        }

        int x_to = x + dx;
        int y_to = y + dy;

        bool inside = (x_to >= 0 && x_to < nx && y_to >= 0 && y_to < ny);

        if (inside) {
            std::size_t dst =
                static_cast<std::size_t>(y_to) * static_cast<std::size_t>(nx)
                + static_cast<std::size_t>(x_to);

            f_stream[k * n + dst] = f_new[k * n + i];
        } else {
            int k_opp = 0;

            if (k == 1) k_opp = 3;
            else if (k == 2) k_opp = 4;
            else if (k == 3) k_opp = 1;
            else if (k == 4) k_opp = 2;
            else if (k == 5) k_opp = 7;
            else if (k == 6) k_opp = 8;
            else if (k == 7) k_opp = 5;
            else if (k == 8) k_opp = 6;

            f_stream[k_opp * n + i] = f_new[k * n + i];
        }
    }).wait();

    // Kernel 4: moving lid boundary at top wall.
    //
    // Top wall moves to the right with velocity u_lid.
    // This modifies the reflected populations at y = ny - 1.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(nx)), [=](sycl::id<1> idx) {
        int x = static_cast<int>(idx[0]);
        int y = ny - 1;

        std::size_t i =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
            + static_cast<std::size_t>(x);

        float local_rho = rho[i];

        // Unknown downward populations at the top wall:
        // f4, f7, f8.
        //
        // Simple moving-wall correction for horizontal lid velocity.
        f_stream[4 * n + i] = f_new[2 * n + i];
        f_stream[7 * n + i] = f_new[5 * n + i] - (1.0f / 6.0f) * local_rho * u_lid;
        f_stream[8 * n + i] = f_new[6 * n + i] + (1.0f / 6.0f) * local_rho * u_lid;
    }).wait();

    float rho_min = rho[0];
    float rho_max = rho[0];
    float max_abs_ux = std::fabs(ux[0]);
    float max_abs_uy = std::fabs(uy[0]);

    for (std::size_t i = 1; i < n; ++i) {
        rho_min = std::min(rho_min, rho[i]);
        rho_max = std::max(rho_max, rho[i]);
        max_abs_ux = std::max(max_abs_ux, std::fabs(ux[i]));
        max_abs_uy = std::max(max_abs_uy, std::fabs(uy[i]));
    }

    // Check collision for center f1.
    float center_rho = rho[center];
    float center_ux = ux[center];
    float center_uy = uy[center];

    float cx1 = 1.0f;
    float cy1 = 0.0f;
    float w1 = 1.0f / 9.0f;

    float cu1 = cx1 * center_ux + cy1 * center_uy;
    float u2_center = center_ux * center_ux + center_uy * center_uy;

    float feq1_center =
        w1 * center_rho *
        (1.0f + 3.0f * cu1 + 4.5f * cu1 * cu1 - 1.5f * u2_center);

    float expected_f1_center_after_collision =
        f[1 * n + center] - omega * (f[1 * n + center] - feq1_center);

    float actual_f1_center_after_collision = f_new[1 * n + center];

    bool collision_check_ok =
        std::fabs(actual_f1_center_after_collision - expected_f1_center_after_collision) < 1e-5f;

    // Check bounce-back at left wall.
    // Population f3 from cell (0, center_y) wants to go left,
    // so it should bounce back into f1 at the same cell.
    int left_x = 0;
    int left_y = ny / 2;

    std::size_t left_cell =
        static_cast<std::size_t>(left_y) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(left_x);

    float expected_left_bounce = f_new[3 * n + left_cell];
    float actual_left_bounce = f_stream[1 * n + left_cell];

    bool bounce_back_check_ok =
        std::fabs(actual_left_bounce - expected_left_bounce) < 1e-5f;

    // Check moving lid at top center.
    int top_x = nx / 2;
    int top_y = ny - 1;

    std::size_t top_cell =
        static_cast<std::size_t>(top_y) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(top_x);

    float expected_top_f8 =
        f_new[6 * n + top_cell] + (1.0f / 6.0f) * rho[top_cell] * u_lid;

    float actual_top_f8 = f_stream[8 * n + top_cell];

    bool moving_lid_check_ok =
        std::fabs(actual_top_f8 - expected_top_f8) < 1e-5f;

    bool check_ok =
        collision_check_ok &&
        bounce_back_check_ok &&
        moving_lid_check_ok;

    py::dict result;

    result["nx"] = nx;
    result["ny"] = ny;

    result["rho_min"] = rho_min;
    result["rho_max"] = rho_max;
    result["max_abs_ux"] = max_abs_ux;
    result["max_abs_uy"] = max_abs_uy;

    result["center_rho"] = center_rho;
    result["center_ux"] = center_ux;
    result["center_uy"] = center_uy;

    result["center_f1_before"] = f[1 * n + center];
    result["center_f1_after_collision"] = actual_f1_center_after_collision;
    result["center_f1_expected_collision"] = expected_f1_center_after_collision;

    result["left_wall_expected_bounce"] = expected_left_bounce;
    result["left_wall_actual_bounce"] = actual_left_bounce;

    result["top_lid_expected_f8"] = expected_top_f8;
    result["top_lid_actual_f8"] = actual_top_f8;

    result["collision_all_populations"] = true;
    result["streaming_all_populations"] = true;
    result["periodic_boundaries"] = false;
    result["bounce_back_walls"] = true;
    result["moving_lid"] = true;
    result["u_lid"] = u_lid;

    result["collision_check_ok"] = collision_check_ok;
    result["bounce_back_check_ok"] = bounce_back_check_ok;
    result["moving_lid_check_ok"] = moving_lid_check_ok;
    result["check_ok"] = check_ok;

    sycl::free(f, q);
    sycl::free(f_new, q);
    sycl::free(f_stream, q);
    sycl::free(rho, q);
    sycl::free(ux, q);
    sycl::free(uy, q);

    return result;
}

PYBIND11_MODULE(_minilb_step, m) {
    m.doc() = "miniLB one-step pybind11 + dpctl + C++ SYCL kernel module";

    m.def("get_device_name", &get_device_name,
          "Return the name of a dpctl/SYCL device");

    m.def("run_minilb_step", &run_minilb_step,
          "Run a simplified Lid-Driven Cavity LBM step with density, collision, streaming and boundary conditions");
}

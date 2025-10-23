use burn::tensor::{backend::Backend, Bool, Tensor};

/// Converts a depth image to a point cloud in camera coordinates.
///
/// # Arguments
///
/// * `depth` - (H, W) Depth image.
/// * `fx` - Focal length x.
/// * `fy` - Focal length y.
/// * `cx` - Optional principal point x. Defaults to width / 2.0.
/// * `cy` - Optional principal point y. Defaults to height / 2.0.
///
/// # Returns
///
/// (H, W, 3) Point cloud in camera coordinates.
pub fn depth_to_point_cloud<B: Backend>(
    depth: Tensor<B, 2>,
    fx: f32,
    fy: f32,
    cx: Option<f32>,
    cy: Option<f32>,
) -> Tensor<B, 3> {
    let device = depth.device();
    let [height, width] = depth.dims();

    let cx = cx.unwrap_or(width as f32 / 2.0);
    let cy = cy.unwrap_or(height as f32 / 2.0);

    let x_coords = (Tensor::arange(0..width as i64, &device).float() + 0.5).reshape([1, width]);
    let y_coords = (Tensor::arange(0..height as i64, &device).float() + 0.5).reshape([height, 1]);

    let x = (x_coords - cx) / fx;
    let y = (y_coords - cy) / fy;

    let cam_x = x * depth.clone();
    let cam_y = y * depth.clone();
    let cam_z = depth;

    Tensor::stack(vec![cam_x, cam_y, cam_z], 2)
}

/// Computes normals from a grid-structured point cloud (H, W, 3).
///
/// # Arguments
///
/// * `xyz` - (H, W, 3) Point cloud.
///
/// # Returns
///
/// (H, W, 3) Normal map.
pub fn normal_from_grid_point_cloud<B: Backend>(xyz: Tensor<B, 3>) -> Tensor<B, 3> {
    let [height, width, _] = xyz.dims();

    let top = xyz.clone().slice([0..height - 2, 1..width - 1, 0..3]);
    let bottom = xyz.clone().slice([2..height, 1..width - 1, 0..3]);
    let left = xyz.clone().slice([1..height - 1, 0..width - 2, 0..3]);
    let right = xyz.clone().slice([1..height - 1, 2..width, 0..3]);

    let left_to_right = right - left;
    let bottom_to_top = top - bottom;

    let [h_prime, w_prime, _] = left_to_right.dims();
    let ltr_x = left_to_right.clone().slice([0..h_prime, 0..w_prime, 0..1]);
    let ltr_y = left_to_right.clone().slice([0..h_prime, 0..w_prime, 1..2]);
    let ltr_z = left_to_right.slice([0..h_prime, 0..w_prime, 2..3]);

    let btt_x = bottom_to_top.clone().slice([0..h_prime, 0..w_prime, 0..1]);
    let btt_y = bottom_to_top.clone().slice([0..h_prime, 0..w_prime, 1..2]);
    let btt_z = bottom_to_top.slice([0..h_prime, 0..w_prime, 2..3]);

    let normal_x = ltr_y.clone() * btt_z.clone() - ltr_z.clone() * btt_y.clone();
    let normal_y = ltr_z * btt_x.clone() - ltr_x.clone() * btt_z;
    let normal_z = ltr_x * btt_y - ltr_y * btt_x;
    let normal = Tensor::cat(vec![normal_x, normal_y, normal_z], 2);

    let norm_sq = normal.clone().powi_scalar(2).sum_dim(2).clamp_min(1e-20);
    let normal = normal / norm_sq.sqrt().unsqueeze();

    let padded_normal = normal
        .permute([2, 0, 1])
        .pad((1, 1, 1, 1), 0.0)
        .permute([1, 2, 0]);
    padded_normal
}

/// Renders a normal image from a depth image.
///
/// # Arguments
///
/// * `depth` - (H, W) The rendered depth image.
/// * `fx` - Focal length x.
/// * `fy` - Focal length y.
/// * `cx` - Optional principal point x. Defaults to width / 2.0.
/// * `cy` - Optional principal point y. Defaults to height / 2.0.
///
/// # Returns
///
/// (H, W, 3) The rendered normal image.
pub fn compute_normal_from_depth<B: Backend>(
    depth: Tensor<B, 2>,
    fx: f32,
    fy: f32,
    cx: Option<f32>,
    cy: Option<f32>,
) -> Tensor<B, 3> {
    let xyz_cam = depth_to_point_cloud(depth, fx, fy, cx, cy);
    normal_from_grid_point_cloud(xyz_cam)
}

/// Computes image gradient weights.
///
/// # Arguments
///
/// * `img` - Input image tensor of shape (H, W, C).
///
/// # Returns
///
/// A tensor of shape (H, W) representing the image gradient weights.
pub fn get_img_grad_weight<B: Backend>(img: Tensor<B, 3>) -> Tensor<B, 2> {
    let [height, width, channels] = img.dims();

    let bottom_point = img.clone().slice([2..height, 1..width - 1, 0..channels]);
    let top_point = img
        .clone()
        .slice([0..height - 2, 1..width - 1, 0..channels]);
    let right_point = img.clone().slice([1..height - 1, 2..width, 0..channels]);
    let left_point = img
        .clone()
        .slice([1..height - 1, 0..width - 2, 0..channels]);

    let grad_img_x = (right_point - left_point).abs().sum_dim(2) / channels as f32;
    let grad_img_y = (top_point - bottom_point).abs().sum_dim(2) / channels as f32;

    let grad_img = grad_img_x.max_pair(grad_img_y);

    let min_val = grad_img.clone().min();
    let max_val = grad_img.clone().max();

    let range = (max_val - min_val.clone()).clamp_min(1e-6);
    let grad_img = (grad_img - min_val.unsqueeze()) / range.unsqueeze();

    grad_img.squeeze().pad((1, 1, 1, 1), 1.0)
}

/// Transforms points by a 4x4 transformation matrix.
///
/// # Arguments
///
/// * `xyz` - Tensor of points, shape (N, 3).
/// * `trafo` - Transformation matrix, shape (4, 4).
///
/// # Returns
///
/// Tensor of transformed points, shape (N, 3).
pub fn transform_points<B: Backend>(xyz: Tensor<B, 2>, trafo: Tensor<B, 2>) -> Tensor<B, 2> {
    let [n, d] = xyz.dims();
    assert!(d == 3, "Input xyz must have shape (N, 3), got {n}x{d}");

    let device = xyz.device();

    // Convert to homogeneous coordinates
    let ones = Tensor::ones([n, 1], &device);
    let xyz_homogeneous = Tensor::cat(vec![xyz, ones], 1);

    // Transform points: (trafo @ xyz_homogeneous.T).T
    // which is equivalent to xyz_homogeneous @ trafo.T
    let xyz_trafo_homogeneous = xyz_homogeneous.matmul(trafo.transpose());

    // Convert back from homogeneous coordinates by dividing xyz by w.
    let xyz_trafo = xyz_trafo_homogeneous.clone().slice([0..n, 0..3]);
    let w = xyz_trafo_homogeneous.slice([0..n, 3..4]).clamp_min(1e-8);

    xyz_trafo / w
}

/// Projects a point cloud in camera space to image coordinates.
///
/// # Arguments
///
/// * `xyz_cam` - (N, 3) Point cloud in camera coordinates.
/// * `fx` - Focal length x.
/// * `fy` - Focal length y.
/// * `cx` - Principal point x.
/// * `cy` - Principal point y.
///
/// # Returns
///
/// (N, 2) Image coordinates (u, v).
pub fn project_point_cloud_to_image<B: Backend>(
    xyz_cam: Tensor<B, 2>,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Tensor<B, 2> {
    let [n, d] = xyz_cam.dims();
    assert!(d == 3, "Input xyz_cam must have shape (N, 3), got {n}x{d}");

    let x = xyz_cam.clone().slice([0..n, 0..1]);
    let y = xyz_cam.clone().slice([0..n, 1..2]);
    let z = xyz_cam.slice([0..n, 2..3]).clamp_min(1e-8);

    let u = x * fx / z.clone() + cx;
    let v = y * fy / z + cy;

    Tensor::cat(vec![u, v], 1)
}

/// Computes a mask for valid projections.
///
/// Checks if pixel coordinates are within image bounds and if points are in front
/// of the camera.
///
/// # Arguments
///
/// * `pixel_coords` - Tensor of shape (N, 2) containing (u, v) pixel coordinates.
/// * `z_in_camera` - Tensor of shape (N,) containing the Z coordinate of the points
///   in camera space.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `min_depth` - Minimum depth value to be considered in front of the camera.
///
/// # Returns
///
/// A boolean tensor of shape (N,) where True indicates the projection is valid.
pub fn compute_projection_mask<B: Backend>(
    pixel_coords: Tensor<B, 2>,
    z_in_camera: Tensor<B, 1>,
    width: u32,
    height: u32,
    min_depth: f32,
) -> Tensor<B, 1, Bool> {
    let [n, _] = pixel_coords.dims();
    let u = pixel_coords.clone().slice([0..n, 0..1]).squeeze();
    let v = pixel_coords.slice([0..n, 1..2]).squeeze();

    let mask_u = u
        .clone()
        .greater_equal_elem(0.0)
        .bool_and(u.lower_elem(width as f32));
    let mask_v = v
        .clone()
        .greater_equal_elem(0.0)
        .bool_and(v.lower_elem(height as f32));
    let mask_z = z_in_camera.greater_elem(min_depth);

    mask_u.bool_and(mask_v).bool_and(mask_z)
}

/// Samples depth values from a depth map at the given pixel coordinates.
///
/// Uses bilinear interpolation for subpixel accuracy. Coordinates outside
/// the depth map bounds are padded using the border values.
///
/// # Arguments
///
/// * `pixel_coords` - Tensor of shape (N, 2) containing (u, v) pixel coordinates.
/// * `depth_map` - Tensor of shape (H, W) representing the depth map.
///
/// # Returns
///
/// A tensor of shape (N,) containing the sampled depth values.
pub fn sample_depth_at_coordinates<B: Backend>(
    pixel_coords: Tensor<B, 2>,
    depth_map: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [height, width] = depth_map.dims();
    let [n, _] = pixel_coords.dims();

    let u: Tensor<B, 1> = pixel_coords.clone().slice([0..n, 0..1]).squeeze();
    let v: Tensor<B, 1> = pixel_coords.slice([0..n, 1..2]).squeeze();

    // Manual bilinear interpolation
    // Clamp raw coordinates to valid sampling range [0, width-1] and [0, height-1]
    // This effectively implements border padding for out-of-bounds samples.
    let u_clamped = u.clamp(0.0, (width - 1) as f32);
    let v_clamped = v.clamp(0.0, (height - 1) as f32);

    let u_floor = u_clamped.clone().floor();
    let v_floor = v_clamped.clone().floor();

    let u_ceil = (u_floor.clone() + 1.0).clamp(0.0, (width - 1) as f32);
    let v_ceil = (v_floor.clone() + 1.0).clamp(0.0, (height - 1) as f32);

    // Convert to integer indices
    let u1_int = u_floor.clone().int();
    let v1_int = v_floor.clone().int();
    let u2_int = u_ceil.clone().int();
    let v2_int = v_ceil.clone().int();

    // Calculate 1D indices for gathering from flattened depth_map
    let width_tensor = Tensor::full([1], width as i32, &depth_map.device());

    let idx_q11 = v1_int.clone() * width_tensor.clone() + u1_int.clone();
    let idx_q21 = v1_int.clone() * width_tensor.clone() + u2_int.clone();
    let idx_q12 = v2_int.clone() * width_tensor.clone() + u1_int.clone();
    let idx_q22 = v2_int.clone() * width_tensor.clone() + u2_int.clone();

    // Flatten the depth map for 1D indexing
    let depth_map_flat = depth_map.reshape([height * width]);

    // Gather values from the four corners
    let q11: Tensor<B, 1> = depth_map_flat.clone().gather(0, idx_q11);
    let q21: Tensor<B, 1> = depth_map_flat.clone().gather(0, idx_q21);
    let q12: Tensor<B, 1> = depth_map_flat.clone().gather(0, idx_q12);
    let q22: Tensor<B, 1> = depth_map_flat.clone().gather(0, idx_q22);

    // Calculate interpolation weights
    let u_weight = u_clamped - u_floor;
    let v_weight = v_clamped - v_floor;

    let u_inv_weight: Tensor<B, 1> = 1.0 - u_weight.clone();
    let v_inv_weight: Tensor<B, 1> = 1.0 - v_weight.clone();

    // Interpolate along u-axis
    let r1 = q11.mul(u_inv_weight.clone()) + q21.mul(u_weight.clone());
    let r2 = q12.mul(u_inv_weight) + q22.mul(u_weight);

    // Interpolate along v-axis
    let sampled_depth = r1 * v_inv_weight + r2 * v_weight;

    sampled_depth
}

/// Computes the pixel errors between reprojected points and a grid.
///
/// # Arguments
///
/// * `p_r` - Tensor of shape (N, 2) containing the reprojected pixel coordinates (u, v).
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Returns
///
/// A tensor of shape (N,) containing the Euclidean distance between each
/// reprojected point and the corresponding grid pixel.
pub fn compute_pixel_errors<B: Backend>(
    p_r: Tensor<B, 2>,
    width: u32,
    height: u32,
) -> Tensor<B, 1> {
    let device = p_r.device();
    let [n, _] = p_r.dims();
    assert_eq!(
        n,
        (width * height) as usize,
        "Input p_r must have shape (width * height, 2)"
    );

    let ix: Tensor<B, 3> = (Tensor::arange(0..width as i64, &device).float() + 0.5)
        .reshape([1, width as usize])
        .repeat_dim(0, height as usize)
        .reshape([height as usize, width as usize, 1]);

    let iy: Tensor<B, 3> = (Tensor::arange(0..height as i64, &device).float() + 0.5)
        .reshape([height as usize, 1])
        .repeat_dim(1, width as usize)
        .reshape([height as usize, width as usize, 1]);

    let p = Tensor::cat(vec![ix, iy], 2).reshape([n, 2]);
    (p_r - p).powi_scalar(2).sum_dim(1).sqrt().squeeze()
}



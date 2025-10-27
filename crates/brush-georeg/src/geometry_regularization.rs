use crate::{geometry_utils};
use brush_render::camera::Camera;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};

use glam::Mat4;
pub fn scale_loss<B: AutodiffBackend>(
    scales: &Tensor<B, 2>,
    visible: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let num_visible = visible.clone().sum();

    let min_scales = scales.clone().min_dim(1);
    let visible_min_scales = min_scales * visible.clone().unsqueeze_dim(1);
    let scale_loss = visible_min_scales.sum() / num_visible.clamp_min(1.0);

    scale_loss
}

pub fn sv_geometry_regularization_loss<B: AutodiffBackend>(
    rendered_image: &Tensor<B, 3>,
    cam: &Camera,
    img_size: glam::UVec2,
    gt_image: &Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [h, w, c] = rendered_image.dims();
    assert_eq!(
        c, 10,
        "rendered_image must have 9 channels (RGBA, Normal, Depth, Distance)"
    );

    let rendered_normal = rendered_image.clone().slice([0..h, 0..w, 4..7]);
    let rendered_depth: Tensor<B, 2> = rendered_image.clone().slice([0..h, 0..w, 8..9]).squeeze();
    let rendered_alpha: Tensor<B, 3> = rendered_image.clone().slice([0..h, 0..w, 3..4]);

    let focal = cam.focal(img_size);
    let center = cam.center(img_size);

    let mut depth_normal = geometry_utils::compute_normal_from_depth(
        rendered_depth.clone(),
        focal.x,
        focal.y,
        Some(center.x),
        Some(center.y),
        true, // use_pixel_centers
    );

    depth_normal = depth_normal * rendered_alpha.detach();

    let img_weight: Tensor<B, 2> = geometry_utils::get_img_grad_weight(gt_image.clone())
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(0.0, 1.0)
        .detach()
        .powf_scalar(2.0);

    let normal_error = (depth_normal.clone() - rendered_normal.clone())
        .abs()
        .sum_dim(2);

    let normal_error = normal_error.squeeze();
    let sv_loss = img_weight.clone() * normal_error;

    sv_loss.mean()
}

pub fn mv_geometry_regularization_loss<B: AutodiffBackend>(
    rendered_image: &Tensor<B, 3>,
    rendered_image_n: &Tensor<B, 3>,
    cam: &Camera,
    cam_n: &Camera,
    img_size: glam::UVec2,
    img_size_n: glam::UVec2,
    use_pixel_centers: bool,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let [h, w, c] = rendered_image.dims();
    let [h_n, w_n, c_n] = rendered_image_n.dims();
    assert!(
        c == 10 && c_n == 10,
        "rendered images must have 10 channels (RGBA, Normal, Depth, Distance)"
    );

    let rendered_depth: Tensor<B, 2> = rendered_image.clone().slice([0..h, 0..w, 8..9]).squeeze();
    let rendered_depth_n: Tensor<B, 2> = rendered_image_n
        .clone()
        .slice([0..h_n, 0..w_n, 8..9])
        .squeeze();

    let focal = cam.focal(img_size);
    let center = cam.center(img_size);
    let focal_n = cam_n.focal(img_size_n);
    let center_n = cam_n.center(img_size_n);

    // MV Geometric Loss
    let pcd_r = geometry_utils::depth_to_point_cloud(
        rendered_depth.clone(),
        focal.x,
        focal.y,
        Some(center.x),
        Some(center.y),
        use_pixel_centers,
    );
    let pcd_r = pcd_r.reshape([h * w, 3]);

    let world_view_transform_r = Mat4::from(cam.world_to_local());
    let world_view_transform_n = Mat4::from(cam_n.world_to_local());

    let r_to_n_transform_mat =
        (world_view_transform_n * world_view_transform_r.inverse()).transpose();
    let r_to_n_transform = Tensor::<B, 1>::from_data(
        TensorData::new(r_to_n_transform_mat.to_cols_array().to_vec(), [16]),
        &rendered_image.device(),
    )
    .reshape([4, 4]);
    let n_to_r_transform_mat =
        (world_view_transform_r * world_view_transform_n.inverse()).transpose();
    let n_to_r_transform = Tensor::<B, 1>::from_data(
        TensorData::new(n_to_r_transform_mat.to_cols_array().to_vec(), [16]),
        &rendered_image.device(),
    )
    .reshape([4, 4]);

    let pcd_r_n = geometry_utils::transform_points(pcd_r.clone(), r_to_n_transform.clone());

    let p_n = geometry_utils::project_point_cloud_to_image(
        pcd_r_n.clone(),
        focal_n.x,
        focal_n.y,
        center_n.x,
        center_n.y,
    );

    let mut depth_mask = geometry_utils::compute_projection_mask(
        p_n.clone(),
        pcd_r_n.clone().slice([0..h * w, 2..3]).squeeze(),
        w_n as u32,
        h_n as u32,
        0.1,
    );
    let p_n_depth =
        geometry_utils::sample_depth_at_coordinates(p_n.clone(), rendered_depth_n.clone());

    let rays_n = pcd_r_n.clone() / (pcd_r_n.clone().slice([0..h * w, 2..3]).clamp_min(1e-9));
    let pcd_n_cam_new = rays_n * p_n_depth.unsqueeze_dim(1);

    let pcd_n_r = geometry_utils::transform_points(pcd_n_cam_new.clone(), n_to_r_transform);

    let p_r_reprojected = geometry_utils::project_point_cloud_to_image(
        pcd_n_r.clone(),
        focal.x,
        focal.y,
        center.x,
        center.y,
    );

    let pixel_error = geometry_utils::compute_pixel_errors(
        p_r_reprojected,
        w as u32,
        h as u32,
        use_pixel_centers,
    );

    let pixel_error_threshold = 1.0;
    depth_mask = depth_mask.bool_and(pixel_error.clone().lower_elem(pixel_error_threshold));

    let mut weights = (pixel_error.clone() * -1.0).exp().detach();
    weights = weights
        .clone()
        .mask_where(depth_mask.clone().bool_not(), Tensor::zeros_like(&weights));
    
    let weighted_error = weights.clone() * pixel_error;
    let masked_error = weighted_error.mask_fill(depth_mask.clone().bool_not(), 0.0);
    let num_valid = depth_mask.clone().float().sum();
    let mv_geometric_loss = masked_error.sum() / num_valid.clamp_min(1.0);

    // MV Photometric Loss
    let mv_photometric_loss = Tensor::zeros([1], &rendered_image.device());

    (mv_geometric_loss, mv_photometric_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_utils;
    use brush_dataset::scene::sample_to_tensor_data;
    use brush_render::MainBackend;
    use burn::backend::{wgpu::WgpuDevice, Autodiff};
    use npyz::npz::NpzArchive;

    type DiffBackend = Autodiff<MainBackend>;

    #[allow(clippy::bool_assert_comparison)]
    fn compare_tensors<B: AutodiffBackend, const D: usize>(
        a: Tensor<B::InnerBackend, D>,
        b: Tensor<B::InnerBackend, D>,
        rtol: f32,
    ) {
        let num_el: usize = a.dims().iter().product();
        let a_flat = a.reshape([num_el]);
        let b_flat = b.reshape([num_el]);
        let a_data = a_flat.into_data().into_vec::<f32>().unwrap();
        let b_data = b_flat.into_data().into_vec::<f32>().unwrap();

        assert_eq!(
            a_data.len(),
            b_data.len(),
            "Tensor element counts do not match."
        );

        for (i, (val_a, val_b)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let tolerance = val_b.abs() * rtol;
            let diff = (val_a - val_b).abs();
            assert!(
                diff <= tolerance,
                "Tensors are not close at index {}. Got: {}, Expected: {} (diff: {}, tol: {})",
                i,
                val_a,
                val_b,
                diff,
                tolerance
            );
        }
    }

    fn load_rendered_image_data(
        data_path: &str,
    ) -> Result<
        (
            Tensor<DiffBackend, 3>, // rendered_image
            Tensor<DiffBackend, 3>, // out_rgb
            Tensor<DiffBackend, 3>, // out_alpha
            Tensor<DiffBackend, 3>, // out_normal
            Tensor<DiffBackend, 3>, // out_depth
            Tensor<DiffBackend, 3>, // out_distance
        ),
        anyhow::Error,
    > {
        let mut archive = NpzArchive::new(std::fs::File::open(data_path)?)?;
        let out_rgb: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "out_rgb")?
                .permute([1, 2, 0])
                .require_grad();
        let out_alpha: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "out_alpha")?
                .permute([1, 2, 0])
                .require_grad();
        let out_normal: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "out_normal")?
                .permute([1, 2, 0])
                .require_grad();
        let out_depth: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "out_depth")?
                .permute([1, 2, 0])
                .require_grad();
        let out_distance: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "out_distance")?
                .permute([1, 2, 0])
                .require_grad();

        let rendered_image = Tensor::cat(
            vec![
                out_rgb.clone(),
                out_alpha.clone(),
                out_normal.clone(),
                out_alpha.clone(),
                out_depth.clone(),
                out_distance.clone(),
            ],
            2,
        );
        Ok((
            rendered_image,
            out_rgb,
            out_alpha,
            out_normal,
            out_depth,
            out_distance,
        ))
    }

    #[test]
    fn test_scale_loss() -> Result<(), anyhow::Error> {
        // Set Up
        let scales_visibility_path = "test_data/scales_visibility_data.npz";
        let mut archive = NpzArchive::new(std::fs::File::open(scales_visibility_path)?)?;
        let log_scales: Tensor<DiffBackend, 2> =
            io_utils::load_tensor_from_npz(&mut archive, "log_scale")?.require_grad();
        let visibility_filter: Tensor<DiffBackend, 1> =
            io_utils::load_tensor_from_npz(&mut archive, "visibility_filter")?;

        let dl_d_scales_grads_data_path = "test_data/dL_d_scales_grads_data.npz";
        let mut archive = NpzArchive::new(std::fs::File::open(dl_d_scales_grads_data_path)?)?;
        let dl_d_scales: Tensor<DiffBackend, 2> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_scales")?;

        // Execute
        let scale_loss = scale_loss(&log_scales.clone().exp(), &visibility_filter);
        let grads = scale_loss.backward();

        // Assert
        let loss_val = scale_loss.into_scalar();
        assert!(
            (loss_val - 6.5665e-06).abs() < 1e-4,
            "scale_loss is not close to 6.5665e-06. Got: {}",
            loss_val
        );

        let scale_grads = log_scales.grad(&grads).expect("scale_grads should exist");
        compare_tensors::<DiffBackend, 2>(scale_grads, dl_d_scales.inner(), 1e-4);

        Ok(())
    }

    #[test]
    fn test_sv_geometry_regularization_loss() -> Result<(), anyhow::Error> {
        // Set Up
        let device = WgpuDevice::DefaultDevice;
        let cam = Camera::new(
            glam::vec3(0.5105, 0.2538, 0.0553),
            glam::Quat::from_xyzw(0.92868414, 0.02292308, -0.36890709, -0.03046073),
            1.1927426335535698,
            0.7294988095857634,
            glam::vec2(0.5, 0.5),
        );
        let img_size = glam::uvec2(640, 360);

        let gt_image_path = String::from("test_data/MotionTrackingPrimary_00315.png");
        let gt_image_dyn = image::open(&gt_image_path)?.resize_to_fill(
            img_size.x,
            img_size.y,
            image::imageops::FilterType::Triangle,
        );
        let gt_image: Tensor<DiffBackend, 3> =
            Tensor::from_data(sample_to_tensor_data(gt_image_dyn), &device);

        let (rendered_image, out_rgb, out_alpha, out_normal, out_depth, out_distance) =
            load_rendered_image_data("test_data/img_out_data.npz")?;

        let dl_dl_sv_d_depth_normal_grads_path = "test_data/dL_sv_d_depth_normal_grads_data.npz";
        let mut archive =
            NpzArchive::new(std::fs::File::open(dl_dl_sv_d_depth_normal_grads_path)?)?;
        let dl_d_depth: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_depth")?.permute([1, 2, 0]);
        let dl_d_normal: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_normal")?.permute([1, 2, 0]);

        // Execute
        let sv_loss = sv_geometry_regularization_loss(&rendered_image, &cam, img_size, &gt_image);
        let grads = sv_loss.backward();

        // Assert
        // Loss value
        let loss_val = sv_loss.into_scalar();
        assert!(
            (loss_val - 0.6308).abs() < 1e-2,
            "sv_loss is not close to 0.6308. Got: {}",
            loss_val
        );

        // alpha_grads and rgb_grads and distance_grads are 0
        assert!(
            out_alpha.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "alpha_grads should be 0"
        );
        assert!(
            out_distance.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "distance_grads should be 0"
        );
        assert!(
            out_rgb.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "rgb_grads should be 0"
        );

        let depth_grads = out_depth.grad(&grads).expect("depth_grads should exist");
        compare_tensors::<DiffBackend, 3>(
            depth_grads.slice([160..166, 320..321, 0..1]),
            dl_d_depth.inner().slice([160..166, 320..321, 0..1]),
            1e-0,
        );

        let normal_grads = out_normal.grad(&grads).expect("normal_grads should exist");
        compare_tensors::<DiffBackend, 3>(
            normal_grads.slice([160..166, 320..321, 0..3]),
            dl_d_normal.inner().slice([160..166, 320..321, 0..3]),
            1e-1,
        );

        Ok(())
    }

    #[test]
    fn test_mv_geometry_regularization_loss() -> Result<(), anyhow::Error> {
        // Set Up
        let cam_r = Camera::new(
            glam::vec3(0.5105, 0.2538, 0.0553),
            glam::Quat::from_xyzw(0.92868413, 0.02292308, -0.36890713, -0.03046074),
            1.1927426335535698,
            0.7294988095857634,
            glam::vec2(0.5, 0.5),
        );
        let img_size_r = glam::uvec2(640, 360);

        let cam_n = Camera::new(
            glam::vec3(0.4006, 0.2903, 0.1195),
            glam::Quat::from_xyzw(0.95607085, 0.05261899, -0.28784764, -0.01742129),
            1.1927426335535698,
            0.7294988095857634,
            glam::vec2(0.5, 0.5),
        );
        let img_size_n = glam::uvec2(640, 360);

        let (rendered_image_r, out_rgb_r, out_alpha_r, out_normal_r, out_depth_r, out_distance_r) =
            load_rendered_image_data("test_data/img_out_r_data.npz")?;
        let (rendered_image_n, out_rgb_n, out_alpha_n, out_normal_n, out_depth_n, out_distance_n) =
            load_rendered_image_data("test_data/img_out_n_data.npz")?;

        let dl_mv_geo_d_depth_r_grads_path = "test_data/dL_mv_geo_d_depth_r_grads.npz";
        let mut archive =
            NpzArchive::new(std::fs::File::open(dl_mv_geo_d_depth_r_grads_path)?)?;
        let dl_d_depth_r: Tensor<DiffBackend, 2> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_depth")?;

        let dl_mv_geo_d_depth_n_grads_path = "test_data/dL_mv_geo_d_depth_n_grads.npz";
        let mut archive =
            NpzArchive::new(std::fs::File::open(dl_mv_geo_d_depth_n_grads_path)?)?;
        let dl_d_depth_n: Tensor<DiffBackend, 2> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_depth")?;

        // Execute
        let mv_geom_loss = mv_geometry_regularization_loss(
            &rendered_image_r,
            &rendered_image_n,
            &cam_r,
            &cam_n,
            img_size_r,
            img_size_n,
            false,
        );
        let mv_geom_geom_loss = mv_geom_loss.0;
        let grads = mv_geom_geom_loss.backward();

        // Assert
        // loss value
        let loss_val = mv_geom_geom_loss.into_scalar();
        assert!(
            (loss_val - 0.1814).abs() < 1e-2,
            "mv_loss is not close to 0.1814. Got: {}",
            loss_val
        );

        // alpha_grads and rgb_grads and distance_grads are 0
        assert!(
            out_alpha_r.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "alpha_grads should be 0"
        );
        assert!(
            out_normal_r.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "normal_grads should be 0"
        );
        assert!(
            out_distance_r.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "distance_grads should be 0"
        );
        assert!(
            out_rgb_r.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "rgb_grads should be 0"
        );

        assert!(
            out_alpha_n.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "alpha_grads should be 0"
        );
        assert!(
            out_normal_n.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "normal_grads should be 0"
        );
        assert!(
            out_distance_n.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "distance_grads should be 0"
        );
        assert!(
            out_rgb_n.grad(&grads).unwrap().sum().into_scalar() < 1e-9,
            "rgb_grads should be 0"
        );

        let depth_r_grads = out_depth_r.grad(&grads).expect("depth_grads should exist");
        compare_tensors::<DiffBackend, 2>(
            depth_r_grads.slice([160..166, 320..321, 0..1]).squeeze_dim(2),
            dl_d_depth_r.inner().slice([160..166, 320..321]),
            1e-1,
        );

        let depth_n_grads = out_depth_n.grad(&grads).expect("depth_grads should exist");
        compare_tensors::<DiffBackend, 2>(
            depth_n_grads.slice([170..180, 320..321, 0..1]).squeeze_dim(2),
            dl_d_depth_n.inner().slice([170..180, 320..321]),
            1e-0,
        );

        Ok(())
    }
}

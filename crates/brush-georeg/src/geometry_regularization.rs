use brush_render::{camera::Camera, render_aux::RenderAux};
use burn::tensor::{
    backend::AutodiffBackend, ElementConversion, Tensor,
    TensorPrimitive,
};
use crate::geometry_utils::{
    compute_normal_from_depth, compute_pixel_errors, compute_projection_mask, depth_to_point_cloud,
    get_img_grad_weight, project_point_cloud_to_image, sample_depth_at_coordinates,
};
use crate::io_utils;

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

    let mut depth_normal = compute_normal_from_depth(
        rendered_depth.clone(),
        focal.x,
        focal.y,
        Some(center.x),
        Some(center.y),
    );

    // println!(
    //     "Depth Normal : {}",
    //     depth_normal.clone().slice([160..161, 320..321, 0..3])
    // );

    depth_normal = depth_normal * rendered_alpha.detach();

    let img_weight: Tensor<B, 2> = get_img_grad_weight(gt_image.clone())
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(0.0, 1.0)
        .detach()
        .powf_scalar(2.0);

    // println!(
    //     "Img Weight : {}",
    //     img_weight.clone().slice([160..171, 320..321])
    // );

    let normal_error = (depth_normal.clone() - rendered_normal.clone())
        .abs()
        .sum_dim(2);

    let normal_error = normal_error.squeeze();
    let sv_loss = img_weight.clone() * normal_error;

    // println!(
    //     "Normal loss: {}",
    //     sv_loss.clone().slice([160..161, 320..321])
    // );

    sv_loss.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_utils;
    use brush_dataset::scene::sample_to_tensor_data;
    use brush_render::MainBackend;
    use burn::backend::{wgpu::WgpuDevice, Autodiff};
    use npyz::npz::NpzArchive;

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

    type DiffBackend = Autodiff<MainBackend>;

    // #[test]
    // fn test_local_distance() ->  -> Result<(), anyhow::Error> {
    //     // Set Up
   
    // }
    #[test]
    fn test_scale_loss() -> Result<(), anyhow::Error> {
        // Set Up
        let scales_visibility_path = "test_data/scales_visibility_data.npz";
        let mut archive = NpzArchive::new(std::fs::File::open(scales_visibility_path)?)?;
        let log_scales: Tensor<DiffBackend, 2> =
            io_utils::load_tensor_from_npz(&mut archive, "log_scale")?.require_grad();
        let visibility_filter: Tensor<DiffBackend, 1> =
            io_utils::load_tensor_from_npz(&mut archive, "visibility_filter")?;

        let dL_d_scales_grads_data_path = "test_data/dL_d_scales_grads_data.npz";
        let mut archive =
            NpzArchive::new(std::fs::File::open(dL_d_scales_grads_data_path)?)?;
        let dL_d_scales: Tensor<DiffBackend, 2> =
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
        compare_tensors::<DiffBackend, 2>(
            scale_grads,
            dL_d_scales.inner(),
            1e-4,
        );

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

        let img_out_data_path = "test_data/img_out_data.npz";
        let mut archive = NpzArchive::new(std::fs::File::open(img_out_data_path)?)?;
        let out_rgb: Tensor<DiffBackend, 3> = io_utils::load_tensor_from_npz(&mut archive, "out_rgb")?
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

        let dL_dL_sv_d_depth_normal_grads_path = "test_data/dL_sv_d_depth_normal_grads_data.npz";
        let mut archive =
            NpzArchive::new(std::fs::File::open(dL_dL_sv_d_depth_normal_grads_path)?)?;
        let dL_d_depth: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_depth")?.permute([1, 2, 0]);
        let dL_d_normal: Tensor<DiffBackend, 3> =
            io_utils::load_tensor_from_npz(&mut archive, "dL_d_normal")?.permute([1, 2, 0]);

        // Execute
        let sv_loss = sv_geometry_regularization_loss(&rendered_image, &cam, img_size, &gt_image);
        let grads = sv_loss.backward();

        // Assert that sv_loss is 0.6308
        let loss_val = sv_loss.into_scalar();
        assert!(
            (loss_val - 0.6308).abs() < 1e-2,
            "sv_loss is not close to 0.6308. Got: {}",
            loss_val
        );

        // Assert that alpha_grads and distance_grads are 0
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
            "distance_grads should be 0"
        );

        let depth_grads = out_depth.grad(&grads).expect("depth_grads should exist");
        compare_tensors::<DiffBackend, 3>(
            depth_grads.slice([160..166, 320..321, 0..1]),
            dL_d_depth.inner().slice([160..166, 320..321, 0..1]),
            1e-0,
        );

        let normal_grads = out_normal.grad(&grads).expect("normal_grads should exist");
        compare_tensors::<DiffBackend, 3>(
            normal_grads.slice([160..166, 320..321, 0..3]),
            dL_d_normal.inner().slice([160..166, 320..321, 0..3]),
            1e-1,
        );

        Ok(())
    }
}

pub fn geometry_regularization_loss<B: AutodiffBackend>(
    scales: &Tensor<B, 2>,
    rendered_image: &Tensor<B, 3>,
    rendered_image_n: &Tensor<B, 3>,
    render_aux: &RenderAux<B>,
    cam: &Camera,
    cam_n: &Camera,
    img_size: glam::UVec2,
    gt_image: &Tensor<B, 3>,
    gt_image_n: &Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [h, w, c] = rendered_image.dims();
    assert_eq!(
        c, 9,
        "rendered_image must have 9 channels (RGBA, Normal, Depth)"
    );

    let rendered_rgba = rendered_image.clone().slice([0..h, 0..w, 0..4]);
    let rendered_normal = rendered_image.clone().slice([0..h, 0..w, 4..7]);
    let rendered_depth: Tensor<B, 2> = rendered_image.clone().slice([0..h, 0..w, 8..9]).squeeze();
    let rendered_alpha: Tensor<B, 3> = rendered_image.clone().slice([0..h, 0..w, 3..4]).detach();

    // Scale Loss
    let visible: Tensor<B, 1> =
        Tensor::from_primitive(TensorPrimitive::Float(render_aux.visible.clone()));
    let num_visible = visible.clone().sum().into_scalar().elem::<f32>();

    let mut scale_loss: Tensor<B, 1> = Tensor::zeros([1], &scales.device());
    if num_visible > 0.0_f32 {
        let visible_scales = scales.clone() * visible.clone().unsqueeze_dim(1);
        let sorted_scales = visible_scales.sort(1);
        let min_scales = sorted_scales.slice([0..scales.dims()[0], 0..1]);
        scale_loss = min_scales.sum() / num_visible;
        println!("Scale Loss: {}", scale_loss);
    }

    // SV Loss
    let focal = cam.focal(img_size);
    let center = cam.center(img_size);

    let mut depth_normal = compute_normal_from_depth(
        rendered_depth.clone(),
        focal.x,
        focal.y,
        Some(center.x),
        Some(center.y),
    );
    depth_normal = depth_normal * rendered_alpha.detach();

    let img_weight: Tensor<B, 2> = get_img_grad_weight(gt_image.clone())
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(0.0, 1.0)
        .powf_scalar(2.0);

    let normal_error = (depth_normal.clone() - rendered_normal.clone())
        .abs()
        .sum_dim(2);

    let normal_error = normal_error.squeeze();
    let sv_loss = (img_weight.clone() * normal_error).mean();

    println!("SV Loss: {}", sv_loss);

    // // MV Geo Loss
    // let [h_n, w_n, _] = rendered_image_n.dims();
    // let depth_n: Tensor<B, 2> = rendered_image_n.clone().slice([0..h_n, 0..w_n, 8..9]).squeeze();

    // let pcd_r = depth_to_point_cloud(
    //     rendered_depth.clone(),
    //     focal.x,
    //     focal.y,
    //     Some(center.x),
    //     Some(center.y),
    // );
    // let [h_pcd, w_pcd, _] = pcd_r.dims();
    // let pcd_r = pcd_r.reshape([h_pcd * w_pcd, 3]);

    // let world_view_transform_r = Mat4::from(cam.world_to_local());
    // let world_view_transform_n = Mat4::from(cam_n.world_to_local());

    // let r_to_n_transform_mat =
    //     (world_view_transform_n * world_view_transform_r.inverse()).transpose();
    // let r_to_n_transform = Tensor::<B, 1>::from_data(
    //     TensorData::new(r_to_n_transform_mat.to_cols_array().to_vec(), [16]),
    //     &gt_image.device(),
    // )
    // .reshape([4, 4]);
    // let n_to_r_transform_mat =
    //     (world_view_transform_r * world_view_transform_n.inverse()).transpose();
    // let n_to_r_transform = Tensor::<B, 1>::from_data(
    //     TensorData::new(n_to_r_transform_mat.to_cols_array().to_vec(), [16]),
    //     &gt_image.device(),
    // )
    // .reshape([4, 4]);

    // let pcd_r_n = crate::utils::transform_points(pcd_r.clone(), r_to_n_transform.clone());

    // let focal_n = cam_n.focal(img_size);
    // let center_n = cam_n.center(img_size);
    // let p_n = project_point_cloud_to_image(
    //     pcd_r_n.clone(),
    //     focal_n.x,
    //     focal_n.y,
    //     center_n.x,
    //     center_n.y,
    // );

    // let mut depth_mask = compute_projection_mask(
    //     p_n.clone(),
    //     pcd_r_n.clone().slice([0..h_pcd * w_pcd, 2..3]).squeeze(),
    //     w_n as u32,
    //     h_n as u32,
    //     0.1,
    // );
    // let p_n_depth = sample_depth_at_coordinates(p_n.clone(), depth_n);

    // let rays_n = pcd_r_n.clone()
    //     / (pcd_r_n
    //         .clone()
    //         .slice([0..h_pcd * w_pcd, 2..3])
    //         .clamp_min(1e-9));
    // let pcd_n_cam_new = rays_n * p_n_depth.unsqueeze_dim(1);

    // let pcd_n_r = crate::utils::transform_points(pcd_n_cam_new.clone(), n_to_r_transform);

    // let p_r_reprojected =
    //     project_point_cloud_to_image(pcd_n_r.clone(), focal.x, focal.y, center.x, center.y);

    // let pixel_error = compute_pixel_errors(p_r_reprojected, w_pcd as u32, h_pcd as u32);

    // let pixel_error_threshold = 1.0;
    // depth_mask = depth_mask.bool_and(pixel_error.clone().lower_elem(pixel_error_threshold));

    // let mut weights = (pixel_error.clone() * -1.0).exp().detach();
    // weights = weights
    //     .clone()
    //     .mask_where(depth_mask.clone().bool_not(), Tensor::zeros_like(&weights));

    // let weighted_error = weights.clone() * pixel_error;
    // let masked_error = weighted_error.mask_fill(depth_mask.clone().bool_not(), 0.0);
    // let num_valid = depth_mask.clone().float().sum();
    // let mv_geometric_loss = masked_error.sum() / num_valid.clamp_min(1.0);

    // println!("MV Geo Loss: {}", mv_geometric_loss);

    // create_visualization_image(
    //     &rendered_rgba,
    //     &rendered_depth,
    //     &rendered_normal,
    //     &weights,
    //     &img_weight,
    //     &depth_normal,
    // )
    // .await?;

    scale_loss + sv_loss
}

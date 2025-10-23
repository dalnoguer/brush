use anyhow::Result;
use brush_dataset::scene::sample_to_tensor_data;
use brush_render::{
    camera::Camera, gaussian_splats::Splats, render_aux::RenderAux, validation, MainBackend,
};
use brush_render_bwd::burn_glue::SplatForwardDiff;
use brush_serde::{load_splat_from_ply, DeserializeError};
use brush_vfs::DataSource;
use burn::backend::Autodiff;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorPrimitive};
use burn_wgpu::WgpuDevice;
use env_logger;
use glam::Vec3;
use npyz::npz::NpzArchive;
use std::sync::Arc;

pub mod geometry_regularization;
mod geometry_utils;
mod visualization_utils;
mod io_utils;


#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    type DiffBackend = Autodiff<MainBackend>;

    let device = WgpuDevice::DefaultDevice;

    // let ply_path = String::from("/Users/daln/results/object_capture/results/golden_set_vim_vs_colmap/apple_keyboard/splats/brush_geo_reg/point_cloud/point_cloud.ply");
    let ply_path = String::from("/Users/daln/results/object_capture/results/golden_set_vim_vs_colmap/apple_keyboard/splats/brush_geo_reg/sv_weight_0.015.ply");
    let source = DataSource::Path(ply_path);
    let vfs = Arc::new(source.into_vfs().await?);
    let ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    let main_ply_path = ply_paths.first().expect("unreachable");

    let reader = vfs
        .reader_at_path(main_ply_path)
        .await
        .map_err(DeserializeError)?;
    let splats: Splats<MainBackend> = load_splat_from_ply(reader, Option::None, device.clone())
        .await?
        .splats;

    let cam = Camera::new(
        glam::vec3(0.5105, 0.2538, 0.0553),
        glam::Quat::from_xyzw(0.92868414, 0.02292308, -0.36890709, -0.03046073),
        1.1927426335535698,
        0.7294988095857634,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(640, 360);

    let gt_image_path =
        String::from("/Users/daln/results/object_capture/results/golden_set_vim_vs_colmap/apple_keyboard/undistorted_colmap_dataset/images/MotionTrackingPrimary_00315.png");
    let gt_image_dyn = image::open(&gt_image_path)?.resize_to_fill(
        img_size.x,
        img_size.y,
        image::imageops::FilterType::Triangle,
    );
    let gt_image: Tensor<DiffBackend, 3> =
        Tensor::from_data(sample_to_tensor_data(gt_image_dyn), &device);

    let splats_diff = splats.into_autodiff::<DiffBackend>();

    // Render R
    let (normals, plane_distances) = splats_diff.local_normals_and_plane_distances(&cam);
    let diff_out = DiffBackend::render_splats(
        &cam,
        img_size,
        splats_diff.means.val().into_primitive().tensor(),
        splats_diff.log_scales.val().into_primitive().tensor(),
        splats_diff.rotation.val().into_primitive().tensor(),
        splats_diff.sh_coeffs.val().into_primitive().tensor(),
        splats_diff.raw_opacity.val().into_primitive().tensor(),
        normals.into_primitive().tensor(),
        plane_distances.into_primitive().tensor(),
        Vec3::ZERO,
        true,
    );

    let output_tensor = Tensor::from_primitive(TensorPrimitive::<DiffBackend>::Float(diff_out.img.clone()));
    visualization_utils::save_render_output(&output_tensor, &cam, img_size, "r", true).await?;

    // Use test data
    // let img_out_data_path = "/Users/daln/results/object_capture/results/golden_set_vim_vs_colmap/apple_keyboard/splats/brush_geo_reg/test_data/img_out_data.npz";
    // let mut archive = NpzArchive::new(std::fs::File::open(img_out_data_path)?)?;
    // let out_rgb: Tensor<DiffBackend, 3> =
    //     utils::load_tensor_from_npz(&mut archive, "out_rgb")?
    //         .permute([1, 2, 0])
    //         .require_grad();
    // let out_alpha: Tensor<DiffBackend, 3> =
    //     utils::load_tensor_from_npz(&mut archive, "out_alpha")?
    //         .permute([1, 2, 0])
    //         .require_grad();
    // let out_normal: Tensor<DiffBackend, 3> =
    //     utils::load_tensor_from_npz(&mut archive, "out_normal")?
    //         .permute([1, 2, 0])
    //         .require_grad();
    // let out_depth: Tensor<DiffBackend, 3> =
    //     utils::load_tensor_from_npz(&mut archive, "out_depth")?
    //         .permute([1, 2, 0])
    //         .require_grad();
    // let out_distance: Tensor<DiffBackend, 3> =
    //     utils::load_tensor_from_npz(&mut archive, "out_distance")?
    //         .permute([1, 2, 0])
    //         .require_grad();

    // let rendered_image = Tensor::cat(vec![out_rgb.clone(), out_alpha.clone(), out_normal.clone(), out_alpha.clone(), out_depth.clone(), out_distance.clone()], 2);

    // println!("Means: {}", splats_diff.means.val());
    // println!("Log scales: {}", splats_diff.log_scales.val());
    // println!("Rotation: {}", splats_diff.rotation.val());
    // println!("SH coeffs: {}", splats_diff.sh_coeffs.val());
    // println!("Opacity: {}", splats_diff.raw_opacity.val());

    // GeoReg losses
    // let visible: Tensor<DiffBackend, 1> =
    //     Tensor::from_primitive(TensorPrimitive::Float(diff_out.aux.visible));
    // let scale_loss = geometry_regularization::scale_loss(&splats_diff.scales(), &visible);
    // println!("Scale loss: {}", scale_loss);

    // let rendered_image: Tensor<DiffBackend, 3> =
    //     Tensor::from_primitive(TensorPrimitive::Float(diff_out.img.clone()));
    // println!("RGB: {:?}", rendered_image.dims());
    // println!(
    //     "RGB: {}",
    //     rendered_image.clone().slice([160..161, 320..321, 0..3])
    // );
    // println!(
    //     "Alpha: {}",
    //     rendered_image.clone().slice([160..161, 320..321, 3..4])
    // );
    // println!(
    //     "Depth: {}",
    //     rendered_image.clone().slice([160..161, 320..321, 8..9])
    // );
    // println!(
    //     "Distance: {}",
    //     rendered_image.clone().slice([160..161, 320..321, 9..10])
    // );
    // println!(
    //     "Normal: {}",
    //     rendered_image.clone().slice([160..161, 320..321, 4..7])
    // );

    // let sv_loss = geometry_regularization::sv_geometry_regularization_loss(
    //     &rendered_image,
    //     &cam,
    //     img_size,
    //     &gt_image,
    // );
    // println!("SV loss: {}", sv_loss);

    // let loss: Tensor<DiffBackend, 1> = 100 * scale_loss + 0.015 * sv_loss; 

    // validation::validate_tensor_val(&sv_loss, "sv_loss", None, None);

    // let grads = loss.backward();

    // if let Some(rgb_grads) = out_rgb.grad(&grads) {
    //     println!("dL/d_rgb: {}", rgb_grads);
    // }

    // if let Some(alpha_grads) = out_alpha.grad(&grads) {
    //     println!("dL/d_alpha: {}", alpha_grads);
    // }


    // if let Some(depth_grads) = out_depth.grad(&grads) {
    //     println!("dL/d_depth: {}", depth_grads.clone().slice([160..166, 320..321, 0..1]));
    // }

    // if let Some(normal_grads) = out_normal.grad(&grads) {
    //     println!("dL/d_normal: {}", normal_grads.clone().slice([160..166, 320..321, 0..3]));
    // }

    // if let Some(distance_grads) = out_distance.grad(&grads) {
    //     println!("dL/d_distance: {}", distance_grads.clone().slice([160..166, 320..321]));
    // }

    // if let Some(alpha_grads) = out_alpha.grad(&grads) {
    //     println!("dL/d_alpha: {}", alpha_grads.clone().slice([160..166, 320..321]));
    // }

    // if let Some(distance_grads) = out_distance.grad(&grads) {
    //     println!("dL/d_distance: {}", distance_grads);
    // }

    // if let Some(mean_grads) = splats_diff.means.grad(&grads) {
    //     println!("dL/d_mean: {}", mean_grads);
    // }

    // if let Some(rot_grads) = splats_diff.rotation.grad(&grads) {
    //     println!("dL/d_rotation: {}", rot_grads);
    // }

    // if let Some(opac_grads) = splats_diff.raw_opacity.grad(&grads) {
    //     println!("dL/d_opac: {}", opac_grads);
    // }

    // if let Some(log_scale_grads) = splats_diff.log_scales.grad(&grads) {
    //     println!("dL/d_log_scale: {}", log_scale_grads);
    // }

    // if let Some(sh_grads) = splats_diff.sh_coeffs.grad(&grads) {
    //     println!("dL/d_sh: {}", sh_grads);
    // }

    // brush_render::validation::validate_splat_gradients(&splats_diff, &grads);

    // // Save depth as uint16 png for debugging
    // let [h, w, _] = output_tensor.dims();
    // let depth: Tensor<DiffBackend, 2> = output_tensor.clone().slice([0..h, 0..w, 8..9]).squeeze();
    // let depth_data = depth
    //     .clone()
    //     .into_data_async()
    //     .await
    //     .into_vec::<f32>()
    //     .expect("Wrong tensor type");

    // // Convert f32 depth (in meters) to u16 (in millimeters)
    // let depth_u16: Vec<u16> = depth_data
    //     .iter()
    //     .map(|&d| (d * 1000.0).round().clamp(0.0, u16::MAX as f32) as u16)
    //     .collect();
    // let depth_image = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(w as u32, h as u32, depth_u16)
    //     .expect("Failed to create depth image from tensor data");
    // depth_image.save("render_depth_u16.png")?;

    Ok(())
}

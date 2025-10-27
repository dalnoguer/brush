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

    let ply_path = String::from("/Users/daln/results/object_capture/results/golden_set_vim_vs_colmap/apple_keyboard/splats/brush_geo_reg/point_cloud/point_cloud.ply");
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

    Ok(())
}

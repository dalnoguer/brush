use anyhow::Result;

use brush_dataset::{config::LoadDataseConfig, load_dataset};
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let device = WgpuDevice::DefaultDevice;
    let scene_path = String::from("/Users/daln/results/object_capture/geo_reg_brush/undistorted_arcore_dataset");

    let source = DataSource::Path(scene_path.clone());
    let vfs = Arc::new(source.into_vfs().await?);

    let load_args = &LoadDataseConfig {
        max_frames: Some(1000),
        max_resolution: 640,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
    };
    let result = load_dataset(vfs, load_args, &device);
    let dataset = result.await.unwrap().1;

    let train_scene = dataset.train;
    println!("Training dataset size: {}", train_scene.views.len());

    // Create a directory to store the output images
    let output_dir = "debug_images";
    std::fs::create_dir_all(output_dir)?;

    for (i, view) in train_scene.views.iter().enumerate() {
        // Save the main image
        let view_target_path = format!("{}/{:05}.png", output_dir, i);
        let view_original_path = format!("{}/{}", scene_path, view.image.path.display());

        std::fs::copy(&view_original_path, &view_target_path)?;
        // Save neighbor images
        let neighbor_indices = &train_scene.nearest_neighbors[i];
        for (j, &neighbor_idx) in neighbor_indices.iter().enumerate() {
            let neighbor_view = &train_scene.views[neighbor_idx as usize];
            let neighbor_target_path = format!("{}/{:05}_neighbor_{}.png", output_dir, i, j);
            let neighbor_original_path = format!("{}/{}", scene_path, neighbor_view.image.path.display());
            std::fs::copy(&neighbor_original_path, &neighbor_target_path)?;
        }
    }

    Ok(())
}

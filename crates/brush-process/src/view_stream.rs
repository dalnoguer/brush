use crate::message::ProcessMessage;

use async_fn_stream::TryStreamEmitter;
use brush_serde;
use brush_vfs::BrushVfs;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use glam::{Quat, Vec3};
use serde::Deserialize;
use std::{path::Path, pin::pin, sync::Arc};
use tokio_stream::StreamExt;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Deserialize)]
struct CameraInfo {
    focal_point: Vec3,
    focus_distance: f32,
}

pub(crate) async fn view_stream(
    vfs: Arc<BrushVfs>,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    let mut paths: Vec<_> = vfs.file_paths().collect();
    alphanumeric_sort::sort_path_slice(&mut paths);
    let client = WgpuRuntime::client(&device);

    println!("Paths: {paths:?}");

    for (i, path) in paths.iter().enumerate() {
        if path.extension() != Some("ply".as_ref()) {
            continue;
        }
        tokio_wasm::task::yield_now().await;

        log::info!("Loading single ply file");
        println!("Processing ply {path:?}");

        emitter
            .emit(ProcessMessage::StartLoading { training: false })
            .await;

        let sub_sample = None; // Subsampling a trained ply doesn't really make sense.
        let splat_stream = brush_serde::stream_splat_from_ply(
            vfs.reader_at_path(path).await?,
            sub_sample,
            device.clone(),
            true,
        );

        let mut splat_stream = pin!(splat_stream);
        while let Some(message) = splat_stream.next().await {
            let message = message?;

            // If there's multiple ply files in a zip, don't support animated plys, that would
            // get rather mind bending.
            let (frame, total_frames) = if paths.len() == 1 {
                (message.meta.current_frame, message.meta.frame_count)
            } else {
                (i as u32, paths.len() as u32)
            };

            // As loading concatenates splats each time, memory usage tends to accumulate a lot
            // over time. Clear out memory after each step to prevent this buildup.
            client.memory_cleanup();

            if path.file_name() == Some("full.ply".as_ref()) {
                emitter
                    .emit(ProcessMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(message.splats),
                        frame,
                        total_frames,
                        progress: message.meta.progress,
                    })
                    .await;
            } else {
                emitter
                    .emit(ProcessMessage::ViewAuxiliarySplat {
                        name: path.file_stem().unwrap().to_string_lossy().to_string(),
                        splats: Box::new(message.splats),
                    })
                    .await;
            }
        }
    }

    let camera_info_path = Path::new("camera.json");
    log::info!("Loading camera info from camera.json");
    if let Ok(mut reader) = vfs.reader_at_path(camera_info_path).await {
        let mut contents = Vec::new();
        tokio_wasm::io::AsyncReadExt::read_to_end(&mut reader, &mut contents).await?;
        let camera_info: CameraInfo = serde_json::from_slice(&contents)?;
        emitter
            .emit(ProcessMessage::CameraData {
                focal_point: camera_info.focal_point,
                focus_distance: camera_info.focus_distance,
                rotation: Quat::from_rotation_x(-(std::f32::consts::FRAC_PI_2 + 0.5)),
            })
            .await;
    } else {
        log::warn!("camera.json not found, using default camera values.");
    }

    let metadata_path = Path::new("metadata.txt");
    let metadata = if let Ok(mut reader) = vfs.reader_at_path(metadata_path).await {
        log::info!("Loading metadata from metadata.txt");
        let mut contents = String::new();
        tokio_wasm::io::AsyncReadExt::read_to_string(&mut reader, &mut contents).await?;
        contents
    } else {
        log::warn!("metadata.txt not found, using empty metadata.");
        String::new()
    };

    emitter.emit(ProcessMessage::ObjectMetadata { metadata }).await;

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Clear out memory after loading is fully done.
    client.memory_cleanup();

    Ok(())
}

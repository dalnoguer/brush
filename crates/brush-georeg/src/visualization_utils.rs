use anyhow::{Ok, Result};
use std::{io::Write};
use burn::tensor::{backend::{AutodiffBackend, Backend}, Bool, Tensor};
use image::{DynamicImage, GenericImage, ImageBuffer, Rgba};
use glam::Vec3;
use brush_render::camera::Camera;

pub async fn save_render_output<B: AutodiffBackend>(
    output_tensor: &Tensor<B, 3>,
    cam: &Camera,
    img_size: glam::UVec2,
    prefix: &str,
    save_point_cloud: bool,
) -> Result<()> {
    // Save rendered image
    let [h, w, _] = output_tensor.dims();
    let rgba = output_tensor.clone().slice([0..h, 0..w, 0..4]);
    let rgba_data = rgba
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    let image_bytes: Vec<u8> = rgba_data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let rgb_image =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(w as u32, h as u32, image_bytes.clone())
            .expect("Failed to create image from tensor data");

    let output_path = format!("{}_render_rgb.png", prefix);
    rgb_image.save(output_path)?;

    // Save normal image
    let normal = output_tensor.clone().slice([0..h, 0..w, 4..7]);
    let rgb_normal = (normal + 1) / 2;

    let [h, w, _] = rgb_normal.dims();

    let data = rgb_normal
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    let image_bytes: Vec<u8> = data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let image = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
        .expect("Failed to create image from tensor data");

    let output_path = format!("{}_render_normal.png", prefix);
    image.save(output_path)?;

    // Save depth image
    let depth: Tensor<B, 2> = output_tensor.clone().slice([0..h, 0..w, 8..9]).squeeze();
    let depth_data = depth
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    let (min_depth, max_depth) = depth_data
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), &v| {
            if v > 0.0 {
                (min.min(v), max.max(v))
            } else {
                (min, max)
            }
        });

    let jet_colormap = |v: f32| {
        let v_c = v.clamp(0.0, 1.0);
        // These formulas correspond to Red, Green, Blue components for a jet colormap
        let r_val = (1.5 - (4.0 * (v_c - 0.75)).abs()).clamp(0.0, 1.0);
        let g_val = (1.5 - (4.0 * (v_c - 0.5)).abs()).clamp(0.0, 1.0);
        let b_val = (1.5 - (4.0 * (v_c - 0.25)).abs()).clamp(0.0, 1.0);
        [
            (r_val * 255.0) as u8,
            (g_val * 255.0) as u8,
            (b_val * 255.0) as u8,
        ]
    };

    let image_bytes: Vec<u8> = depth_data
        .iter()
        .flat_map(|d| {
            let normalized = if *d > 0.0 && max_depth > min_depth {
                (d - min_depth) / (max_depth - min_depth)
            } else {
                0.0
            };
            let [r, g, b] = jet_colormap(normalized);
            [r, g, b].to_vec()
        })
        .collect();

    let image = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
        .expect("Failed to create image from tensor data");
    image.save(format!("{}_render_depth.png", prefix))?;

    // Save distance image
    let distance: Tensor<B, 2> = output_tensor.clone().slice([0..h, 0..w, 9..10]).squeeze();
    let distance_data = distance
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    let (min_distance, max_distance) =
        distance_data
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| {
                if v > 0.0 {
                    (min.min(v), max.max(v))
                } else {
                    (min, max)
                }
            });

    let image_bytes: Vec<u8> = distance_data
        .iter()
        .flat_map(|d| {
            let normalized = if *d > 0.0 && max_distance > min_distance {
                (d - min_distance) / (max_distance - min_distance)
            } else {
                0.0
            };
            let [r, g, b] = jet_colormap(normalized);
            [r, g, b].to_vec()
        })
        .collect();

    let image = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
        .expect("Failed to create image from tensor data");
    image.save(format!("{}_render_distance.png", prefix))?;

    if save_point_cloud {
        // Backproject depth to point cloud
        let focal = cam.focal(img_size);
        let center = cam.center(img_size);

        let mut points = Vec::new();
        let mut point_colors = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let depth_val = depth_data[y * w + x];
                if depth_val <= 0.0 || depth_val > 1.0 {
                    continue;
                }

                // Backproject to camera space, then transform to world space
                let cam_x = (x as f32 + 0.5 - center.x) * depth_val / focal.x;
                let cam_y = (y as f32 + 0.5 - center.y) * depth_val / focal.y;
                let cam_z = depth_val;

                let point_in_cam_space = Vec3::new(cam_x, cam_y, cam_z);

                let pixel = rgb_image.get_pixel(x as u32, y as u32);

                points.push(point_in_cam_space);
                point_colors.push([pixel.0[0], pixel.0[1], pixel.0[2]]);
            }
        }

        if !points.is_empty() {
            let filename = format!("{}_pointcloud.ply", prefix);
            let mut file = std::fs::File::create(&filename)?;
            writeln!(file, "ply")?;
            writeln!(file, "format ascii 1.0")?;
            writeln!(file, "element vertex {}", points.len())?;
            writeln!(file, "property float x")?;
            writeln!(file, "property float y")?;
            writeln!(file, "property float z")?;
            writeln!(file, "property uchar red")?;
            writeln!(file, "property uchar green")?;
            writeln!(file, "property uchar blue")?;
            writeln!(file, "end_header")?;

            for (point, color) in points.iter().zip(point_colors.iter()) {
                writeln!(
                    file,
                    "{} {} {} {} {} {}",
                    point.x, point.y, point.z, color[0], color[1], color[2]
                )?;
            }
            println!("✅ Saved point cloud to {}", filename);
        }
    }

    println!("✅ Rendered images for prefix '{}'", prefix);
    Ok(())
}

async fn save_normal<B: AutodiffBackend>(path: &str, normal: &Tensor<B, 3>) -> Result<()> {
    let [h, w, c] = normal.dims();
    assert!(c == 3, "normal must have 3 channels");
    let rgb_normal = (normal.clone().slice([0..h, 0..w, 0..3]) + 1) / 2;
    let data = rgb_normal
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");
    let image_bytes: Vec<u8> = data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let rendered_normal_image =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
            .expect("Failed to create image from tensor data");
    rendered_normal_image.save(path)?;

    Ok(())
}

async fn create_visualization_image<B: AutodiffBackend>(
    rendered_image_tensor: &Tensor<B, 3>,
    rendered_depth_tensor: &Tensor<B, 2>,
    rendered_normal_tensor: &Tensor<B, 3>,
    weights_tensor: &Tensor<B, 1>,
    img_weight_tensor: &Tensor<B, 2>,
    depth_normal_tensor: &Tensor<B, 3>,
) -> Result<()> {
    let [h, w] = rendered_depth_tensor.dims();

    // Rendered depth tensor
    let depth_data = rendered_depth_tensor
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    let (min_depth, max_depth) = depth_data
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), &v| {
            if v > 0.0 {
                (min.min(v), max.max(v))
            } else {
                (min, max)
            }
        });

    let jet_colormap = |v: f32| {
        let v_c = v.clamp(0.0, 1.0);
        let r_val = (1.5 - (4.0 * (v_c - 0.75)).abs()).clamp(0.0, 1.0);
        let g_val = (1.5 - (4.0 * (v_c - 0.5)).abs()).clamp(0.0, 1.0);
        let b_val = (1.5 - (4.0 * (v_c - 0.25)).abs()).clamp(0.0, 1.0);
        [
            (r_val * 255.0) as u8,
            (g_val * 255.0) as u8,
            (b_val * 255.0) as u8,
        ]
    };

    let image_bytes: Vec<u8> = depth_data
        .into_iter()
        .flat_map(|d| {
            let normalized = if d > 0.0 && max_depth > min_depth {
                (d - min_depth) / (max_depth - min_depth)
            } else {
                0.0
            };
            jet_colormap(normalized).to_vec()
        })
        .collect();
    let rendered_depth_image =
        ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
            .expect("Failed to create image from tensor data");

    // Rendered image
    let rgb_data = rendered_image_tensor
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");
    let image_bytes: Vec<u8> = rgb_data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    // Assuming rendered_image_tensor has 4 channels (RGBA)
    let rendered_image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(w as u32, h as u32, image_bytes.clone())
            .expect("Failed to create rendered image from tensor data");

    // Rendered normal
    let rgb_normal = (rendered_normal_tensor.clone().slice([0..h, 0..w, 0..3]) + 1) / 2;
    let data = rgb_normal
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");
    let image_bytes: Vec<u8> = data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let rendered_normal_image =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
            .expect("Failed to create image from tensor data");

    // Weights
    let weights_data = weights_tensor
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type for weights");
    let max_weight = weights_data.iter().fold(0.0f32, |acc, &x| acc.max(x));
    let norm_factor = if max_weight > 0.0 {
        1.0 / max_weight
    } else {
        0.0
    };
    let weights_bytes: Vec<u8> = weights_data
        .into_iter()
        .flat_map(|v| jet_colormap(v * norm_factor).to_vec())
        .collect();
    let weights_image =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, weights_bytes)
            .expect("Failed to create weights image from tensor data");

    // Image Weight
    let img_weight_data = img_weight_tensor
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");
    let img_weight_bytes: Vec<u8> = img_weight_data
        .into_iter()
        .flat_map(|v| jet_colormap(v.clamp(0.0, 1.0)).to_vec())
        .collect();
    let img_weight_image =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, img_weight_bytes)
            .expect("Failed to create image weight image from tensor data");

    // Depth normal
    let rgb_normal = (depth_normal_tensor.clone().slice([0..h, 0..w, 0..3]) + 1) / 2;
    let data = rgb_normal
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");
    let image_bytes: Vec<u8> = data
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let depth_normal_image =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
            .expect("Failed to create image from tensor data");

    // --- Compose images into a 2x3 grid ---
    let mut composite_image = ImageBuffer::new(w as u32 * 3, h as u32 * 2);
    composite_image.copy_from(&DynamicImage::ImageRgb8(rendered_depth_image), 0, 0)?;
    composite_image.copy_from(&DynamicImage::ImageRgba8(rendered_image), w as u32, 0)?;
    composite_image.copy_from(
        &DynamicImage::ImageRgb8(rendered_normal_image),
        w as u32 * 2,
        0,
    )?;
    composite_image.copy_from(&DynamicImage::ImageRgb8(weights_image), 0, h as u32)?;
    composite_image.copy_from(
        &DynamicImage::ImageRgb8(img_weight_image),
        w as u32,
        h as u32,
    )?;
    composite_image.copy_from(
        &DynamicImage::ImageRgb8(depth_normal_image),
        w as u32 * 2,
        h as u32,
    )?;

    composite_image.save("visualization.png")?;
    println!("✅ Saved visualization to visualization.png");

    Ok(())
}

/// Saves a boolean tensor as a grayscale image.
///
/// # Arguments
///
/// * `path` - The path to save the image file.
/// * `mask` - A boolean tensor of shape (H, W). True values will be white, False will be black.
pub async fn save_mask<B: Backend>(path: &str, mask: &Tensor<B, 2, Bool>) -> Result<()> {
    let [h, w] = mask.dims();

    // Convert boolean tensor to float tensor (true -> 1.0, false -> 0.0)
    let float_mask = mask.clone().float();

    let data = float_mask
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type");

    // Map float values to grayscale u8 values (0 or 255)
    let image_bytes: Vec<u8> = data.into_iter().map(|v| (v * 255.0) as u8).collect();

    // Create a grayscale image
    let image = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(w as u32, h as u32, image_bytes)
        .expect("Failed to create image from tensor data");

    image.save(path)?;
    Ok(())
}

/// Saves a point cloud tensor to a PLY file.
///
/// # Arguments
///
/// * `point_cloud` - Tensor of shape (N, 3) representing the point cloud.
/// * `colors` - Optional tensor of shape (N, 3) for RGB colors, with values in [0, 1].
/// * `filename` - The path to save the PLY file.
pub async fn save_point_cloud_to_ply<B: Backend>(
    point_cloud: &Tensor<B, 2>,
    colors: Option<&Tensor<B, 2>>,
    filename: &str,
) -> Result<()> {
    let [num_points, dim] = point_cloud.dims();
    assert!(dim == 3, "Point cloud tensor must have shape (N, 3)");

    let points_data = point_cloud
        .clone()
        .into_data_async()
        .await
        .into_vec::<f32>()
        .expect("Wrong tensor type for point cloud");

    let colors_data = if let Some(colors_tensor) = colors {
        let [num_colors, color_dim] = colors_tensor.dims();
        assert!(
            num_colors == num_points && color_dim == 3,
            "Colors tensor must have shape (N, 3)"
        );
        let color_vec = colors_tensor
            .clone()
            .into_data_async()
            .await
            .into_vec::<f32>()
            .expect("Wrong tensor type for colors");
        Some(color_vec)
    } else {
        None
    };

    let mut file = std::fs::File::create(filename)?;
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", num_points)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    if colors.is_some() {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }
    writeln!(file, "end_header")?;

    for i in 0..num_points {
        write!(
            file,
            "{} {} {}",
            points_data[i * 3],
            points_data[i * 3 + 1],
            points_data[i * 3 + 2]
        )?;
        if let Some(colors) = &colors_data {
            write!(
                file,
                " {} {} {}",
                (colors[i * 3] * 255.0) as u8,
                (colors[i * 3 + 1] * 255.0) as u8,
                (colors[i * 3 + 2] * 255.0) as u8
            )?;
        }
        writeln!(file)?;
    }
    println!("✅ Saved point cloud to {}", filename);
    Ok(())
}

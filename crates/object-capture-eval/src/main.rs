#![recursion_limit = "256"]

use std::path::{Path, PathBuf};

use anyhow::Result;
use argh::FromArgs;
use brush_dataset::{
    load_dataset,
    scene::{sample_to_tensor_data, view_to_sample_image},
};
use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::{burn_init_setup, create_process};
use brush_render::MainBackend;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, SplatForward};
use brush_train::eval::EvalSample;
use brush_train::ssim::Ssim;
use brush_vfs::DataSource;
use burn::tensor::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive, s};
use burn_wgpu::WgpuDevice;
use glam::Vec3;
use image::DynamicImage;
use image::Rgb32FImage;
use tokio_stream::StreamExt;

/// Options for the evaluation script.
#[derive(FromArgs, Debug)]
struct EvalOptions {
    /// paths to the datasets to evaluate
    #[argh(positional)]
    dataset_paths: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).init();

    let opts: EvalOptions = argh::from_env();

    // Evaluation config
    let mut config = TrainStreamConfig::default();
    config.load_config.eval_split_every = Some(8);

    let device = burn_init_setup().await;
    <MainBackend as Backend>::seed(&device, 42);
    for dataset_path in &opts.dataset_paths {
        println!("Processing dataset: {}", dataset_path.display());

        let nerfstudio_dataset_path = dataset_path.join("nerfstudio_dataset");

        let (psnr, ssim, num_gaussians) =
            train_and_eval(&nerfstudio_dataset_path, config.clone(), &device.clone()).await?;

        println!("  PSNR: {}", psnr);
        println!("  SSIM: {}", ssim);
        println!("  Gaussians: {}", num_gaussians);
    }

    Ok(())
}

async fn train_and_eval(
    dataset_path: &Path,
    mut config: TrainStreamConfig,
    device: &WgpuDevice,
) -> anyhow::Result<(f32, f32, usize)> {
    let dataset_path_str = dataset_path.to_str().unwrap().to_string();
    let source = DataSource::Path(dataset_path_str);

    let dataset_name = dataset_path
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let output_dir = PathBuf::from("eval_results").join(dataset_name);
    std::fs::create_dir_all(&output_dir)?;
    config.process_config.export_path = output_dir.to_str().unwrap().to_string();

    let mut load_config = config.load_config.clone();
    let mut process = create_process(source.clone(), async move |_| config);

    let mut splats: Option<Splats<MainBackend>> = None;

    while let Some(message_result) = process.stream.next().await {
        match message_result {
            Ok(ProcessMessage::SplatsUpdated { .. }) => {
                if let Some(s) = process.splat_view.get_main() {
                    splats = Some(s.clone());
                }
            }
            Ok(ProcessMessage::TrainMessage(TrainMessage::DoneTraining)) => {
                break;
            }
            Ok(_) => {}
            Err(e) => {
                println!("Error during training: {e}");
            }
        }
    }

    let splats = splats.expect("Training finished without producing splats");

    // Load eval dataset with masks
    let vfs = source.clone().into_vfs().await?;
    load_config.load_masks = true;
    let load_result = load_dataset(vfs.clone(), &load_config).await?;
    let dataset = load_result.dataset;

    let eval_scene = dataset
        .eval
        .as_ref()
        .expect("Dataset has no evaluation scene");

    let (psnr, ssim, num_gaussians) = run_eval(device, splats, eval_scene, output_dir).await?;

    Ok((psnr, ssim, num_gaussians))
}

async fn run_eval(
    device: &WgpuDevice,
    splats: Splats<MainBackend>,
    eval_scene: &brush_dataset::scene::Scene,
    output_dir: PathBuf,
) -> anyhow::Result<(f32, f32, usize)> {
    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;

    for (i, view) in eval_scene.views.iter().enumerate() {
        let eval_img = view.image.load().await?;
        let sample = eval_stats_with_mask(
            &splats,
            &view.camera,
            eval_img,
            view.image.alpha_mode(),
            device,
        )?;

        save_eval_sample_to_disk(&sample, &output_dir.join(&format!("eval_image_{}.png", i)))
            .await?;

        count += 1;
        psnr += sample.psnr.into_scalar().elem::<f32>();
        ssim += sample.ssim.into_scalar().elem::<f32>();
    }

    psnr /= count as f32;
    ssim /= count as f32;
    let n_splats = splats.means.val().shape().dims[0];

    Ok((psnr, ssim, n_splats))
}

fn eval_stats_with_mask<B: Backend + SplatForward<B>>(
    splats: &Splats<B>,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &B::Device,
) -> Result<EvalSample<B>> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let gt_tensor = sample_to_tensor_data(view_to_sample_image(gt_img.clone(), alpha_mode));
    let gt_tensor = Tensor::from_data(gt_tensor, device);
    
    let gt_rgb = gt_tensor.clone().slice(s![.., .., 0..3]);

    let gt_alpha = gt_tensor.slice(s![.., .., 3..4]); 

    let alpha_sum = gt_alpha.clone().sum(); 
    
    let (img, aux) = {
        let (img, aux) = B::render_splats(
            gt_cam,
            res,
            splats.means.val().into_primitive().tensor(),
            splats.log_scales.val().into_primitive().tensor(),
            splats.rotations.val().into_primitive().tensor(),
            splats.sh_coeffs.val().into_primitive().tensor(),
            splats.raw_opacities.val().into_primitive().tensor(),
            splats.render_mode,
            Vec3::ZERO,
            true,
        );
        (Tensor::from_primitive(TensorPrimitive::Float(img)), aux)
    };
    let render_rgb = img.slice(s![.., .., 0..3]);

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let diff = render_rgb.clone() - gt_rgb.clone();
    let squared_error = diff.powi_scalar(2);
    
    let masked_squared_error = squared_error * gt_alpha.clone();
    let mse = masked_squared_error.sum() / (alpha_sum.clone() * 3.0);

    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;

    let ssim_measure = Ssim::new(11, 3, device);
    
    let ssim_map = ssim_measure.ssim(render_rgb.clone(), gt_rgb);
    
    let masked_ssim_sum = (ssim_map * gt_alpha).sum();
    let ssim = masked_ssim_sum / (alpha_sum * 3.0);

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        rendered: render_rgb,
        aux,
    })
}

pub async fn save_eval_sample_to_disk<B: Backend>(
    sample: &EvalSample<B>,
    path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("eval");
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let dir = path.parent().unwrap_or(Path::new("."));

    let gt_path = dir.join(format!("{}_gt.{}", stem, extension));
    let rendered_path = dir.join(format!("{}_rendered.{}", stem, extension));

    sample.gt_img.save(gt_path)?;

    let img_tensor = sample.rendered.clone();
    let dims = img_tensor.dims();
    let (h, w) = (dims[0], dims[1]);

    let data = img_tensor.into_data_async().await?.into_vec::<f32>()?;

    let rendered_img: DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from tensor data"))?
        .into();

    let rendered_rgb8 = rendered_img.into_rgb8();
    rendered_rgb8.save(rendered_path)?;

    Ok(())
}

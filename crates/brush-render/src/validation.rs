use burn::{prelude::Backend, tensor::Tensor};

pub fn validate_tensor_val<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    name: &str,
    min_val: Option<f32>,
    max_val: Option<f32>,
) {
    let data = tensor.clone().into_data();
    let values = data
        .into_vec::<f32>()
        .expect("Failed to convert tensor to f32 vec");

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut below_min_count = 0;
    let mut above_max_count = 0;

    for &value in &values {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
        } else {
            if let Some(min) = min_val
                && value < min
            {
                below_min_count += 1;
            }
            if let Some(max) = max_val
                && value > max
            {
                above_max_count += 1;
            }
        }
    }

    if nan_count > 0 || inf_count > 0 {
        log::error!(
            "Tensor '{}' contains invalid values: {} NaN, {} infinite (out of {} total).\nSample: {:?}",
            name,
            nan_count,
            inf_count,
            values.len(),
            &values[0..values.len().min(16)]
        );
    }

    if below_min_count > 0 {
        log::error!(
            "Tensor '{}' contains {} values below minimum {} (out of {} total)",
            name,
            below_min_count,
            min_val.unwrap(),
            values.len()
        );
    }

    if above_max_count > 0 {
        log::error!(
            "Tensor '{}' contains {} values above maximum {} (out of {} total)",
            name,
            above_max_count,
            max_val.unwrap(),
            values.len()
        );
    }
}

pub fn validate_gradient_finite<B: Backend, const D: usize>(gradient: &Tensor<B, D>, name: &str) {
    validate_tensor_val(gradient, &format!("gradient_{name}"), None, None);
}

pub fn validate_splat_gradients<B>(
    splats: &crate::gaussian_splats::Splats<B>,
    gradients: &B::Gradients,
) where
    B: burn::tensor::backend::AutodiffBackend,
{
    if let Some(mean_grad) = splats.means.grad(gradients) {
        validate_gradient_finite(&mean_grad, "means");
    }
    if let Some(rotation_grad) = splats.rotations.grad(gradients) {
        validate_gradient_finite(&rotation_grad, "rotation");
    }
    if let Some(scales_grad) = splats.log_scales.grad(gradients) {
        validate_gradient_finite(&scales_grad, "log_scales");
    }
    if let Some(sh_grad) = splats.sh_coeffs.grad(gradients) {
        validate_gradient_finite(&sh_grad, "sh_coeffs");
    }
    if let Some(opacity_grad) = splats.raw_opacities.grad(gradients) {
        validate_gradient_finite(&opacity_grad, "raw_opacity");
    }
}

pub fn print_tensor_stats<B: Backend>(tensor: &Tensor<B, 1>, name: &str) {
    let mean_tensor = tensor.clone().mean();
    let var_tensor = tensor.clone().var(0);
    let std_tensor = var_tensor.sqrt();

    let to_f32 = |t: Tensor<B, 1>| -> f32 {
        t.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0]
    };

    let mean_val = to_f32(mean_tensor);
    let std_val = to_f32(std_tensor);

    let data = tensor.to_data();
    let mut values = data.convert::<f32>().to_vec::<f32>().unwrap();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if values.is_empty() {
        println!("Tensor '{}' is empty.", name);
        return;
    }

    let len = values.len();
    let min = values[0];
    let max = values[len - 1];
    let median = values[len / 2];

    let p90_idx = ((len as f32) * 0.90) as usize;
    let p90 = values[p90_idx.min(len - 1)];
    
    let p99_idx = ((len as f32) * 0.99) as usize;
    let p99 = values[p99_idx.min(len - 1)];

    println!("--- Statistics for '{}' ---", name);
    println!("Shape:      {:?}", tensor.shape());
    println!("Count:      {}", len);
    println!("Mean:       {:.4}", mean_val);
    println!("Std Dev:    {:.4}", std_val);
    println!("Min:        {:.4}", min);
    println!("Median:     {:.4}", median);
    println!("P90:        {:.4}", p90);
    println!("P99:        {:.4}", p99);
    println!("Max:        {:.4}", max);
    println!("----------------------------");
}
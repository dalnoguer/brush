use anyhow::Result;
use std::io::{Read, Seek};
use npyz::npz::NpzArchive;
use burn::tensor::{backend::Backend, Element, Tensor, TensorData};

/// Loads a tensor from an NPZ archive by name.
///
/// # Arguments
///
/// * `archive` - A mutable reference to the `NpzArchive`.
/// * `name` - The name of the tensor to load (e.g., "tensor.npy").
///
/// # Returns
///
/// A `Result` containing the loaded `Tensor` or an error.
pub fn load_tensor_from_npz<
    B: Backend,
    R: Read + Seek,
    const D: usize,
>(
    archive: &mut NpzArchive<R>,
    name: &str,
) -> Result<Tensor<B, D>>
where
    B::FloatElem: Element + npyz::Deserialize,
{
    let npy_file = archive
        .by_name(name)?
        .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found in NPZ archive", name))?;
    let shape: Vec<usize> = npy_file.shape().iter().map(|&d| d as usize).collect();

    anyhow::ensure!(
        shape.len() == D,
        "Mismatched dimensions for tensor '{}'. Expected {}, got {}",
        name,
        D,
        shape.len()
    );

    let values: Vec<B::FloatElem> = npy_file.into_vec()?;
    let data = TensorData::new(values, shape);
    Ok(Tensor::<B, D>::from_data(data, &B::Device::default()))
}
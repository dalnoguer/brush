#![recursion_limit = "256"]

// Platform-specific modules.
#[cfg(all(target_os = "android", feature = "ui"))]
mod android;
#[cfg(target_family = "wasm")]
pub mod wasm;

// FFI for native training API.
#[cfg(feature = "training")]
#[cfg(not(target_family = "wasm"))]
pub mod ffi;

mod shared;

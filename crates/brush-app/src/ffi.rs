use brush_process::burn_init_setup;
use brush_process::config::TrainStreamConfig;
use brush_process::message::TrainMessage;
use brush_process::{create_process, message::ProcessMessage};
use brush_vfs::DataSource;
use log::error;
use std::convert::TryFrom;
use std::ffi::{CStr, c_char, c_void};
use std::panic;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

use crate::shared::startup;

#[repr(C)]
pub enum TrainExitCode {
    Success = 0,
    Error = 1,
}

#[repr(C)]
pub enum ProgressMessage {
    NewProcess,
    Training { iter: u32 },
    DoneTraining,
    VisualizationUpdated {
        data: *const u8,
        len: u32,
        width: u32,
        height: u32,
    },
}

impl TryFrom<&ProcessMessage> for ProgressMessage {
    type Error = ();

    fn try_from(value: &ProcessMessage) -> Result<Self, Self::Error> {
        match value {
            ProcessMessage::NewProcess => Ok(Self::NewProcess),
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
                Ok(Self::Training { iter: *iter })
            }
            ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => Ok(Self::DoneTraining),
            
            // Map the Rust Vec to raw pointers for C#
            ProcessMessage::VisualizationUpdated { image, width, height } => {
                Ok(Self::VisualizationUpdated {
                    data: image.as_ptr(),
                    len: image.len() as u32,
                    width: *width,
                    height: *height,
                })
            }
            
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptions {
    pub total_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
}

impl TrainOptions {
    /// # Safety
    ///
    /// If `output_path` is not null, it must be a valid pointer to a null-terminated C string.
    unsafe fn into_train_stream_config(self) -> TrainStreamConfig {
        let process_args = TrainStreamConfig::default();
        let mut process_args = process_args;
        if !self.output_path.is_null() {
            // SAFETY: Path is not null, caller guarantees the string is a valid C-string.
            process_args.process_config.export_path = unsafe {
                CStr::from_ptr(self.output_path)
                    .to_string_lossy()
                    .into_owned()
            };
        }
        process_args.train_config.total_steps = self.total_steps;
        process_args.train_config.refine_every = self.refine_every;
        process_args.load_config.max_resolution = self.max_resolution;
        process_args.process_config.export_every = self.export_every;
        process_args.process_config.eval_save_to_disk = true;
        process_args
    }
}

pub type ProgressCallback =
    extern "C" fn(progress_message: ProgressMessage, user_data: *mut c_void);

static SETUP: OnceCell<()> = OnceCell::const_new();

#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) -> TrainExitCode {
    #[cfg(target_os = "android")]
    {
        // 1. Initialize Logger
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Error)
                .with_tag("RustLayer"),
        );

        // 2. REGISTER PANIC HOOK
        // This catches the panic info and logs it to Logcat before the app aborts.
        panic::set_hook(Box::new(|panic_info| {
            let (file, line) = match panic_info.location() {
                Some(loc) => (loc.file(), loc.line()),
                None => ("unknown", 0),
            };

            let msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                *s
            } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                &**s
            } else {
                "Box<Any>"
            };

            // Log with "Fatal" priority so it stands out
            error!("RUST PANIC at {}:{}: {}", file, line, msg);
        }));
    }

    // Wrap the logic in catch_unwind to prevent unwinding across FFI boundary
    // (If you use panic="abort" in Cargo.toml, this won't catch it,
    // but the hook above WILL still log it).
    let result = panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        run_training_logic(dataset_path, options, progress_callback, user_data)
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            error!("Rust panicked and was caught at FFI boundary.");
            TrainExitCode::Error
        }
    }
}

unsafe fn run_training_logic(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) -> TrainExitCode {
    if dataset_path.is_null() || options.is_null() {
        return TrainExitCode::Error;
    }

    let dataset_path_str = unsafe { CStr::from_ptr(dataset_path).to_string_lossy().into_owned() };
    let source = DataSource::Path(dataset_path_str);
    let train_options = unsafe { *options };
    let process_args = unsafe { train_options.into_train_stream_config() };

    let mut process = create_process(source, async move |_| process_args);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
        .block_on(async {
            SETUP
                .get_or_init(async move || {
                    startup();
                    burn_init_setup().await;
                })
                .await;

            while let Some(message_result) = process.stream.next().await {
                match message_result {
                    Ok(message) => {
                        if let Ok(progress_message) = (&message).try_into() {
                            progress_callback(progress_message, user_data);
                        }
                    }
                    Err(_) => {
                        return TrainExitCode::Error;
                    }
                }
            }

            TrainExitCode::Success
        })
}

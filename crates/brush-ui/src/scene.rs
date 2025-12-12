use brush_dataset::config::AlphaMode;
use brush_process::message::ProcessMessage;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use egui::TextBuffer;
use egui::{Align2, Area, Frame, Pos2, Ui, epaint::mutex::RwLock as EguiRwLock};
use parking_lot::Mutex;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::io::Write;

use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
    sh::rgb_to_sh,
};
use burn::tensor::{Tensor, s};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect, Slider, collapsing_header::CollapsingState};
use glam::{UVec2, Vec3};
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;
use web_time::Instant;

use brush_gemini::GeminiClient;

use crate::{
    UiMode,
    app::CameraSettings,
    burn_texture::BurnTexture,
    draw_checkerboard,
    panels::AppPane,
    ui_process::UiProcess,
    widget_3d::Widget3D,
};

#[derive(Clone, PartialEq)]
struct RenderState {
    size: UVec2,
    cam: Camera,
    frame: f32,
    settings: CameraSettings,
    grid_opacity: f32,
    selected_aux_splat: Option<String>,
}

pub enum SceneCommand {
    SelectSplat(Option<String>),
}

struct ErrorDisplay {
    headline: String,
    context: Vec<String>,
}

impl ErrorDisplay {
    fn new(error: &anyhow::Error) -> Self {
        let headline = error.to_string();
        let context = error
            .chain()
            .skip(1)
            .map(|cause| format!("{cause}"))
            .collect();
        Self { headline, context }
    }

    fn draw(&self, ui: &mut egui::Ui) {
        ui.heading(format!("❌ {}", self.headline));
        ui.indent("err_context", |ui| {
            for c in &self.context {
                ui.label(format!("• {c}"));
                ui.add_space(2.0);
            }
        });
    }
}

async fn export(splat: Splats<MainBackend>) -> Result<(), anyhow::Error> {
    let data = brush_serde::splat_to_ply(splat).await?;
    rrfd::save_file("export.ply", data).await?;
    Ok(())
}

#[derive(Default, PartialEq, Clone, Copy)]
enum AudioState {
    #[default]
    Idle,
    Recording,
    Thinking,
    Playing,
}

fn box_ui<R>(
    id: &str,
    ui: &egui::Ui,
    pivot: Align2,
    pos: Pos2,
    add_contents: impl FnOnce(&mut Ui) -> R,
) {
    // Controls window in bottom right
    let id = ui.id().with(id);
    egui::Area::new(id)
        .kind(egui::UiKind::Window)
        .pivot(pivot)
        .current_pos(pos)
        .movable(false)
        .show(ui.ctx(), |ui| {
            let style = ui.style_mut();
            let fill = style.visuals.window_fill;
            style.visuals.window_fill =
                Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 200);
            let frame = Frame::window(style);

            frame.show(ui, add_contents);
        });
}

pub struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<MainBackend>>,
    view_auxiliary_splats: HashMap<String, Splats<MainBackend>>,

    object_metadata: String,
    gemini_client: Option<Arc<GeminiClient>>,

    fully_loaded: bool,
    frame_count: u32,
    frame: f32,

    // Ui state.
    live_update: bool,
    paused: bool,
    audio_state: Arc<Mutex<AudioState>>,
    recording_stop_signal: Option<Arc<AtomicBool>>,
    err: Option<ErrorDisplay>,
    warnings: Vec<ErrorDisplay>,

    export_channel: (
        UnboundedSender<anyhow::Error>,
        UnboundedReceiver<anyhow::Error>,
    ),

    command_tx: UnboundedSender<SceneCommand>,
    command_rx: UnboundedReceiver<SceneCommand>,

    // Keep track of what was last rendered.
    last_state: Option<RenderState>,

    // 3D widgets for visualization
    widget_3d: Option<Widget3D>,

    // Selected auxiliary splat to render
    selected_aux_splat: Option<String>,
}

impl ScenePanel {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        let channel = tokio::sync::mpsc::unbounded_channel();

        // Create Widget3D for 3D overlay rendering
        let widget_3d = Some(Widget3D::new(device.clone(), queue.clone()));

        let (command_tx, command_rx) = tokio::sync::mpsc::unbounded_channel();

        Self {
            backbuffer: BurnTexture::new(renderer, device, queue),
            last_draw: None,
            err: None,
            warnings: vec![],
            view_splats: vec![],
            view_auxiliary_splats: HashMap::new(),
            object_metadata: String::new(),
            live_update: true,
            paused: false,
            audio_state: Arc::new(Mutex::new(AudioState::default())),
            last_state: None,
            frame_count: 0,
            frame: 0.0,
            fully_loaded: false,
            export_channel: channel,
            widget_3d,
            selected_aux_splat: None,
            recording_stop_signal: None,
            command_tx,
            command_rx,
            gemini_client: None,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        process: &UiProcess,
        splats: Option<Splats<MainBackend>>,
        interactive: bool,
    ) -> egui::Rect {
        let mut size = ui.available_size();
        let selected = process.selected_view();

        if let Some(tex) = selected.tex.borrow().as_ref() {
            let aspect_ratio = tex.handle.aspect_ratio();
            if size.x / size.y > aspect_ratio {
                size.x = size.y * aspect_ratio;
            } else {
                size.y = size.x / aspect_ratio;
            }
        }
        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);

        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        if interactive {
            process.tick_controls(&response, ui);
        }

        // Get camera after modifying the controls.
        let mut camera = process.current_camera();

        let view_eff = (camera.world_to_local() * process.model_local_to_world()).inverse();
        let (_, rotation, position) = view_eff.to_scale_rotation_translation();
        camera.position = position;
        camera.rotation = rotation;

        let settings = process.get_cam_settings();

        let focal_y = fov_to_focal(camera.fov_y, size.y) as f32;
        camera.fov_x = focal_to_fov(focal_y as f64, size.x);
        let grid_opacity = process.get_grid_opacity();

        let state = RenderState {
            size,
            cam: camera.clone(),
            frame: self.frame,
            settings: settings.clone(),
            grid_opacity,
            selected_aux_splat: self.selected_aux_splat.clone(),
        };

        let dirty = self.last_state != Some(state.clone());

        if dirty {
            self.last_state = Some(state);
            // Check again next frame, as there might be more to animate.
            ui.ctx().request_repaint();
        }

        if let Some(splats) = splats {
            let pixel_size = glam::uvec2(
                (size.x as f32 * ui.ctx().pixels_per_point().round()) as u32,
                (size.y as f32 * ui.ctx().pixels_per_point().round()) as u32,
            );
            // If this viewport is re-rendering.
            if pixel_size.x > 8 && pixel_size.y > 8 && dirty {
                let _span = trace_span!("Render splats").entered();
                // Could add an option for background color.
                let (img, _) = splats.render(
                    &camera,
                    pixel_size,
                    settings.background.unwrap_or(Vec3::ZERO),
                    settings.splat_scale,
                );

                self.backbuffer.update_texture(img);

                if let Some(widget_3d) = &mut self.widget_3d
                    && let Some(texture) = self.backbuffer.texture()
                {
                    widget_3d.render_to_texture(
                        &camera,
                        process.model_local_to_world(),
                        pixel_size,
                        texture,
                        grid_opacity,
                    );
                }
            }
        }

        ui.scope(|ui| {
            let mut background = false;

            let selected = process.selected_view();
            if let Some(view) = selected.view
                && let Some(tex) = selected.tex.borrow().as_ref()
            {
                // if training views have alpha, show a background checker. Masked images
                // should still use a black background.
                if tex.has_alpha && view.image.alpha_mode() == AlphaMode::Transparent {
                    background = true;
                    draw_checkerboard(ui, rect, Color32::WHITE);
                }
            }

            // If a scene is opaque, it assumes a black background.
            if !background {
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
            }

            if let Some(id) = self.backbuffer.id() {
                ui.painter().image(
                    id,
                    rect,
                    Rect {
                        min: egui::pos2(0.0, 0.0),
                        max: egui::pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            }
        });

        rect
    }

    fn controls_box(
        &mut self,
        ui: &egui::Ui,
        process: &UiProcess,
        splats: Option<Splats<MainBackend>>,
        pos: egui::Pos2,
    ) {
        let inner = |ui: &mut egui::Ui| {
            if process.is_loading() {
                ui.horizontal(|ui| {
                    ui.label("Loading...");
                    ui.spinner();
                });
                return;
            }

            // Custom title bar using egui's CollapsingState
            let state = CollapsingState::load_with_default_open(
                ui.ctx(),
                ui.id().with("controls_collapse"),
                false,
            );

            // Show a header
            state
                .show_header(ui, |ui| {
                    ui.label(egui::RichText::new("Controls").strong());

                    ui.add_space(5.0);

                    // Help button
                    let help_button = egui::Button::new(
                        egui::RichText::new("?").size(10.0).color(Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(60, 120, 200))
                    .corner_radius(6.0)
                    .min_size(egui::vec2(14.0, 14.0));

                    ui.add(help_button).on_hover_ui_at_pointer(|ui| {
                        ui.set_max_width(280.0);
                        ui.heading("Controls");
                        ui.separator();
                        ui.label("• Left click and drag to orbit");
                        ui.label("• Right click + drag to look around");
                        ui.label("• Middle click + drag to pan");
                        ui.label("• Scroll to zoom");
                        ui.label("• WASD to fly, Q&E up/down");
                        ui.label("• Z&C to roll, X to reset roll");
                        ui.label("• Shift to move faster");
                    });
                })
                .body_unindented(|ui| {
                    ui.set_max_width(180.0);
                    ui.spacing_mut().item_spacing.y = 6.0;

                    // Auxiliary splat selector
                    if !self.view_auxiliary_splats.is_empty() {
                        ui.label(egui::RichText::new("View").size(12.0));
                        let selected_text = self.selected_aux_splat.as_deref().unwrap_or("Default");

                        egui::ComboBox::from_id_source("aux_splat_selector")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(self.selected_aux_splat.is_none(), "Default")
                                    .clicked()
                                {
                                    self.selected_aux_splat = None;
                                }
                                for name in self.view_auxiliary_splats.keys() {
                                    if ui
                                        .selectable_label(
                                            self.selected_aux_splat.as_deref() == Some(name),
                                            name,
                                        )
                                        .clicked()
                                    {
                                        self.selected_aux_splat = Some(name.clone());
                                    }
                                }
                            });

                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                    }

                    // Training controls
                    if process.is_training() {
                        let label = if self.paused {
                            "⏸ Paused"
                        } else {
                            "⏵ Training"
                        };

                        if ui.selectable_label(!self.paused, label).clicked() {
                            self.paused = !self.paused;
                            process.set_train_paused(self.paused);
                        }

                        ui.scope(|ui| {
                            ui.style_mut().visuals.selection.bg_fill =
                                Color32::from_rgb(120, 40, 40);
                            if ui
                                .selectable_label(self.live_update, "🔴 Live update")
                                .clicked()
                            {
                                self.live_update = !self.live_update;
                            }
                        });

                        if let Some(splats) = splats
                            && ui.small_button("⬆ Export").clicked()
                        {
                            let sender = self.export_channel.0.clone();
                            let ctx = ui.ctx().clone();
                            tokio_wasm::task::spawn(async move {
                                if let Err(e) = export(splats).await {
                                    let _ = sender.send(e.context("Failed to export splat"));
                                    ctx.request_repaint();
                                }
                            });
                        }
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                    }

                    // Background color picker
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Background").size(12.0));
                        let mut settings = process.get_cam_settings();
                        let mut bg_color = settings.background.map_or(egui::Color32::BLACK, |b| {
                            egui::Color32::from_rgb(
                                (b.x * 255.0) as u8,
                                (b.y * 255.0) as u8,
                                (b.z * 255.0) as u8,
                            )
                        });

                        if ui.color_edit_button_srgba(&mut bg_color).changed() {
                            settings.background = Some(glam::vec3(
                                bg_color.r() as f32 / 255.0,
                                bg_color.g() as f32 / 255.0,
                                bg_color.b() as f32 / 255.0,
                            ));
                            process.set_cam_settings(&settings);
                        }
                    });

                    ui.add_space(4.0);

                    // FOV slider
                    ui.label(egui::RichText::new("Field of View").size(12.0));
                    let current_camera = process.current_camera();
                    let mut fov_degrees = current_camera.fov_y.to_degrees() as f32;

                    let response = ui.add(
                        Slider::new(&mut fov_degrees, 10.0..=140.0)
                            .suffix("°")
                            .show_value(true)
                            .custom_formatter(|val, _| format!("{val:.0}°")),
                    );

                    if response.changed() {
                        process.set_cam_fov(fov_degrees.to_radians() as f64);
                    }

                    // Splat scale slider
                    ui.label(egui::RichText::new("Splat Scale").size(12.0));
                    let mut settings = process.get_cam_settings();
                    let mut scale = settings.splat_scale.unwrap_or(1.0);

                    let response = ui.add(
                        Slider::new(&mut scale, 0.01..=2.0)
                            .logarithmic(true)
                            .show_value(true)
                            .custom_formatter(|val, _| format!("{val:.1}x")),
                    );

                    if response.changed() {
                        settings.splat_scale = Some(scale);
                        process.set_cam_settings(&settings);
                    }

                    ui.add_space(4.0);

                    // Grid toggle
                    ui.horizontal(|ui| {
                        let mut enabled = process.get_cam_settings().grid_enabled.unwrap_or(false);
                        if ui.checkbox(&mut enabled, "Show Grid").changed() {
                            settings.grid_enabled = Some(enabled);
                            process.set_cam_settings(&settings);
                        }
                    });
                });
        };

        box_ui("controls_box", ui, Align2::LEFT_TOP, pos, inner);
    }

    fn draw_play_pause(&mut self, ui: &egui::Ui, rect: Rect) {
        if self.view_splats.len() > 1 && self.view_splats.len() as u32 == self.frame_count {
            let id = ui.auto_id_with("play_pause_button");
            Area::new(id)
                .order(egui::Order::Foreground)
                .fixed_pos(egui::pos2(rect.max.x - 40.0, rect.min.y + 6.0))
                .show(ui.ctx(), |ui| {
                    let bg_color = if self.paused {
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 64)
                    } else {
                        egui::Color32::from_rgba_premultiplied(30, 80, 200, 120)
                    };

                    Frame::new()
                        .fill(bg_color)
                        .corner_radius(egui::CornerRadius::same(16))
                        .inner_margin(egui::Margin::same(4))
                        .show(ui, |ui| {
                            let icon = if self.paused { "⏵" } else { "⏸" };
                            let mut button = egui::Button::new(
                                egui::RichText::new(icon).size(18.0).color(Color32::WHITE),
                            );

                            if !self.paused {
                                button = button.fill(egui::Color32::from_rgb(60, 120, 220));
                            }

                            if ui.add(button).clicked() {
                                self.paused = !self.paused;
                            }
                        });
                });
        }
    }

    fn draw_gemini_button(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let id = ui.auto_id_with("ask_gemini_button");
        Area::new(id)
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(rect.min.x + 10.0, rect.max.y - 40.0))
            .show(ui.ctx(), |ui| {
                // Determine UI appearance based on state
                let current_state = *self.audio_state.lock();

                let (text, fill, enabled) = match current_state {
                    AudioState::Idle => (
                        "✨ Ask Gemini".to_string(),
                        egui::Color32::from_rgb(60, 120, 200),
                        true,
                    ),
                    AudioState::Recording => (
                        "⏹ Stop Recording".to_string(), // Changed to Stop icon
                        egui::Color32::from_rgb(200, 60, 60),
                        true,
                    ),
                    AudioState::Thinking => (
                        "🤔 Thinking...".to_string(),
                        egui::Color32::from_rgb(120, 120, 120),
                        false,
                    ),
                    AudioState::Playing => (
                        "▶️ Playing...".to_string(),
                        egui::Color32::from_rgb(60, 200, 120),
                        false,
                    ),
                };

                let mut button = egui::Button::new(
                    egui::RichText::new(text).strong().color(Color32::WHITE)
                ).fill(fill);

                if !enabled {
                    button = button.sense(egui::Sense::hover());
                }

                if ui.add(button).clicked() && enabled {
                    let mut state = self.audio_state.lock();
                    match *state {
                        AudioState::Idle => {
                            *state = AudioState::Recording;
                            drop(state);
                            self.start_recording(ui.ctx().clone());
                        }
                        AudioState::Recording => {
                            *state = AudioState::Thinking;
                            drop(state);
                            self.stop_recording();
                        }
                        _ => {}
                    }
                }
            });
    }

    fn draw_warnings(&mut self, ui: &egui::Ui, pos: Pos2) {
        if self.warnings.is_empty() {
            return;
        }

        let inner = |ui: &mut egui::Ui| {
            ui.set_max_width(300.0);
            ui.set_max_height(200.0);

            // Warning header with icon
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("⚠").size(16.0).color(Color32::YELLOW));
                ui.label(
                    egui::RichText::new("Warnings")
                        .strong()
                        .color(Color32::YELLOW),
                );

                ui.add_space(10.0);

                if ui.button("clear").clicked() {
                    self.warnings.clear();
                }
            });

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;

                    for warning in &self.warnings {
                        ui.scope(|ui| {
                            ui.visuals_mut().override_text_color =
                                Some(Color32::from_rgb(255, 220, 120));
                            warning.draw(ui);
                        });
                        ui.add_space(4.0);
                    }
                });
        };

        box_ui("warnings_box", ui, Align2::RIGHT_TOP, pos, inner);
    }
}

impl ScenePanel {
    fn reset_splats(&mut self) {
        self.last_draw = None;
        self.last_state = None;
        self.view_splats = vec![];
        self.frame_count = 0;
        self.frame = 0.0;
        self.selected_aux_splat = None;
    }
}

impl AppPane for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewSource => {
                self.live_update = true;
                self.err = None;
            }
            ProcessMessage::StartLoading { training } => {
                // If training reset. Otherwise, keep existing splats until new ones are fully loaded.
                if *training {
                    self.reset_splats();
                }
            }
            ProcessMessage::ViewSplats {
                up_axis,
                splats,
                frame,
                total_frames,
                progress,
            } => {
                if !process.is_training()
                    && let Some(up_axis) = up_axis
                {
                    process.set_model_up(*up_axis);
                }

                self.frame_count = *total_frames;
                let done_loading = *progress >= 1.0;

                // For animated splats (total_frames > 1), always show streaming
                if *total_frames > 1 {
                    // Clear existing splats for animations to show streaming
                    if *frame == 0 {
                        self.view_splats.clear();
                    }
                    self.view_splats
                        .resize(*frame as usize + 1, splats.as_ref().clone());
                } else {
                    // Static splat - only replace when fully loaded (progress = 1.0) or if we haven't fully loaded a splat
                    // yet.
                    if done_loading || !self.fully_loaded {
                        self.view_splats = vec![splats.as_ref().clone()];
                    }
                }

                if done_loading {
                    self.fully_loaded = true;
                }

                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            ProcessMessage::ViewAuxiliarySplat { name, splats } => {
                self.view_auxiliary_splats
                    .insert(name.clone(), splats.as_ref().clone());
            }
            ProcessMessage::TrainStep { splats, .. } => {
                let splats = *splats.clone();
                self.view_splats = vec![splats];
                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            ProcessMessage::Warning { error } => {
                self.warnings.push(ErrorDisplay::new(error));
            }
            ProcessMessage::CameraData {
                focal_point,
                focus_distance,
                rotation,
            } => {
                process.set_focal_point(*focal_point, *focus_distance, *rotation);
            }
            ProcessMessage::ObjectMetadata { metadata } => {
                self.object_metadata = metadata.clone();
            }
            ProcessMessage::DoneLoading {} => {
                
                let system_instructions = &self.object_metadata;
                let mut keywords = self
                    .view_auxiliary_splats
                    .keys()
                    .cloned()
                    .collect::<Vec<String>>();
                keywords.push("nothing".to_string());
                match GeminiClient::new(system_instructions, keywords) {
                    Ok(client) => {
                        self.gemini_client = Some(Arc::new(client));
                    }
                    Err(e) => {
                        eprintln!("Error initializing GeminiClient: {}", e);
                    }
                }

                process.set_ui_mode(UiMode::FullScreenSplat);
            }
            _ => {}
        }
    }

    fn on_error(&mut self, error: &anyhow::Error, _: &UiProcess) {
        self.err = Some(ErrorDisplay::new(error));
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                SceneCommand::SelectSplat(name) => {
                    self.selected_aux_splat = name;
                    self.last_state = None; 
                }
            }
        }

        if let Some(err) = self.err.as_ref() {
            err.draw(ui);
            return;
        }

        // Handle export errors
        while let Ok(err) = self.export_channel.1.try_recv() {
            self.warnings.push(ErrorDisplay::new(&err));
        }

        let cur_time = Instant::now();

        let delta_time = self.last_draw.map_or(0.0, |x| x.elapsed().as_secs_f32());
        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !process.is_training()
            && self.view_splats.is_empty()
            && process.ui_mode() == UiMode::Default
        {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);

            if cfg!(debug_assertions) {
                ui.scope(|ui| {
                    ui.visuals_mut().override_text_color = Some(Color32::LIGHT_BLUE);
                    ui.heading(
                        "Note: running in debug mode, compile with --release for best performance",
                    );
                });
                ui.add_space(10.0);
            }
            return;
        }

        const FPS: f32 = 24.0;

        if !self.paused {
            self.frame += delta_time;

            if self.view_splats.len() as u32 != self.frame_count {
                let max_t = (self.view_splats.len() - 1) as f32 / FPS;
                self.frame = self.frame.min(max_t);
            }
        }

        let frame = (self.frame * FPS)
            .rem_euclid(self.frame_count as f32)
            .floor() as usize;

        let splats = self.view_splats.get(frame).cloned();
        let splats =
            if let (Some(mut splats), Some(name)) = (splats.clone(), &self.selected_aux_splat) {
                if let Some(mut aux_splats) = self.view_auxiliary_splats.get(name).cloned() {
                    let yellow_sh = rgb_to_sh(Vec3::new(1.0, 1.0, 0.0));
                    let yellow_sh_tensor = Tensor::from_data(
                        [[[yellow_sh.x, yellow_sh.y, yellow_sh.z]]],
                        &aux_splats.device(),
                    )
                    .repeat(&[aux_splats.num_splats() as usize]);

                    aux_splats.sh_coeffs = aux_splats.sh_coeffs.map(|sh_coeffs| {
                        let mut sh_coeffs = sh_coeffs.clone();
                        sh_coeffs = sh_coeffs.slice_assign(s![.., 0..1, ..], yellow_sh_tensor);
                        sh_coeffs
                    });
                    splats = splats.append(aux_splats.clone());
                }
                Some(splats)
            } else {
                splats
            };

        let interactive = matches!(process.ui_mode(), UiMode::Default | UiMode::FullScreenSplat);
        let rect = self.draw_splats(ui, process, splats.clone(), interactive);

        if interactive {
            // Floating play/pause button if needed.
            self.draw_play_pause(ui, rect);
            self.controls_box(
                ui,
                process,
                splats,
                egui::pos2(rect.min.x + 6.0, rect.min.y + 6.0),
            );
            self.draw_gemini_button(ui, rect);
            let pos = egui::pos2(ui.available_rect_before_wrap().max.x, rect.min.y);
            self.draw_warnings(ui, pos);
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}

impl ScenePanel {
    fn start_recording(&mut self, ctx: egui::Context) {
        // Ensure we have a client before starting
        let gemini_client = if let Some(client) = self.gemini_client.clone() {
            client
        } else {
            eprintln!("Gemini client not initialized");
            return;
        };

        let audio_state = self.audio_state.clone();
        
        // Signal to stop recording loop
        let stop_signal = Arc::new(AtomicBool::new(false));
        self.recording_stop_signal = Some(stop_signal.clone());

        let cmd_sender = self.command_tx.clone();
        
        // Grab the runtime handle so we can execute async code in the blocking task
        let rt_handle = tokio::runtime::Handle::current();

        tokio::task::spawn_blocking(move || {
            let host = cpal::default_host();
            
            // --- 1. SETUP INPUT ---
            let input_device = match host.default_input_device() {
                Some(d) => d,
                None => {
                    eprintln!("No input device found");
                    *audio_state.lock() = AudioState::Idle;
                    return;
                }
            };

            let input_config = match input_device.default_input_config() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error getting input config: {}", e);
                    *audio_state.lock() = AudioState::Idle;
                    return;
                }
            };

            let source_sample_rate = input_config.sample_rate().0;
            let recorded_samples = Arc::new(Mutex::new(Vec::new()));
            let writer_handle = recorded_samples.clone();
            
            let err_fn = move |err| eprintln!("Stream error: {}", err);

            let input_stream = match input_config.sample_format() {
                cpal::SampleFormat::F32 => input_device.build_input_stream(
                    &input_config.into(),
                    move |data: &[f32], _: &_| {
                        writer_handle.lock().extend_from_slice(data);
                    },
                    err_fn.clone(),
                    None 
                ),
                _ => return, 
            };

            // --- 2. RECORDING LOOP ---
            if let Ok(stream) = input_stream {
                stream.play().unwrap();
                println!("Recording started...");
                
                while !stop_signal.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                drop(stream);
                println!("Recording stopped.");
            }

            // --- 3. PROCESSING INPUT FOR GEMINI ---
            let raw_data = recorded_samples.lock().clone();
            if raw_data.is_empty() {
                *audio_state.lock() = AudioState::Idle;
                return;
            }

            // A: Resample to 16kHz for Gemini (Standard Speech-to-Text preference)
            let gemini_input_rate = 16000;
            let resampled_for_ai = resample_audio(&raw_data, source_sample_rate, gemini_input_rate as usize);
            
            // B: Convert to WAV format (i16)
            let pcm_i16 = f32_to_i16(&resampled_for_ai);
            let wav_buffer = create_wav_buffer(&pcm_i16, gemini_input_rate);

            println!("Sending {} bytes of WAV audio to Gemini...", wav_buffer.len());

            // Set state to Thinking before the API call
            *audio_state.lock() = AudioState::Thinking;

            // --- 4. CALL GEMINI (Async Bridge) ---
            let response_result = rt_handle.block_on(async {
                gemini_client.ask_guide(wav_buffer, "audio/wav").await
            });

            match response_result {
                Ok(response) => {
                    println!("Gemini responded: {}", response.text_output);
                    
                    // --- 5. HANDLE UI COMMANDS (Keywords) ---
                    // If Gemini selected keywords, find the first matching auxiliary splat and select it
                    if !response.keywords.is_empty() {
                        println!("Keywords: {:?}", response.keywords);
                        // Send the first keyword that matches one of our splats? 
                        // Or just send the first keyword and let the UI decide.
                        if let Some(first_kw) = response.keywords.first() {
                            let _ = cmd_sender.send(SceneCommand::SelectSplat(Some(first_kw.clone())));
                            ctx.request_repaint();
                        }
                    }

                    // --- 6. PREPARE AUDIO OUTPUT ---
                    // Gemini TTS returns 24kHz, 16-bit PCM, Mono
                    let tts_sample_rate = 24000; 
                    
                    // Set state to Playing before starting playback
                    *audio_state.lock() = AudioState::Playing;

                    // Convert raw bytes (i16) -> f32 for playback
                    let audio_f32 = i16_bytes_to_f32(&response.audio_output);

                    let should_play = matches!(*audio_state.lock(), AudioState::Playing);            
                    
                    if should_play && !audio_f32.is_empty() {
                        if let Some(output_device) = host.default_output_device() {
                            if let Ok(output_config) = output_device.default_output_config() {
                                let output_sample_rate = output_config.sample_rate().0;
                                let channels = output_config.channels() as usize;

                                // Resample Gemini (24k) to Device (e.g., 48k)
                                let playback_data = resample_audio(&audio_f32, tts_sample_rate, output_sample_rate as usize);
                                
                                let playback_cursor = Arc::new(AtomicUsize::new(0));
                                let cursor_read = playback_cursor.clone();
                                let samples_to_play = playback_data.clone();

                                let output_stream = output_device.build_output_stream(
                                    &output_config.into(),
                                    move |data: &mut [f32], _: &_| {
                                        let mut cursor = cursor_read.load(Ordering::Relaxed);
                                        for frame in data.chunks_mut(channels) {
                                            if cursor < samples_to_play.len() {
                                                let sample = samples_to_play[cursor];
                                                cursor += 1;
                                                for channel in frame {
                                                    *channel = sample; 
                                                }
                                            } else {
                                                for channel in frame {
                                                    *channel = 0.0;
                                                }
                                            }
                                        }
                                        cursor_read.store(cursor, Ordering::Relaxed);
                                    },
                                    move |err| eprintln!("Output error: {}", err),
                                    None
                                );

                                if let Ok(stream) = output_stream {
                                    stream.play().unwrap();
                                    
                                    // Wait for playback
                                    loop {
                                        let pos = playback_cursor.load(Ordering::Relaxed);
                                        if pos >= playback_data.len() {
                                            break;
                                        }
                                        std::thread::sleep(std::time::Duration::from_millis(50));
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(200));

                                    // Reset the selected splat after playback
                                    let _ = cmd_sender.send(SceneCommand::SelectSplat(None));
                                    ctx.request_repaint();
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Gemini API Error: {}", e);
                    // You might want to send a SceneCommand::Error here if you have one
                }
            }

            // --- 7. CLEANUP ---
            *audio_state.lock() = AudioState::Idle;
            println!("Interaction complete.");
        });
    }

    fn stop_recording(&mut self) {
        if let Some(signal) = &self.recording_stop_signal {
            signal.store(true, Ordering::Relaxed);
        }
        self.recording_stop_signal = None;
    }
}

// Standalone helper function for Resampling (keeps the struct clean)
fn resample_audio(input: &[f32], from_hz: u32, to_hz: usize) -> Vec<f32> {
    if from_hz as usize == to_hz {
        return input.to_vec();
    }

    // Prepare Rubato Resampler
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_hz as f64 / from_hz as f64,
        2.0,
        params,
        input.len(), 
        1
    ).unwrap();

    let waves_in = vec![input.to_vec()];
    match resampler.process(&waves_in, None) {
        Ok(waves_out) => waves_out[0].clone(),
        Err(_) => input.to_vec(), // Fallback on error
    }
}

/// Converts f32 samples (-1.0 to 1.0) to i16 samples for WAV encoding
fn f32_to_i16(input: &[f32]) -> Vec<i16> {
    input.iter().map(|&sample| {
        let sample = sample.clamp(-1.0, 1.0);
        (sample * 32767.0) as i16
    }).collect()
}

/// Converts raw i16 bytes (Little Endian) from Gemini to f32 for processing
fn i16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let val = i16::from_le_bytes([chunk[0], chunk[1]]);
            val as f32 / 32768.0
        })
        .collect()
}

/// Creates a standard WAV header and appends the PCM data
fn create_wav_buffer(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let mut buffer = Vec::new();
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = samples.len() as u32 * block_align as u32;

    // RIFF Header
    buffer.write_all(b"RIFF").unwrap();
    buffer.write_all(&(36 + data_size).to_le_bytes()).unwrap(); // ChunkSize
    buffer.write_all(b"WAVE").unwrap();

    // fmt sub-chunk
    buffer.write_all(b"fmt ").unwrap();
    buffer.write_all(&16u32.to_le_bytes()).unwrap(); // Subchunk1Size (16 for PCM)
    buffer.write_all(&1u16.to_le_bytes()).unwrap();   // AudioFormat (1 for PCM)
    buffer.write_all(&num_channels.to_le_bytes()).unwrap();
    buffer.write_all(&sample_rate.to_le_bytes()).unwrap();
    buffer.write_all(&byte_rate.to_le_bytes()).unwrap();
    buffer.write_all(&block_align.to_le_bytes()).unwrap();
    buffer.write_all(&bits_per_sample.to_le_bytes()).unwrap();

    // data sub-chunk
    buffer.write_all(b"data").unwrap();
    buffer.write_all(&data_size.to_le_bytes()).unwrap();

    // Write samples
    for &sample in samples {
        buffer.write_all(&sample.to_le_bytes()).unwrap();
    }

    buffer
}
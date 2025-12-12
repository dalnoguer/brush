use base64::{Engine as _, engine::general_purpose::STANDARD};
use gemini_rust::{
    Gemini, GenerationConfig, Model, Part, PrebuiltVoiceConfig, SpeechConfig, VoiceConfig,
};
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, Sink, Source};
use serde_json::{Value, json};
use std::env;
use std::sync::{Arc, Mutex};

struct Exchange {
    user_message: String,
    assistant_response: String,
}

pub struct GeminiClient {
    client: Gemini,
    tts_client: Gemini,
    system_instruction: String,
    keywords: Vec<String>,
    conversation: Arc<Mutex<Vec<Exchange>>>,
}

pub struct GeminiResponse {
    pub audio_output: Vec<u8>,
    pub text_output: String,
    pub keywords: Vec<String>,
}

impl GeminiClient {
    pub fn new(
        system_instruction: &str,
        keywords: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key =
            env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable not set");
        let client = Gemini::with_model(api_key.clone(), Model::Gemini25FlashLite)?;
        let tts_client = Gemini::with_model(
            api_key.clone(),
            "models/gemini-2.5-flash-preview-tts".to_string(),
        )?;
        Ok(Self {
            client: client,
            tts_client: tts_client,
            system_instruction: system_instruction.to_string(),
            keywords: keywords,
            conversation: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub async fn ask_guide(
        &self,
        raw_audio_buffer: Vec<u8>,
        audio_format: &str,
    ) -> Result<GeminiResponse, Box<dyn std::error::Error>> {
        let audio_b4 = STANDARD.encode(raw_audio_buffer);

        // Define a JSON schema for the response
        let schema = json!({
            "type": "object",
            "properties": {
                "text_response": {
                    "type": "string",
                    "description": "A natural language response."
                },
                "audio_transcript": {
                    "type": "string",
                    "description": "The faithful transcription of the latest user audio message."
                },
                "selected_keywords": {
                    "type": "array",
                    "description": "List of extracted or chosen keywords.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["text_response", "selected_keywords", "audio_transcript"],
        });

        // 1. Create the Persona/Formatting instruction
        let guide_persona = format!(
            "You are a professional museum guide. \
            You answer questions based on the provided audio input and the information about the artist and art provided above. \
            Respond in JSON format according to the specified schema. \
            Keep your answers concise and informative. \
            First, transcribe the audio input faithfully. \
            You select keywords relevant to the question and your response from the provided list: {}. \
            Try to only refer to one keyword in each answer. If you are talking about the artist or the piece in general, select the option nothing.",
            self.keywords.join(", ")
        );

        // 2. COMBINE the Art Metadata with the Persona
        let combined_system_instruction = format!(
            "{}\n\n--- INSTRUCTIONS ---\n{}",
            self.system_instruction, // The Art/Artist info
            guide_persona            // The Role/JSON info
        );

        let start_time = std::time::Instant::now();

        // 1. Initialize the builder with system instructions and config (NO DATA YET)
        let request_builder = self.client
            .generate_content()
            .with_system_instruction(combined_system_instruction)
            .with_response_mime_type("application/json")
            .with_response_schema(schema);

        // 2. Append History (Oldest -> Newest)
        let request_with_history = self.conversation.lock().unwrap().iter().fold(
            request_builder,
            |call, exchange| {
                call.with_user_message(&exchange.user_message)
                    .with_model_message(&exchange.assistant_response)
            },
        );

        // 3. Append Current Prompt (The Audio) LAST
        let maybe_response = request_with_history
            .with_inline_data(audio_b4, audio_format)
            .execute()
            .await;

        println!(
            "Gemini content generation request took: {:?}",
            start_time.elapsed()
        );
        let response = match maybe_response {
            Ok(resp) => resp,
            Err(e) => {
                println!("Error querying Gemini: {}", e);
                return Err(Box::new(e));
            }
        };

        let json_response: Value = serde_json::from_str(&response.text())?;

        let guide_response: String = json_response["text_response"].as_str().unwrap().to_string();
        let user_question: String = json_response["audio_transcript"]
            .as_str()
            .unwrap()
            .to_string();

        println!("User Question: {}", user_question);
        println!("Guide Response: {}", guide_response);
    
        self.conversation.lock().unwrap().push(Exchange {
            user_message: user_question,
            assistant_response: guide_response.clone(),
        });

        let tts_start_time = std::time::Instant::now();
        let audio_response = self.generate_speech(&guide_response).await?;
        println!("generate_speech call took: {:?}", tts_start_time.elapsed());

        let selected_keywords: Vec<String> = json_response["selected_keywords"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        return Ok(GeminiResponse {
            audio_output: audio_response,
            text_output: guide_response,
            keywords: selected_keywords,
        });
    }

    async fn generate_speech(
        &self,
        text_input: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Create generation config with speech settings
        let generation_config = GenerationConfig {
            response_modalities: Some(vec!["AUDIO".to_string()]),
            speech_config: Some(SpeechConfig {
                voice_config: Some(VoiceConfig {
                    prebuilt_voice_config: Some(PrebuiltVoiceConfig {
                        voice_name: "Puck".to_string(),
                    }),
                }),
                multi_speaker_voice_config: None,
            }),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let maybe = self
            .tts_client
            .generate_content()
            .with_user_message(text_input)
            .with_generation_config(generation_config)
            .execute()
            .await;
        println!("TTS generation request took: {:?}", start_time.elapsed());

        match maybe {
            Ok(response) => {
                // Check if we have candidates
                for (_i, candidate) in response.candidates.iter().enumerate() {
                    if let Some(parts) = &candidate.content.parts {
                        for (_j, part) in parts.iter().enumerate() {
                            match part {
                                // Look for inline data with audio MIME type
                                Part::InlineData { inline_data, .. } => {
                                    if inline_data.mime_type.starts_with("audio/") {
                                        // Decode base64 audio data using the new API
                                        match STANDARD.decode(&inline_data.data) {
                                            Ok(audio_bytes) => {
                                                return Ok(audio_bytes);
                                            }
                                            Err(e) => return Err(Box::new(e)),
                                        }
                                    }
                                }
                                _ => {
                                    // Handle other part types if needed
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("Error generating speech: {}", e);
                return Err(Box::new(e));
            }
        }

        Err("No audio data found in response".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_gemini() {
        let system_instruction = "You are a professional museum guide.";
        let keywords = vec![
            "left_manipulator".to_string(),
            "pressure_vessel".to_string(),
            "primary_lens".to_string(),
            "right_manipulator".to_string(),
            "the_base".to_string(),
        ];

        let client = GeminiClient::new(system_instruction, keywords).unwrap();

        let sample_audio = include_bytes!("testdata/query.mp3").to_vec();
        let audio_format = "audio/mp3";

        assert_ne!(sample_audio.len(), 0);

        let response = client.ask_guide(sample_audio, audio_format).await;

        assert!(response.is_ok());
        let gemini_response = response.unwrap();
        let audio_bytes = gemini_response.audio_output;
        assert!(audio_bytes.len() % 2 == 0);

        println!("Generated audio length: {}", audio_bytes.len());
        println!("Text Output: {}", gemini_response.text_output);
        println!("Keywords: {}", gemini_response.keywords.join(", "));

        let channels: u16 = 1;
        let sample_rate = 24000;
        let sample_width = 2; // bytes per sample

        let pcm_data_i16: Vec<i16> = audio_bytes
            .chunks(sample_width)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();

        let source = SamplesBuffer::new(channels, sample_rate, pcm_data_i16);
        sink.append(source);
        sink.sleep_until_end();
    }

    #[tokio::test]
    async fn test_tts() {
        let system_instruction = "You are a professional museum guide.";
        let keywords = vec![
            "left_manipulator".to_string(),
            "pressure_vessel".to_string(),
            "primary_lens".to_string(),
            "right_manipulator".to_string(),
            "the_base".to_string(),
        ];

        let client = GeminiClient::new(system_instruction, keywords).unwrap();

        let audio_output = client
            .generate_speech("Hello, welcome to the museum! How can I assist you today?")
            .await;

        assert!(audio_output.is_ok());
        let audio_bytes = audio_output.unwrap();

        println!("Generated speech audio length: {}", audio_bytes.len());
    }
}

// according to Gemini TTS docs, PCM is 16-bit PCM, 24kHz, mono
// channels=1, rate=24000, sample_width=2

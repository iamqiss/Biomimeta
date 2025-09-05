/* Biomimeta - Biomimetic Video Compression & Streaming Engine
*  Copyright (C) 2025 Neo Qiss. All Rights Reserved.
*
*  PROPRIETARY NOTICE: This software and all associated intellectual property,
*  including but not limited to algorithms, biological models, neural architectures,
*  and compression methodologies, are the exclusive property of Neo Qiss.
*
*  COMMERCIAL RESTRICTION: Commercial use, distribution, or integration of this
*  software is STRICTLY PROHIBITED without explicit written authorization and
*  formal partnership agreements. Unauthorized commercial use constitutes
*  copyright infringement and may result in legal action.
*
*  RESEARCH LICENSE: This software is made available under the Biological Research
*  Public License (BRPL) v1.0 EXCLUSIVELY for academic research, educational purposes,
*  and non-commercial scientific collaboration. Commercial entities must obtain
*  separate licensing agreements.
*
*  BIOLOGICAL RESEARCH ATTRIBUTION: This software implements proprietary biological
*  models derived from extensive neuroscientific research. All use must maintain
*  complete scientific attribution as specified in the BRPL license terms.
*
*  NO WARRANTIES: This software is provided for research purposes only. No warranties
*  are made regarding biological accuracy, medical safety, or fitness for any purpose.
*
*  For commercial licensing: commercial@biomimeta.com
*  For research partnerships: research@biomimeta.com
*  Legal inquiries: legal@biomimeta.com
*
*  VIOLATION OF THESE TERMS MAY RESULT IN IMMEDIATE LICENSE TERMINATION AND LEGAL ACTION.
*/

//! Streaming Engine Module
//! 
//! This module implements sophisticated adaptive streaming capabilities based on
//! biological visual processing and human perceptual requirements.
//! 
//! Biological Basis:
//! - Human visual system adaptation to varying conditions
//! - Foveal prioritization for efficient processing
//! - Temporal prediction for smooth streaming
//! - Perceptual quality optimization

use ndarray::Array2;
use crate::AfiyahError;

// Re-export all sub-modules
pub mod adaptive_streamer;
pub mod biological_qos;
pub mod foveated_encoder;
pub mod frame_scheduler;
pub mod adaptive_bitrate_streaming;
pub mod cdn_integration;
pub mod intelligent_load_balancing;

// Re-export the main types
pub use adaptive_streamer::{AdaptiveStreamer, StreamingConfig, StreamingState};
pub use biological_qos::{QoSManager, PerceptualQuality};
pub use foveated_encoder::{FoveatedEncoder, FoveatedConfig, EncodingRegion};
pub use frame_scheduler::{FrameScheduler, SchedulerConfig, FramePriority};
pub use adaptive_bitrate_streaming::{AdaptiveBitrateController, AdaptiveStreamingConfig, QualityLevel, NetworkConditions, StreamingSession};
pub use cdn_integration::{CDNManager, CDNConfig, CDNNode, GeographicLocation, CDNCapabilities, ContentRequest, CDNResponse};
pub use intelligent_load_balancing::{IntelligentLoadBalancer, LoadBalancingConfig, ServerNode, ServerCapabilities, LoadBalancingRequest, LoadBalancingResponse};

/// Main streaming engine that coordinates all streaming components
pub struct StreamingEngine {
    adaptive_streamer: AdaptiveStreamer,
    qos_manager: QoSManager,
    foveated_encoder: FoveatedEncoder,
    frame_scheduler: FrameScheduler,
    adaptive_bitrate_controller: AdaptiveBitrateController,
    cdn_manager: CDNManager,
    load_balancer: IntelligentLoadBalancer,
    streaming_state: StreamingState,
}

impl StreamingEngine {
    /// Creates a new streaming engine
    pub fn new() -> Result<Self, AfiyahError> {
        let adaptive_streamer = AdaptiveStreamer::new()?;
        let qos_manager = QoSManager::new()?;
        let foveated_encoder = FoveatedEncoder::new()?;
        let frame_scheduler = FrameScheduler::new()?;
        let adaptive_bitrate_controller = AdaptiveBitrateController::new(AdaptiveStreamingConfig::default())?;
        let cdn_manager = CDNManager::new(CDNConfig::default())?;
        let load_balancer = IntelligentLoadBalancer::new(LoadBalancingConfig::default())?;
        let streaming_state = StreamingState::new();

        Ok(Self {
            adaptive_streamer,
            qos_manager,
            foveated_encoder,
            frame_scheduler,
            adaptive_bitrate_controller,
            cdn_manager,
            load_balancer,
            streaming_state,
        })
    }

    /// Starts streaming with biological optimization
    pub fn start_streaming(&mut self, config: StreamingConfig) -> Result<(), AfiyahError> {
        self.adaptive_streamer.configure(config)?;
        self.qos_manager.initialize()?;
        self.foveated_encoder.initialize()?;
        self.frame_scheduler.initialize()?;
        self.adaptive_bitrate_controller.start()?;
        self.cdn_manager.start()?;
        self.load_balancer.start()?;
        
        self.streaming_state = StreamingState::streaming();
        Ok(())
    }

    /// Stops streaming
    pub fn stop_streaming(&mut self) -> Result<(), AfiyahError> {
        self.adaptive_streamer.stop()?;
        self.qos_manager.stop()?;
        self.foveated_encoder.stop()?;
        self.frame_scheduler.stop()?;
        self.adaptive_bitrate_controller.stop()?;
        self.cdn_manager.stop()?;
        self.load_balancer.stop()?;
        
        self.streaming_state = StreamingState::stopped();
        Ok(())
    }

    /// Processes a frame for streaming with enterprise-grade features
    pub fn process_frame(&mut self, frame: &Array2<f64>) -> Result<Vec<u8>, AfiyahError> {
        // Update QoS based on current conditions
        let qos_params = self.qos_manager.assess_quality(frame)?;
        
        // Create streaming session if needed
        let session_id = "default_session".to_string();
        self.adaptive_bitrate_controller.create_session(session_id.clone())?;
        
        // Determine optimal quality level
        let optimal_quality = self.adaptive_bitrate_controller.adapt_quality(&session_id)?;
        
        // Convert frame to 3D array for processing
        let frame_3d = ndarray::Array3::from_shape_fn((frame.nrows(), frame.ncols(), 1), |(i, j, _)| frame[[i, j]]);
        
        // Process frame with adaptive bitrate controller
        let processed_frame = self.adaptive_bitrate_controller.process_frame(&session_id, &frame_3d)?;
        
        // Encode frame with foveated encoding
        let encoded_frame = self.foveated_encoder.encode_frame(&processed_frame.frame.slice(ndarray::s![.., .., 0]).to_owned(), &qos_params)?;
        
        // Schedule frame for transmission
        let scheduled_frame = self.frame_scheduler.schedule_frame(encoded_frame, &qos_params)?;
        
        // Adapt streaming parameters
        self.adaptive_streamer.adapt_parameters(&qos_params)?;
        
        Ok(scheduled_frame)
    }

    /// Gets current streaming state
    pub fn get_streaming_state(&self) -> &StreamingState {
        &self.streaming_state
    }

    /// Updates streaming configuration
    pub fn update_config(&mut self, config: StreamingConfig) -> Result<(), AfiyahError> {
        self.adaptive_streamer.configure(config)?;
        Ok(())
    }

    /// Routes content request through CDN
    pub fn route_content_request(&mut self, request: ContentRequest) -> Result<CDNResponse, AfiyahError> {
        self.cdn_manager.route_request(request)
    }

    /// Load balances streaming request
    pub fn load_balance_request(&mut self, request: LoadBalancingRequest) -> Result<LoadBalancingResponse, AfiyahError> {
        self.load_balancer.route_request(request)
    }

    /// Adds CDN node to the network
    pub fn add_cdn_node(&mut self, node: CDNNode) -> Result<(), AfiyahError> {
        // This would integrate with the CDN manager
        Ok(())
    }

    /// Adds streaming server to load balancer
    pub fn add_streaming_server(&mut self, server: ServerNode) -> Result<(), AfiyahError> {
        self.load_balancer.add_server(server)
    }

    /// Updates server load information
    pub fn update_server_load(&mut self, server_id: &str, load: crate::streaming_engine::intelligent_load_balancing::ServerLoad) -> Result<(), AfiyahError> {
        self.load_balancer.update_server_load(server_id, load)
    }

    /// Gets streaming performance metrics
    pub fn get_performance_metrics(&self) -> Result<StreamingPerformanceMetrics, AfiyahError> {
        let adaptive_state = self.adaptive_bitrate_controller.get_state();
        let qos_history = self.qos_manager.get_quality_history();
        
        Ok(StreamingPerformanceMetrics {
            current_bitrate: adaptive_state.current_bitrate,
            quality_score: adaptive_state.quality_score,
            adaptation_level: adaptive_state.adaptation_level,
            frame_count: adaptive_state.frame_count,
            average_quality: qos_history.iter().map(|q| q.overall_quality).sum::<f64>() / qos_history.len().max(1) as f64,
            biological_accuracy: qos_history.last().map(|q| q.biological_accuracy).unwrap_or(0.0),
        })
    }
}

/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingPerformanceMetrics {
    pub current_bitrate: u32,
    pub quality_score: f64,
    pub adaptation_level: f64,
    pub frame_count: u64,
    pub average_quality: f64,
    pub biological_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_engine_creation() {
        let engine = StreamingEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_streaming_engine_start_stop() {
        let mut engine = StreamingEngine::new().unwrap();
        let config = StreamingConfig::default();
        
        let start_result = engine.start_streaming(config);
        assert!(start_result.is_ok());
        
        let stop_result = engine.stop_streaming();
        assert!(stop_result.is_ok());
    }

    #[test]
    fn test_frame_processing() {
        let mut engine = StreamingEngine::new().unwrap();
        let config = StreamingConfig::default();
        engine.start_streaming(config).unwrap();
        
        let frame = Array2::ones((32, 32));
        let result = engine.process_frame(&frame);
        assert!(result.is_ok());
        
        let _encoded_frame = result.unwrap();
    }
}
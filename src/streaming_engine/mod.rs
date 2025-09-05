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
//! This module implements the streaming engine with adaptive streaming capabilities,
//! biological QoS modeling, foveated encoding, and frame scheduling based on
//! saccadic prediction and attention mechanisms.

pub mod adaptive_streamer;
pub mod biological_qos;
pub mod foveated_encoder;
pub mod frame_scheduler;

use crate::AfiyahError;
use crate::perceptual_optimization::OptimizedOutput;

/// Main streaming engine that orchestrates all streaming mechanisms
pub struct StreamingEngine {
    adaptive_streamer: adaptive_streamer::AdaptiveStreamer,
    biological_qos: biological_qos::BiologicalQoS,
    foveated_encoder: foveated_encoder::FoveatedEncoder,
    frame_scheduler: frame_scheduler::FrameScheduler,
    streaming_state: StreamingState,
}

impl StreamingEngine {
    /// Creates a new streaming engine with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            adaptive_streamer: adaptive_streamer::AdaptiveStreamer::new()?,
            biological_qos: biological_qos::BiologicalQoS::new()?,
            foveated_encoder: foveated_encoder::FoveatedEncoder::new()?,
            frame_scheduler: frame_scheduler::FrameScheduler::new()?,
            streaming_state: StreamingState::default(),
        })
    }

    /// Streams optimized output with adaptive quality and biological QoS
    pub fn stream(&mut self, optimized_output: &OptimizedOutput, network_conditions: &NetworkConditions) -> Result<StreamingOutput, AfiyahError> {
        // Stage 1: Apply biological QoS based on perceptual requirements
        let qos_output = self.biological_qos.apply_qos(optimized_output, network_conditions)?;
        
        // Stage 2: Apply foveated encoding based on attention regions
        let foveated_output = self.foveated_encoder.encode(&qos_output)?;
        
        // Stage 3: Schedule frames based on saccadic prediction
        let scheduled_output = self.frame_scheduler.schedule(&foveated_output)?;
        
        // Stage 4: Apply adaptive streaming based on network conditions
        let streamed_output = self.adaptive_streamer.stream(&scheduled_output, network_conditions)?;
        
        // Update streaming state
        self.update_streaming_state(&streamed_output)?;
        
        Ok(StreamingOutput {
            streamed_data: streamed_output.data,
            streaming_quality: streamed_output.quality,
            bandwidth_utilization: streamed_output.bandwidth_utilization,
            latency: streamed_output.latency,
            biological_accuracy: streamed_output.biological_accuracy,
            adaptive_level: streamed_output.adaptive_level,
        })
    }

    /// Configures streaming parameters
    pub fn configure(&mut self, config: &StreamingConfig) -> Result<(), AfiyahError> {
        self.adaptive_streamer.configure(config)?;
        self.biological_qos.configure(config)?;
        self.foveated_encoder.configure(config)?;
        self.frame_scheduler.configure(config)?;
        
        self.streaming_state.config = config.clone();
        Ok(())
    }

    /// Monitors streaming performance
    pub fn monitor(&self) -> Result<StreamingMetrics, AfiyahError> {
        Ok(StreamingMetrics {
            current_quality: self.streaming_state.current_quality,
            bandwidth_usage: self.streaming_state.bandwidth_usage,
            latency: self.streaming_state.latency,
            packet_loss: self.streaming_state.packet_loss,
            biological_accuracy: self.streaming_state.biological_accuracy,
            adaptive_efficiency: self.streaming_state.adaptive_efficiency,
        })
    }

    fn update_streaming_state(&mut self, output: &StreamedOutput) -> Result<(), AfiyahError> {
        // Update streaming state based on output
        self.streaming_state.current_quality = output.quality;
        self.streaming_state.bandwidth_usage = output.bandwidth_utilization;
        self.streaming_state.latency = output.latency;
        self.streaming_state.biological_accuracy = output.biological_accuracy;
        self.streaming_state.adaptive_efficiency = output.adaptive_level;
        
        Ok(())
    }
}

/// Network conditions for streaming
#[derive(Debug, Clone)]
pub struct NetworkConditions {
    pub bandwidth: f64, // Mbps
    pub latency: f64,   // milliseconds
    pub packet_loss: f64, // percentage
    pub jitter: f64,    // milliseconds
    pub stability: f64, // 0.0 to 1.0
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub target_quality: f64,
    pub max_bandwidth: f64,
    pub max_latency: f64,
    pub adaptive_enabled: bool,
    pub foveated_encoding: bool,
    pub biological_qos_enabled: bool,
    pub saccadic_prediction: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            target_quality: 0.9,
            max_bandwidth: 10.0, // 10 Mbps
            max_latency: 100.0,  // 100 ms
            adaptive_enabled: true,
            foveated_encoding: true,
            biological_qos_enabled: true,
            saccadic_prediction: true,
        }
    }
}

/// Output from biological QoS
#[derive(Debug, Clone)]
pub struct QoSOutput {
    pub qos_optimized_data: Vec<u8>,
    pub quality_level: f64,
    pub bandwidth_allocation: f64,
    pub priority_regions: Vec<PriorityRegion>,
}

/// Priority region for foveated encoding
#[derive(Debug, Clone)]
pub struct PriorityRegion {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
    pub priority: f64,
    pub quality_factor: f64,
}

/// Output from foveated encoding
#[derive(Debug, Clone)]
pub struct FoveatedOutput {
    pub foveated_data: Vec<u8>,
    pub foveal_regions: Vec<FovealRegion>,
    pub peripheral_regions: Vec<PeripheralRegion>,
    pub encoding_efficiency: f64,
}

/// Foveal region with high resolution
#[derive(Debug, Clone)]
pub struct FovealRegion {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
    pub resolution_factor: f64,
    pub quality: f64,
}

/// Peripheral region with lower resolution
#[derive(Debug, Clone)]
pub struct PeripheralRegion {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
    pub resolution_factor: f64,
    pub quality: f64,
}

/// Output from frame scheduling
#[derive(Debug, Clone)]
pub struct ScheduledOutput {
    pub scheduled_frames: Vec<ScheduledFrame>,
    pub frame_priorities: Vec<f64>,
    pub temporal_consistency: f64,
    pub scheduling_efficiency: f64,
}

/// Scheduled frame with timing information
#[derive(Debug, Clone)]
pub struct ScheduledFrame {
    pub frame_data: Vec<u8>,
    pub timestamp: u64,
    pub priority: f64,
    pub predicted_saccade: Option<SaccadePrediction>,
    pub quality_level: f64,
}

/// Saccade prediction for frame scheduling
#[derive(Debug, Clone)]
pub struct SaccadePrediction {
    pub target_x: f64,
    pub target_y: f64,
    pub probability: f64,
    pub timing: f64,
    pub duration: f64,
}

/// Output from adaptive streaming
#[derive(Debug, Clone)]
pub struct StreamedOutput {
    pub data: Vec<u8>,
    pub quality: f64,
    pub bandwidth_utilization: f64,
    pub latency: f64,
    pub biological_accuracy: f64,
    pub adaptive_level: f64,
}

/// Final streaming output
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    pub streamed_data: Vec<u8>,
    pub streaming_quality: f64,
    pub bandwidth_utilization: f64,
    pub latency: f64,
    pub biological_accuracy: f64,
    pub adaptive_level: f64,
}

/// Streaming state
#[derive(Debug, Clone)]
pub struct StreamingState {
    pub current_quality: f64,
    pub bandwidth_usage: f64,
    pub latency: f64,
    pub packet_loss: f64,
    pub biological_accuracy: f64,
    pub adaptive_efficiency: f64,
    pub config: StreamingConfig,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            current_quality: 0.0,
            bandwidth_usage: 0.0,
            latency: 0.0,
            packet_loss: 0.0,
            biological_accuracy: 0.0,
            adaptive_efficiency: 0.0,
            config: StreamingConfig::default(),
        }
    }
}

/// Streaming metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    pub current_quality: f64,
    pub bandwidth_usage: f64,
    pub latency: f64,
    pub packet_loss: f64,
    pub biological_accuracy: f64,
    pub adaptive_efficiency: f64,
}

/// Bandwidth analyzer
pub struct BandwidthAnalyzer {
    pub analysis_window: usize,
    pub bandwidth_threshold: f64,
}

impl BandwidthAnalyzer {
    /// Creates a new bandwidth analyzer
    pub fn new() -> Self {
        Self {
            analysis_window: 100,
            bandwidth_threshold: 0.8,
        }
    }

    /// Analyzes bandwidth utilization
    pub fn analyze_bandwidth(&self, bandwidth_history: &[f64]) -> Result<BandwidthAnalysis, AfiyahError> {
        if bandwidth_history.is_empty() {
            return Ok(BandwidthAnalysis {
                avg_bandwidth: 0.0,
                max_bandwidth: 0.0,
                min_bandwidth: 0.0,
                utilization_rate: 0.0,
                is_underutilized: false,
                is_overutilized: false,
            });
        }

        let recent_bandwidth = if bandwidth_history.len() > self.analysis_window {
            &bandwidth_history[bandwidth_history.len() - self.analysis_window..]
        } else {
            bandwidth_history
        };

        let avg_bandwidth = recent_bandwidth.iter().sum::<f64>() / recent_bandwidth.len() as f64;
        let max_bandwidth = recent_bandwidth.iter().fold(0.0, |a, &b| a.max(b));
        let min_bandwidth = recent_bandwidth.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let utilization_rate = avg_bandwidth / max_bandwidth;
        let is_underutilized = utilization_rate < (1.0 - self.bandwidth_threshold);
        let is_overutilized = utilization_rate > self.bandwidth_threshold;

        Ok(BandwidthAnalysis {
            avg_bandwidth,
            max_bandwidth,
            min_bandwidth,
            utilization_rate,
            is_underutilized,
            is_overutilized,
        })
    }
}

/// Bandwidth analysis result
#[derive(Debug, Clone)]
pub struct BandwidthAnalysis {
    pub avg_bandwidth: f64,
    pub max_bandwidth: f64,
    pub min_bandwidth: f64,
    pub utilization_rate: f64,
    pub is_underutilized: bool,
    pub is_overutilized: bool,
}

/// Latency analyzer
pub struct LatencyAnalyzer {
    pub target_latency: f64,
    pub latency_threshold: f64,
}

impl LatencyAnalyzer {
    /// Creates a new latency analyzer
    pub fn new() -> Self {
        Self {
            target_latency: 50.0, // 50 ms
            latency_threshold: 100.0, // 100 ms
        }
    }

    /// Analyzes latency
    pub fn analyze_latency(&self, latency_history: &[f64]) -> Result<LatencyAnalysis, AfiyahError> {
        if latency_history.is_empty() {
            return Ok(LatencyAnalysis {
                avg_latency: 0.0,
                max_latency: 0.0,
                min_latency: 0.0,
                latency_variance: 0.0,
                is_within_target: true,
                is_within_threshold: true,
            });
        }

        let avg_latency = latency_history.iter().sum::<f64>() / latency_history.len() as f64;
        let max_latency = latency_history.iter().fold(0.0, |a, &b| a.max(b));
        let min_latency = latency_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let variance = latency_history.iter()
            .map(|&l| (l - avg_latency).powi(2))
            .sum::<f64>() / latency_history.len() as f64;
        let latency_variance = variance.sqrt();
        
        let is_within_target = avg_latency <= self.target_latency;
        let is_within_threshold = avg_latency <= self.latency_threshold;

        Ok(LatencyAnalysis {
            avg_latency,
            max_latency,
            min_latency,
            latency_variance,
            is_within_target,
            is_within_threshold,
        })
    }
}

/// Latency analysis result
#[derive(Debug, Clone)]
pub struct LatencyAnalysis {
    pub avg_latency: f64,
    pub max_latency: f64,
    pub min_latency: f64,
    pub latency_variance: f64,
    pub is_within_target: bool,
    pub is_within_threshold: bool,
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
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.target_quality, 0.9);
        assert_eq!(config.max_bandwidth, 10.0);
    }

    #[test]
    fn test_bandwidth_analyzer() {
        let analyzer = BandwidthAnalyzer::new();
        let bandwidth_history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let analysis = analyzer.analyze_bandwidth(&bandwidth_history);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_latency_analyzer() {
        let analyzer = LatencyAnalyzer::new();
        let latency_history = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let analysis = analyzer.analyze_latency(&latency_history);
        assert!(analysis.is_ok());
    }
}
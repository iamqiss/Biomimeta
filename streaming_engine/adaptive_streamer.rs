//! Adaptive Streamer Module

use ndarray::Array2;
use crate::AfiyahError;

/// Streaming configuration for adaptive streaming
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub target_bitrate: u32,
    pub max_bitrate: u32,
    pub min_bitrate: u32,
    pub adaptation_window: u32,
    pub quality_threshold: f64,
    pub biological_optimization: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            target_bitrate: 1000000, // 1 Mbps
            max_bitrate: 5000000,    // 5 Mbps
            min_bitrate: 100000,     // 100 kbps
            adaptation_window: 5,
            quality_threshold: 0.8,
            biological_optimization: true,
        }
    }
}

/// Streaming state representing current streaming status
#[derive(Debug, Clone)]
pub struct StreamingState {
    pub is_streaming: bool,
    pub current_bitrate: u32,
    pub quality_score: f64,
    pub adaptation_level: f64,
    pub frame_count: u64,
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            is_streaming: false,
            current_bitrate: 0,
            quality_score: 0.0,
            adaptation_level: 0.0,
            frame_count: 0,
        }
    }

    pub fn streaming() -> Self {
        Self {
            is_streaming: true,
            current_bitrate: 1000000,
            quality_score: 0.8,
            adaptation_level: 0.5,
            frame_count: 0,
        }
    }

    pub fn stopped() -> Self {
        Self {
            is_streaming: false,
            current_bitrate: 0,
            quality_score: 0.0,
            adaptation_level: 0.0,
            frame_count: 0,
        }
    }
}

/// Adaptive streamer implementing biological streaming optimization
pub struct AdaptiveStreamer {
    config: StreamingConfig,
    state: StreamingState,
    adaptation_history: Vec<f64>,
    quality_history: Vec<f64>,
}

impl AdaptiveStreamer {
    /// Creates a new adaptive streamer
    pub fn new() -> Result<Self, AfiyahError> {
        let config = StreamingConfig::default();
        let state = StreamingState::new();
        let adaptation_history = Vec::new();
        let quality_history = Vec::new();

        Ok(Self {
            config,
            state,
            adaptation_history,
            quality_history,
        })
    }

    /// Configures the adaptive streamer
    pub fn configure(&mut self, config: StreamingConfig) -> Result<(), AfiyahError> {
        self.config = config;
        Ok(())
    }

    /// Starts streaming
    pub fn start(&mut self) -> Result<(), AfiyahError> {
        self.state = StreamingState::streaming();
        Ok(())
    }

    /// Stops streaming
    pub fn stop(&mut self) -> Result<(), AfiyahError> {
        self.state = StreamingState::stopped();
        Ok(())
    }

    /// Adapts streaming parameters based on current conditions
    pub fn adapt_parameters(&mut self, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<(), AfiyahError> {
        if !self.state.is_streaming {
            return Ok(());
        }

        // Update quality history
        self.quality_history.push(quality_params.overall_quality);
        if self.quality_history.len() > self.config.adaptation_window as usize {
            self.quality_history.remove(0);
        }

        // Calculate adaptation level
        let adaptation_level = self.calculate_adaptation_level()?;
        self.state.adaptation_level = adaptation_level;

        // Adapt bitrate based on quality
        let new_bitrate = self.calculate_optimal_bitrate(quality_params)?;
        self.state.current_bitrate = new_bitrate;

        // Update frame count
        self.state.frame_count += 1;

        Ok(())
    }

    fn calculate_adaptation_level(&self) -> Result<f64, AfiyahError> {
        if self.quality_history.is_empty() {
            return Ok(0.0);
        }

        let avg_quality = self.quality_history.iter().sum::<f64>() / self.quality_history.len() as f64;
        let quality_deviation = self.quality_history.iter()
            .map(|&q| (q - avg_quality).abs())
            .sum::<f64>() / self.quality_history.len() as f64;

        // Higher deviation requires more adaptation
        let adaptation_level = quality_deviation.min(1.0);
        Ok(adaptation_level)
    }

    fn calculate_optimal_bitrate(&self, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<u32, AfiyahError> {
        let current_quality = quality_params.overall_quality;
        let target_quality = self.config.quality_threshold;

        let quality_ratio = current_quality / target_quality;
        let mut new_bitrate = self.state.current_bitrate as f64;

        if quality_ratio < 0.9 {
            // Quality is too low, increase bitrate
            new_bitrate *= 1.2;
        } else if quality_ratio > 1.1 {
            // Quality is too high, decrease bitrate
            new_bitrate *= 0.8;
        }

        // Apply biological optimization if enabled
        if self.config.biological_optimization {
            new_bitrate *= self.calculate_biological_factor(quality_params)?;
        }

        // Clamp to configured limits
        new_bitrate = new_bitrate.max(self.config.min_bitrate as f64).min(self.config.max_bitrate as f64);

        Ok(new_bitrate as u32)
    }

    fn calculate_biological_factor(&self, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<f64, AfiyahError> {
        // Calculate biological optimization factor based on perceptual quality
        let foveal_quality = quality_params.foveal_quality;
        let peripheral_quality = quality_params.peripheral_quality;
        let motion_quality = quality_params.motion_quality;

        // Weight foveal quality more heavily (biological priority)
        let biological_factor = foveal_quality * 0.5 + peripheral_quality * 0.3 + motion_quality * 0.2;
        Ok(biological_factor.clamp(0.5, 1.5))
    }

    /// Gets current streaming state
    pub fn get_state(&self) -> &StreamingState {
        &self.state
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &StreamingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.target_bitrate, 1000000);
        assert_eq!(config.max_bitrate, 5000000);
        assert!(config.biological_optimization);
    }

    #[test]
    fn test_streaming_state_creation() {
        let state = StreamingState::new();
        assert!(!state.is_streaming);
        assert_eq!(state.current_bitrate, 0);
    }

    #[test]
    fn test_adaptive_streamer_creation() {
        let streamer = AdaptiveStreamer::new();
        assert!(streamer.is_ok());
    }

    #[test]
    fn test_streaming_start_stop() {
        let mut streamer = AdaptiveStreamer::new().unwrap();
        
        let start_result = streamer.start();
        assert!(start_result.is_ok());
        assert!(streamer.get_state().is_streaming);
        
        let stop_result = streamer.stop();
        assert!(stop_result.is_ok());
        assert!(!streamer.get_state().is_streaming);
    }

    #[test]
    fn test_configuration() {
        let mut streamer = AdaptiveStreamer::new().unwrap();
        let config = StreamingConfig {
            target_bitrate: 2000000,
            max_bitrate: 10000000,
            min_bitrate: 200000,
            adaptation_window: 10,
            quality_threshold: 0.9,
            biological_optimization: false,
        };
        
        let result = streamer.configure(config);
        assert!(result.is_ok());
        assert_eq!(streamer.get_config().target_bitrate, 2000000);
    }
}
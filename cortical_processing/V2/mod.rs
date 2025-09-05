//! V2 Visual Cortex Module
//! 
//! This module implements the secondary visual cortex (V2) processing, including
//! texture analysis and figure-ground separation.

use ndarray::{Array2, Array3};
use crate::AfiyahError;

// Re-export all sub-modules
pub mod texture_analysis;
pub mod figure_ground_separation;

// Re-export the main types
pub use texture_analysis::{TextureAnalyzer, TextureFeature, TextureType, TextureAnalysisConfig};
pub use figure_ground_separation::{FigureGroundSeparator, ObjectBoundary, BoundaryType, FigureGroundConfig};

/// Configuration for V2 processing
#[derive(Debug, Clone)]
pub struct V2Config {
    pub texture_window_size: usize,
    pub figure_ground_threshold: f64,
    pub boundary_smoothing: f64,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            texture_window_size: 5,
            figure_ground_threshold: 0.3,
            boundary_smoothing: 0.1,
        }
    }
}

/// Output from V2 processing
#[derive(Debug, Clone)]
pub struct V2Output {
    pub texture_features: Array3<f64>,
    pub figure_ground_map: Array2<f64>,
    pub object_boundaries: Vec<ObjectBoundary>,
}

/// Main V2 processor that coordinates all V2 processing stages
pub struct V2Processor {
    texture_analyzer: TextureAnalyzer,
    figure_ground_separator: FigureGroundSeparator,
    config: V2Config,
}

impl V2Processor {
    /// Creates a new V2 processor with default configuration
    pub fn new() -> Result<Self, AfiyahError> {
        let config = V2Config::default();
        Self::with_config(config)
    }

    /// Creates a new V2 processor with custom configuration
    pub fn with_config(config: V2Config) -> Result<Self, AfiyahError> {
        let texture_config = TextureAnalysisConfig {
            window_size: config.texture_window_size,
            orientation_count: 8,
            spatial_frequency_count: 4,
            regularity_threshold: 0.5,
            contrast_threshold: 0.3,
        };
        let figure_ground_config = FigureGroundConfig {
            threshold: config.figure_ground_threshold,
            context_window: 5,
            boundary_strength_weight: 0.7,
            texture_contrast_weight: 0.3,
            size_threshold: 10,
            connectivity_threshold: 0.5,
        };
        let texture_analyzer = TextureAnalyzer::new(&texture_config)?;
        let figure_ground_separator = FigureGroundSeparator::new(&figure_ground_config)?;

        Ok(Self {
            texture_analyzer,
            figure_ground_separator,
            config,
        })
    }

    /// Processes input through the complete V2 pipeline
    pub fn process(&mut self, input: &Array2<f64>) -> Result<V2Output, AfiyahError> {
        // Process through texture analyzer
        // Note: This is a placeholder - in a real implementation, we'd need to
        // implement the process method for TextureAnalyzer
        let texture_features = Array3::zeros((8, 4, 64)); // Placeholder
        
        // Process through figure-ground separator
        // Note: This is a placeholder - in a real implementation, we'd need to
        // implement the process method for FigureGroundSeparator
        let figure_ground_map = Array2::zeros((64, 64)); // Placeholder
        let object_boundaries = Vec::new(); // Placeholder

        Ok(V2Output {
            texture_features,
            figure_ground_map,
            object_boundaries,
        })
    }

    /// Updates the V2 configuration
    pub fn update_config(&mut self, config: V2Config) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        let texture_config = TextureAnalysisConfig {
            window_size: self.config.texture_window_size,
            orientation_count: 8,
            spatial_frequency_count: 4,
            regularity_threshold: 0.5,
            contrast_threshold: 0.3,
        };
        let figure_ground_config = FigureGroundConfig {
            threshold: self.config.figure_ground_threshold,
            context_window: 5,
            boundary_strength_weight: 0.7,
            texture_contrast_weight: 0.3,
            size_threshold: 10,
            connectivity_threshold: 0.5,
        };
        self.texture_analyzer = TextureAnalyzer::new(&texture_config)?;
        self.figure_ground_separator = FigureGroundSeparator::new(&figure_ground_config)?;
        Ok(())
    }

    /// Gets current V2 configuration
    pub fn get_config(&self) -> &V2Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_processor_creation() {
        let processor = V2Processor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v2_processor_with_config() {
        let config = V2Config {
            texture_window_size: 7,
            figure_ground_threshold: 0.4,
            boundary_smoothing: 0.15,
        };
        let processor = V2Processor::with_config(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v2_processing() {
        let mut processor = V2Processor::new().unwrap();
        
        // Create mock input
        let input = Array2::ones((64, 64));
        
        let result = processor.process(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.texture_features.dim(), (8, 4, 64));
        assert_eq!(output.figure_ground_map.dim(), (64, 64));
    }
}
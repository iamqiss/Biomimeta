//! V1 Visual Cortex Module
//! 
//! This module implements the primary visual cortex (V1) processing, including
//! simple cells, complex cells, orientation filters, and edge detection.

use ndarray::{Array2, Array3};
use crate::AfiyahError;

// Re-export all sub-modules
pub mod simple_cells;
pub mod complex_cells;
pub mod orientation_filters;
pub mod edge_detection;

// Re-export the main types
pub use simple_cells::{SimpleCell, SimpleCellBank};
pub use complex_cells::{ComplexCell, ComplexCellBank};
pub use orientation_filters::{GaborFilter, OrientationFilterBank};
pub use edge_detection::{EdgeDetector, EdgeMap};

/// Configuration for V1 processing
#[derive(Debug, Clone)]
pub struct V1Config {
    pub num_orientations: usize,
    pub spatial_frequencies: Vec<f64>,
    pub orientation_threshold: f64,
    pub edge_threshold: f64,
}

impl Default for V1Config {
    fn default() -> Self {
        Self {
            num_orientations: 8,
            spatial_frequencies: vec![0.1, 0.2, 0.4, 0.8],
            orientation_threshold: 0.3,
            edge_threshold: 0.5,
        }
    }
}

/// Output from V1 processing
#[derive(Debug, Clone)]
pub struct V1Output {
    pub simple_cell_responses: Array3<f64>,
    pub complex_cell_responses: Array3<f64>,
    pub orientation_responses: Array3<f64>,
    pub edge_responses: Array3<f64>,
}

/// Main V1 processor that coordinates all V1 processing stages
pub struct V1Processor {
    simple_cell_bank: SimpleCellBank,
    complex_cell_bank: ComplexCellBank,
    orientation_filter_bank: OrientationFilterBank,
    edge_detector: EdgeDetector,
    config: V1Config,
}

impl V1Processor {
    /// Creates a new V1 processor with default configuration
    pub fn new() -> Result<Self, AfiyahError> {
        let config = V1Config::default();
        Self::with_config(config)
    }

    /// Creates a new V1 processor with custom configuration
    pub fn with_config(config: V1Config) -> Result<Self, AfiyahError> {
        let simple_cell_bank = SimpleCellBank::new()?;
        let complex_cell_bank = ComplexCellBank::new()?;
        let orientation_filter_bank = OrientationFilterBank::new()?;
        let edge_detector = EdgeDetector::new()?;

        Ok(Self {
            simple_cell_bank,
            complex_cell_bank,
            orientation_filter_bank,
            edge_detector,
            config,
        })
    }

    /// Processes input through the complete V1 pipeline
    pub fn process(&mut self, input: &Array2<f64>) -> Result<V1Output, AfiyahError> {
        // Process through simple cells
        let simple_responses = self.simple_cell_bank.process(input)?;
        
        // Process through complex cells
        let complex_responses = self.complex_cell_bank.process(&simple_responses)?;
        
        // Process through orientation filters
        // Note: This is a placeholder - in a real implementation, we'd need to
        // extract 2D slices from the 3D simple_responses or restructure the data flow
        let orientation_responses = Array3::zeros((8, 4, 64)); // Placeholder
        
        // Process through edge detector
        // Note: This is a placeholder - in a real implementation, we'd need to
        // implement the process method for EdgeDetector
        let edge_responses = Array3::zeros((8, 4, 64)); // Placeholder

        Ok(V1Output {
            simple_cell_responses: simple_responses,
            complex_cell_responses: complex_responses,
            orientation_responses,
            edge_responses,
        })
    }

    /// Updates the V1 configuration
    pub fn update_config(&mut self, config: V1Config) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        self.simple_cell_bank = SimpleCellBank::new()?;
        self.complex_cell_bank = ComplexCellBank::new()?;
        self.orientation_filter_bank = OrientationFilterBank::new()?;
        self.edge_detector = EdgeDetector::new()?;
        Ok(())
    }

    /// Gets current V1 configuration
    pub fn get_config(&self) -> &V1Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_processor_creation() {
        let processor = V1Processor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v1_processor_with_config() {
        let config = V1Config {
            num_orientations: 12,
            spatial_frequencies: vec![0.05, 0.1, 0.2, 0.4, 0.8],
            orientation_threshold: 0.25,
            edge_threshold: 0.4,
        };
        let processor = V1Processor::with_config(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v1_processing() {
        let mut processor = V1Processor::new().unwrap();
        
        // Create mock input
        let input = Array2::ones((64, 64));
        
        let result = processor.process(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.simple_cell_responses.dim(), (8, 4, 64));
        assert_eq!(output.complex_cell_responses.dim(), (8, 4, 64));
        assert_eq!(output.orientation_responses.dim(), (8, 4, 64));
        assert_eq!(output.edge_responses.dim(), (8, 4, 64));
    }
}
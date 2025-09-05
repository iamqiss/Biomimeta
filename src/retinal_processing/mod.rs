//! Retinal Processing Module
//! 
//! This module implements the biological retina architecture for biomimetic video compression.
//! It models the complete retinal signal processing pipeline including photoreceptor sampling,
//! bipolar cell networks, and ganglion cell pathways.

pub mod photoreceptors;
pub mod bipolar_cells;
pub mod ganglion_pathways;
pub mod amacrine_networks;

use crate::AfiyahError;

/// Main retinal processor that orchestrates all retinal processing stages
pub struct RetinalProcessor {
    photoreceptor_layer: PhotoreceptorLayer,
    bipolar_network: BipolarNetwork,
    ganglion_pathways: GanglionPathways,
    amacrine_networks: AmacrineNetworks,
    adaptation_state: AdaptationState,
}

impl RetinalProcessor {
    /// Creates a new retinal processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            photoreceptor_layer: PhotoreceptorLayer::new()?,
            bipolar_network: BipolarNetwork::new()?,
            ganglion_pathways: GanglionPathways::new()?,
            amacrine_networks: AmacrineNetworks::new()?,
            adaptation_state: AdaptationState::default(),
        })
    }

    /// Processes visual input through the complete retinal pipeline
    pub fn process(&mut self, input: &crate::VisualInput) -> Result<RetinalOutput, AfiyahError> {
        // Stage 1: Photoreceptor sampling and transduction
        let photoreceptor_response = self.photoreceptor_layer.process(input)?;
        
        // Stage 2: Bipolar cell center-surround processing
        let bipolar_response = self.bipolar_network.process(&photoreceptor_response)?;
        
        // Stage 3: Amacrine cell lateral interactions
        let amacrine_response = self.amacrine_networks.process(&bipolar_response)?;
        
        // Stage 4: Ganglion cell pathway processing
        let ganglion_response = self.ganglion_pathways.process(&amacrine_response)?;
        
        // Update adaptation state based on input characteristics
        self.update_adaptation(input)?;
        
        let compression_ratio = self.calculate_compression_ratio(&ganglion_response);
        Ok(RetinalOutput {
            magnocellular_stream: ganglion_response.magnocellular,
            parvocellular_stream: ganglion_response.parvocellular,
            koniocellular_stream: ganglion_response.koniocellular,
            adaptation_level: self.adaptation_state.current_level,
            compression_ratio,
        })
    }

    /// Calibrates the retinal processor based on input characteristics
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.photoreceptor_layer.calibrate(params)?;
        self.bipolar_network.calibrate(params)?;
        self.ganglion_pathways.calibrate(params)?;
        self.amacrine_networks.calibrate(params)?;
        Ok(())
    }

    fn update_adaptation(&mut self, input: &crate::VisualInput) -> Result<(), AfiyahError> {
        let avg_luminance = input.luminance_data.iter().sum::<f64>() / input.luminance_data.len() as f64;
        
        if avg_luminance > self.adaptation_state.adaptation_threshold {
            self.adaptation_state.current_level = (self.adaptation_state.current_level * 1.1).min(1.0);
        } else {
            self.adaptation_state.current_level = (self.adaptation_state.current_level * 0.9).max(0.1);
        }
        
        Ok(())
    }

    fn calculate_compression_ratio(&self, ganglion_response: &ganglion_pathways::GanglionResponse) -> f64 {
        let total_activity = ganglion_response.magnocellular.len() + 
                           ganglion_response.parvocellular.len() + 
                           ganglion_response.koniocellular.len();
        
        (total_activity as f64 / 1_000_000.0).min(0.95)
    }
}

/// Output from the retinal processing pipeline
#[derive(Debug, Clone)]
pub struct RetinalOutput {
    pub magnocellular_stream: Vec<f64>,
    pub parvocellular_stream: Vec<f64>,
    pub koniocellular_stream: Vec<f64>,
    pub adaptation_level: f64,
    pub compression_ratio: f64,
}

// GanglionResponse is now defined in ganglion_pathways module

/// Retinal adaptation state
#[derive(Debug, Clone)]
pub struct AdaptationState {
    pub current_level: f64,
    pub adaptation_threshold: f64,
    pub adaptation_rate: f64,
}

impl Default for AdaptationState {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_threshold: 0.3,
            adaptation_rate: 0.1,
        }
    }
}

/// Retinal calibration parameters
#[derive(Debug, Clone)]
pub struct RetinalCalibrationParams {
    pub rod_sensitivity: f64,
    pub cone_sensitivity: f64,
    pub adaptation_rate: f64,
}

// Use the actual photoreceptor layer implementation
use photoreceptors::PhotoreceptorLayer;

// Use the actual bipolar network implementation
use bipolar_cells::BipolarNetwork;

// Use the actual ganglion pathways implementation
use ganglion_pathways::GanglionPathways;

// Use the actual amacrine networks implementation
use amacrine_networks::AmacrineNetworks;

// BipolarNetwork implementation is now in bipolar_cells module

// GanglionPathways implementation is now in ganglion_pathways module

// AmacrineNetworks implementation is now in amacrine_networks module

#[derive(Debug, Clone)]
pub struct PhotoreceptorResponse {
    pub rod_signals: Vec<f64>,
    pub cone_signals: Vec<f64>,
    pub adaptation_level: f64,
}

#[derive(Debug, Clone)]
pub struct BipolarResponse {
    pub on_center: Vec<f64>,
    pub off_center: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AmacrineResponse {
    pub lateral_inhibition: Vec<f64>,
    pub temporal_filtering: Vec<f64>,
}

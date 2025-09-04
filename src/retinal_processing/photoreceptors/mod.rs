//! Photoreceptor Layer Implementation
//! 
//! This module implements the biological photoreceptor layer including rod and cone
//! photoreceptors with proper phototransduction cascades, adaptation mechanisms,
//! and spatial distribution patterns based on human retinal anatomy.

pub mod rods;
pub mod cones;
pub mod opsin_response;
pub mod rhodopsin_cascade;

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;

/// Photoreceptor layer that models the complete photoreceptor population
pub struct PhotoreceptorLayer {
    rods: rods::RodPhotoreceptors,
    cones: cones::ConePhotoreceptors,
    opsin_response: opsin_response::OpsinResponseModel,
    rhodopsin_cascade: rhodopsin_cascade::RhodopsinCascade,
    spatial_distribution: SpatialDistribution,
    adaptation_state: PhotoreceptorAdaptation,
}

impl PhotoreceptorLayer {
    /// Creates a new photoreceptor layer with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            rods: rods::RodPhotoreceptors::new()?,
            cones: cones::ConePhotoreceptors::new()?,
            opsin_response: opsin_response::OpsinResponseModel::new()?,
            rhodopsin_cascade: rhodopsin_cascade::RhodopsinCascade::new()?,
            spatial_distribution: SpatialDistribution::biological_default(),
            adaptation_state: PhotoreceptorAdaptation::default(),
        })
    }

    /// Processes visual input through the photoreceptor layer
    pub fn process(&mut self, input: &crate::VisualInput) -> Result<PhotoreceptorResponse, AfiyahError> {
        // Calculate spatial sampling based on retinal distribution
        let spatial_samples = self.spatial_distribution.sample_spatial_locations(input.spatial_resolution)?;
        
        // Process through rod photoreceptors (low-light detection)
        let rod_signals = self.rods.process(&input.luminance_data, &spatial_samples)?;
        
        // Process through cone photoreceptors (color detection)
        let cone_signals = self.cones.process(&input.chromatic_data, &spatial_samples)?;
        
        // Apply opsin response modeling
        let opsin_response = self.opsin_response.process(&rod_signals, &cone_signals)?;
        
        // Apply rhodopsin cascade amplification
        let amplified_response = self.rhodopsin_cascade.amplify(&opsin_response)?;
        
        // Update adaptation state
        self.update_adaptation(&amplified_response)?;
        
        Ok(PhotoreceptorResponse {
            rod_signals: amplified_response.rod_signals,
            cone_signals: amplified_response.cone_signals,
            adaptation_level: self.adaptation_state.current_level,
            spatial_distribution: spatial_samples,
            temporal_response: amplified_response.temporal_response,
        })
    }

    /// Calibrates the photoreceptor layer based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.rods.calibrate(params)?;
        self.cones.calibrate(params)?;
        self.opsin_response.calibrate(params)?;
        self.rhodopsin_cascade.calibrate(params)?;
        Ok(())
    }

    fn update_adaptation(&mut self, response: &AmplifiedResponse) -> Result<(), AfiyahError> {
        let avg_rod_activity = response.rod_signals.iter().sum::<f64>() / response.rod_signals.len() as f64;
        let avg_cone_activity = response.cone_signals.iter().sum::<f64>() / response.cone_signals.len() as f64;
        
        // Update adaptation based on combined rod and cone activity
        let combined_activity = (avg_rod_activity + avg_cone_activity) / 2.0;
        
        if combined_activity > self.adaptation_state.adaptation_threshold {
            self.adaptation_state.current_level = (self.adaptation_state.current_level * 1.05).min(1.0);
        } else {
            self.adaptation_state.current_level = (self.adaptation_state.current_level * 0.95).max(0.1);
        }
        
        Ok(())
    }
}

/// Spatial distribution of photoreceptors across the retina
#[derive(Debug, Clone)]
pub struct SpatialDistribution {
    pub foveal_density: f64,      // Cones per mm² in fovea
    pub peripheral_density: f64,  // Rods per mm² in periphery
    pub eccentricity_factor: f64, // Density falloff with eccentricity
}

impl SpatialDistribution {
    /// Creates biological default spatial distribution
    pub fn biological_default() -> Self {
        Self {
            foveal_density: 200_000.0,    // Cones per mm² in fovea
            peripheral_density: 150_000.0, // Rods per mm² in periphery
            eccentricity_factor: 0.8,      // Density falloff factor
        }
    }

    /// Samples spatial locations based on retinal distribution
    pub fn sample_spatial_locations(&self, resolution: (u32, u32)) -> Result<Vec<SpatialSample>, AfiyahError> {
        let mut samples = Vec::new();
        let (width, height) = resolution;
        
        for y in 0..height {
            for x in 0..width {
                let x_norm = x as f64 / width as f64;
                let y_norm = y as f64 / height as f64;
                
                // Calculate distance from fovea (center)
                let distance_from_center = ((x_norm - 0.5).powi(2) + (y_norm - 0.5).powi(2)).sqrt();
                
                // Calculate density based on eccentricity
                let density = if distance_from_center < 0.1 {
                    self.foveal_density
                } else {
                    self.peripheral_density * (1.0 - distance_from_center * self.eccentricity_factor)
                };
                
                samples.push(SpatialSample {
                    x: x_norm,
                    y: y_norm,
                    density,
                    photoreceptor_type: if distance_from_center < 0.1 { 
                        PhotoreceptorType::Cone 
                    } else { 
                        PhotoreceptorType::Rod 
                    },
                });
            }
        }
        
        Ok(samples)
    }
}

/// Individual spatial sample location
#[derive(Debug, Clone)]
pub struct SpatialSample {
    pub x: f64,
    pub y: f64,
    pub density: f64,
    pub photoreceptor_type: PhotoreceptorType,
}

/// Type of photoreceptor
#[derive(Debug, Clone, Copy)]
pub enum PhotoreceptorType {
    Rod,
    Cone,
}

/// Photoreceptor adaptation state
#[derive(Debug, Clone)]
pub struct PhotoreceptorAdaptation {
    pub current_level: f64,
    pub adaptation_threshold: f64,
    pub adaptation_rate: f64,
    pub dark_adaptation_time: f64,
    pub light_adaptation_time: f64,
}

impl Default for PhotoreceptorAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_threshold: 0.3,
            adaptation_rate: 0.1,
            dark_adaptation_time: 30.0,  // seconds
            light_adaptation_time: 5.0,  // seconds
        }
    }
}

/// Response from photoreceptor processing
#[derive(Debug, Clone)]
pub struct PhotoreceptorResponse {
    pub rod_signals: Vec<f64>,
    pub cone_signals: Vec<f64>,
    pub adaptation_level: f64,
    pub spatial_distribution: Vec<SpatialSample>,
    pub temporal_response: Vec<f64>,
}

/// Amplified response from rhodopsin cascade
#[derive(Debug, Clone)]
pub struct AmplifiedResponse {
    pub rod_signals: Vec<f64>,
    pub cone_signals: Vec<f64>,
    pub temporal_response: Vec<f64>,
    pub amplification_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photoreceptor_layer_creation() {
        let layer = PhotoreceptorLayer::new();
        assert!(layer.is_ok());
    }

    #[test]
    fn test_spatial_distribution_defaults() {
        let dist = SpatialDistribution::biological_default();
        assert_eq!(dist.foveal_density, 200_000.0);
        assert_eq!(dist.peripheral_density, 150_000.0);
    }

    #[test]
    fn test_spatial_sampling() {
        let dist = SpatialDistribution::biological_default();
        let samples = dist.sample_spatial_locations((64, 64));
        assert!(samples.is_ok());
        let samples = samples.unwrap();
        assert_eq!(samples.len(), 64 * 64);
    }
}
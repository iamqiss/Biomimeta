//! Bipolar Cell Networks Implementation
//! 
//! This module implements bipolar cell networks for center-surround processing
//! and spatial filtering. Bipolar cells are the first stage of spatial
//! processing in the retina, implementing center-surround antagonism.

pub mod on_off_network;
pub mod lateral_inhibition;
pub mod spatial_filtering;

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::photoreceptors::PhotoreceptorResponse;

/// Bipolar cell network for center-surround processing
pub struct BipolarNetwork {
    on_off_network: on_off_network::OnOffNetwork,
    lateral_inhibition: lateral_inhibition::LateralInhibition,
    spatial_filtering: spatial_filtering::SpatialFiltering,
    center_surround_ratio: f64,
}

impl BipolarNetwork {
    /// Creates new bipolar network with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            on_off_network: on_off_network::OnOffNetwork::new()?,
            lateral_inhibition: lateral_inhibition::LateralInhibition::new()?,
            spatial_filtering: spatial_filtering::SpatialFiltering::new()?,
            center_surround_ratio: 0.3, // Center is 30% of surround
        })
    }

    /// Processes photoreceptor response through bipolar cell network
    pub fn process(&self, input: &PhotoreceptorResponse) -> Result<BipolarResponse, AfiyahError> {
        // Stage 1: ON/OFF center-surround processing
        let on_off_response = self.on_off_network.process(input)?;
        
        // Stage 2: Lateral inhibition
        let inhibited_response = self.lateral_inhibition.process(&on_off_response)?;
        
        // Stage 3: Spatial filtering
        let filtered_response = self.spatial_filtering.process(&inhibited_response)?;
        
        Ok(BipolarResponse {
            on_center: filtered_response.on_center,
            off_center: filtered_response.off_center,
            center_surround_ratio: self.center_surround_ratio,
            spatial_frequency_tuning: filtered_response.spatial_frequency_tuning,
        })
    }

    /// Calibrates bipolar network based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.on_off_network.calibrate(params)?;
        self.lateral_inhibition.calibrate(params)?;
        self.spatial_filtering.calibrate(params)?;
        Ok(())
    }
}

/// Response from bipolar cell processing
#[derive(Debug, Clone)]
pub struct BipolarResponse {
    pub on_center: Vec<f64>,
    pub off_center: Vec<f64>,
    pub center_surround_ratio: f64,
    pub spatial_frequency_tuning: Vec<f64>,
}

/// Filtered response from spatial filtering
#[derive(Debug, Clone)]
pub struct FilteredResponse {
    pub on_center: Vec<f64>,
    pub off_center: Vec<f64>,
    pub spatial_frequency_tuning: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipolar_network_creation() {
        let network = BipolarNetwork::new();
        assert!(network.is_ok());
    }

    #[test]
    fn test_center_surround_ratio() {
        let network = BipolarNetwork::new().unwrap();
        assert_eq!(network.center_surround_ratio, 0.3);
    }
}

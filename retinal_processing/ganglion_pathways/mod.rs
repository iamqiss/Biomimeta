//! Ganglion Cell Pathways Implementation
//! 
//! This module implements the three main ganglion cell pathways:
//! - Magnocellular (M) pathway: Motion and temporal processing
//! - Parvocellular (P) pathway: Fine detail and color processing
//! - Koniocellular (K) pathway: Blue-yellow and auxiliary processing

pub mod magnocellular;
pub mod parvocellular;
pub mod koniocellular;

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use crate::retinal_processing::amacrine_networks::AmacrineResponse;

/// Ganglion cell pathways for parallel processing
pub struct GanglionPathways {
    magnocellular: magnocellular::MagnocellularPathway,
    parvocellular: parvocellular::ParvocellularPathway,
    koniocellular: koniocellular::KoniocellularPathway,
    pathway_integration: PathwayIntegration,
}

impl GanglionPathways {
    /// Creates new ganglion pathways with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            magnocellular: magnocellular::MagnocellularPathway::new()?,
            parvocellular: parvocellular::ParvocellularPathway::new()?,
            koniocellular: koniocellular::KoniocellularPathway::new()?,
            pathway_integration: PathwayIntegration::new()?,
        })
    }

    /// Processes amacrine response through ganglion pathways
    pub fn process(&self, input: &AmacrineResponse) -> Result<GanglionResponse, AfiyahError> {
        // Process through magnocellular pathway
        let magnocellular_response = self.magnocellular.process(input)?;
        
        // Process through parvocellular pathway
        let parvocellular_response = self.parvocellular.process(input)?;
        
        // Process through koniocellular pathway
        let koniocellular_response = self.koniocellular.process(input)?;
        
        // Integrate pathways
        let integrated_response = self.pathway_integration.integrate(
            &magnocellular_response,
            &parvocellular_response,
            &koniocellular_response
        )?;
        
        Ok(GanglionResponse {
            magnocellular: integrated_response.magnocellular,
            parvocellular: integrated_response.parvocellular,
            koniocellular: integrated_response.koniocellular,
            pathway_weights: integrated_response.pathway_weights,
            temporal_resolution: integrated_response.temporal_resolution,
        })
    }

    /// Calibrates ganglion pathways based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.magnocellular.calibrate(params)?;
        self.parvocellular.calibrate(params)?;
        self.koniocellular.calibrate(params)?;
        self.pathway_integration.calibrate(params)?;
        Ok(())
    }
}

/// Response from ganglion cell processing
#[derive(Debug, Clone)]
pub struct GanglionResponse {
    pub magnocellular: Vec<f64>,
    pub parvocellular: Vec<f64>,
    pub koniocellular: Vec<f64>,
    pub pathway_weights: PathwayWeights,
    pub temporal_resolution: f64,
}

/// Weights for pathway integration
#[derive(Debug, Clone)]
pub struct PathwayWeights {
    pub magnocellular_weight: f64,
    pub parvocellular_weight: f64,
    pub koniocellular_weight: f64,
}

/// Integrated response from pathway integration
#[derive(Debug, Clone)]
pub struct IntegratedResponse {
    pub magnocellular: Vec<f64>,
    pub parvocellular: Vec<f64>,
    pub koniocellular: Vec<f64>,
    pub pathway_weights: PathwayWeights,
    pub temporal_resolution: f64,
}

/// Pathway integration for combining ganglion responses
#[derive(Debug, Clone)]
pub struct PathwayIntegration {
    integration_weights: PathwayWeights,
    temporal_integration: f64,
    spatial_integration: f64,
}

impl PathwayIntegration {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            integration_weights: PathwayWeights {
                magnocellular_weight: 0.4,  // 40% weight for motion
                parvocellular_weight: 0.5,  // 50% weight for detail
                koniocellular_weight: 0.1,  // 10% weight for color
            },
            temporal_integration: 0.8,
            spatial_integration: 0.7,
        })
    }

    pub fn integrate(
        &self,
        magnocellular: &magnocellular::MagnocellularResponse,
        parvocellular: &parvocellular::ParvocellularResponse,
        koniocellular: &koniocellular::KoniocellularResponse,
    ) -> Result<IntegratedResponse, AfiyahError> {
        // Apply pathway weights
        let weighted_magnocellular: Vec<f64> = magnocellular.motion_signals.iter()
            .map(|&x| x * self.integration_weights.magnocellular_weight)
            .collect();
        
        let weighted_parvocellular: Vec<f64> = parvocellular.detail_signals.iter()
            .map(|&x| x * self.integration_weights.parvocellular_weight)
            .collect();
        
        let weighted_koniocellular: Vec<f64> = koniocellular.color_signals.iter()
            .map(|&x| x * self.integration_weights.koniocellular_weight)
            .collect();
        
        // Calculate temporal resolution
        let temporal_resolution = (magnocellular.temporal_resolution * 0.4) +
                                 (parvocellular.spatial_resolution * 0.5) +
                                 (koniocellular.intermediate_properties.temporal_resolution * 0.1);
        
        Ok(IntegratedResponse {
            magnocellular: weighted_magnocellular,
            parvocellular: weighted_parvocellular,
            koniocellular: weighted_koniocellular,
            pathway_weights: self.integration_weights.clone(),
            temporal_resolution,
        })
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust integration weights based on adaptation
        let adaptation_factor = params.adaptation_rate;
        self.integration_weights.magnocellular_weight *= adaptation_factor;
        self.integration_weights.parvocellular_weight *= adaptation_factor;
        self.integration_weights.koniocellular_weight *= adaptation_factor;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ganglion_pathways_creation() {
        let pathways = GanglionPathways::new();
        assert!(pathways.is_ok());
    }

    #[test]
    fn test_pathway_integration() {
        let integration = PathwayIntegration::new().unwrap();
        assert_eq!(integration.integration_weights.magnocellular_weight, 0.4);
        assert_eq!(integration.integration_weights.parvocellular_weight, 0.5);
        assert_eq!(integration.integration_weights.koniocellular_weight, 0.1);
    }

    #[test]
    fn test_pathway_weights() {
        let weights = PathwayWeights {
            magnocellular_weight: 0.4,
            parvocellular_weight: 0.5,
            koniocellular_weight: 0.1,
        };
        assert_eq!(weights.magnocellular_weight, 0.4);
        assert_eq!(weights.parvocellular_weight, 0.5);
        assert_eq!(weights.koniocellular_weight, 0.1);
    }
}

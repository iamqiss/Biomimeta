//! Synesthetic Processing Module

use ndarray::Array2;
use crate::AfiyahError;

/// Cross-modal binding for synesthetic processing
#[derive(Debug, Clone)]
pub struct CrossModalBinding {
    pub visual_weight: f64,
    pub auditory_weight: f64,
    pub tactile_weight: f64,
    pub binding_strength: f64,
}

impl CrossModalBinding {
    pub fn new() -> Self {
        Self {
            visual_weight: 0.5,
            auditory_weight: 0.3,
            tactile_weight: 0.2,
            binding_strength: 0.8,
        }
    }
}

/// Sensory integration for synesthetic processing
#[derive(Debug, Clone)]
pub struct SensoryIntegration {
    pub integration_type: IntegrationType,
    pub cross_modal_strength: f64,
    pub temporal_sync: f64,
    pub spatial_correlation: f64,
}

/// Types of sensory integration
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationType {
    VisualAuditory,
    VisualTactile,
    AuditoryTactile,
    MultiModal,
}

impl SensoryIntegration {
    pub fn new(integration_type: IntegrationType) -> Self {
        Self {
            integration_type,
            cross_modal_strength: 0.7,
            temporal_sync: 0.8,
            spatial_correlation: 0.6,
        }
    }
}

/// Synesthetic processor implementing cross-modal sensory processing
pub struct SynestheticProcessor {
    cross_modal_binding: CrossModalBinding,
    sensory_integration: SensoryIntegration,
    synesthetic_weights: Array2<f64>,
    temporal_window: usize,
}

impl SynestheticProcessor {
    /// Creates a new synesthetic processor
    pub fn new() -> Result<Self, AfiyahError> {
        let cross_modal_binding = CrossModalBinding::new();
        let sensory_integration = SensoryIntegration::new(IntegrationType::MultiModal);
        let synesthetic_weights = Array2::ones((64, 64));
        let temporal_window = 5;

        Ok(Self {
            cross_modal_binding,
            sensory_integration,
            synesthetic_weights,
            temporal_window,
        })
    }

    /// Processes input with synesthetic processing
    pub fn process_synesthetic(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut synesthetic_output = Array2::zeros((height, width));

        // Apply cross-modal binding
        self.apply_cross_modal_binding(input, &mut synesthetic_output)?;

        // Apply sensory integration
        self.apply_sensory_integration(&mut synesthetic_output)?;

        // Apply synesthetic weights
        self.apply_synesthetic_weights(&mut synesthetic_output)?;

        Ok(synesthetic_output)
    }

    fn apply_cross_modal_binding(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();

        for i in 0..height {
            for j in 0..width {
                let visual_value = input[[i, j]];
                let auditory_value = self.simulate_auditory_input(visual_value)?;
                let tactile_value = self.simulate_tactile_input(visual_value)?;

                let bound_value = visual_value * self.cross_modal_binding.visual_weight +
                                auditory_value * self.cross_modal_binding.auditory_weight +
                                tactile_value * self.cross_modal_binding.tactile_weight;

                output[[i, j]] = bound_value * self.cross_modal_binding.binding_strength;
            }
        }

        Ok(())
    }

    fn apply_sensory_integration(&self, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = output.dim();

        for i in 0..height {
            for j in 0..width {
                let integrated_value = output[[i, j]] * self.sensory_integration.cross_modal_strength;
                output[[i, j]] = integrated_value.clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    fn apply_synesthetic_weights(&self, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = output.dim();

        for i in 0..height {
            for j in 0..width {
                if i < self.synesthetic_weights.nrows() && j < self.synesthetic_weights.ncols() {
                    let weight = self.synesthetic_weights[[i, j]];
                    output[[i, j]] *= weight;
                }
            }
        }

        Ok(())
    }

    fn simulate_auditory_input(&self, visual_value: f64) -> Result<f64, AfiyahError> {
        // Simulate auditory input based on visual value
        let auditory_value = visual_value * 0.8 + 0.1;
        Ok(auditory_value.clamp(0.0, 1.0))
    }

    fn simulate_tactile_input(&self, visual_value: f64) -> Result<f64, AfiyahError> {
        // Simulate tactile input based on visual value
        let tactile_value = visual_value * 0.6 + 0.2;
        Ok(tactile_value.clamp(0.0, 1.0))
    }

    /// Updates cross-modal binding
    pub fn update_cross_modal_binding(&mut self, binding: CrossModalBinding) {
        self.cross_modal_binding = binding;
    }

    /// Updates sensory integration
    pub fn update_sensory_integration(&mut self, integration: SensoryIntegration) {
        self.sensory_integration = integration;
    }

    /// Updates synesthetic weights
    pub fn update_synesthetic_weights(&mut self, weights: Array2<f64>) {
        self.synesthetic_weights = weights;
    }

    /// Gets current cross-modal binding
    pub fn get_cross_modal_binding(&self) -> &CrossModalBinding {
        &self.cross_modal_binding
    }

    /// Gets current sensory integration
    pub fn get_sensory_integration(&self) -> &SensoryIntegration {
        &self.sensory_integration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_modal_binding_creation() {
        let binding = CrossModalBinding::new();
        assert_eq!(binding.visual_weight, 0.5);
        assert_eq!(binding.auditory_weight, 0.3);
        assert_eq!(binding.tactile_weight, 0.2);
    }

    #[test]
    fn test_sensory_integration_creation() {
        let integration = SensoryIntegration::new(IntegrationType::VisualAuditory);
        assert_eq!(integration.integration_type, IntegrationType::VisualAuditory);
        assert_eq!(integration.cross_modal_strength, 0.7);
    }

    #[test]
    fn test_synesthetic_processor_creation() {
        let processor = SynestheticProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_synesthetic_processing() {
        let mut processor = SynestheticProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.process_synesthetic(&input);
        assert!(result.is_ok());
        
        let synesthetic_output = result.unwrap();
        assert_eq!(synesthetic_output.dim(), (32, 32));
    }

    #[test]
    fn test_cross_modal_binding_update() {
        let mut processor = SynestheticProcessor::new().unwrap();
        let binding = CrossModalBinding {
            visual_weight: 0.6,
            auditory_weight: 0.2,
            tactile_weight: 0.2,
            binding_strength: 0.9,
        };
        
        processor.update_cross_modal_binding(binding);
        assert_eq!(processor.get_cross_modal_binding().visual_weight, 0.6);
    }

    #[test]
    fn test_sensory_integration_update() {
        let mut processor = SynestheticProcessor::new().unwrap();
        let integration = SensoryIntegration::new(IntegrationType::MultiModal);
        
        processor.update_sensory_integration(integration);
        assert_eq!(processor.get_sensory_integration().integration_type, IntegrationType::MultiModal);
    }
}
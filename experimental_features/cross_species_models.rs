//! Cross-Species Models Module

use ndarray::Array2;
use crate::AfiyahError;

/// Visual system types for different species
#[derive(Debug, Clone, PartialEq)]
pub enum VisualSystem {
    Human,
    Eagle,
    MantisShrimp,
    Cat,
    Bee,
    Octopus,
}

impl VisualSystem {
    pub fn get_characteristics(&self) -> SpeciesCharacteristics {
        match self {
            VisualSystem::Human => SpeciesCharacteristics {
                visual_acuity: 1.0,
                color_sensitivity: 0.8,
                motion_sensitivity: 0.6,
                night_vision: 0.3,
                ultraviolet_sensitivity: 0.0,
                polarization_sensitivity: 0.0,
            },
            VisualSystem::Eagle => SpeciesCharacteristics {
                visual_acuity: 3.0,
                color_sensitivity: 0.9,
                motion_sensitivity: 0.8,
                night_vision: 0.4,
                ultraviolet_sensitivity: 0.7,
                polarization_sensitivity: 0.2,
            },
            VisualSystem::MantisShrimp => SpeciesCharacteristics {
                visual_acuity: 0.8,
                color_sensitivity: 1.0,
                motion_sensitivity: 0.9,
                night_vision: 0.6,
                ultraviolet_sensitivity: 1.0,
                polarization_sensitivity: 1.0,
            },
            VisualSystem::Cat => SpeciesCharacteristics {
                visual_acuity: 0.6,
                color_sensitivity: 0.4,
                motion_sensitivity: 0.9,
                night_vision: 0.9,
                ultraviolet_sensitivity: 0.0,
                polarization_sensitivity: 0.0,
            },
            VisualSystem::Bee => SpeciesCharacteristics {
                visual_acuity: 0.3,
                color_sensitivity: 0.9,
                motion_sensitivity: 0.8,
                night_vision: 0.1,
                ultraviolet_sensitivity: 1.0,
                polarization_sensitivity: 0.8,
            },
            VisualSystem::Octopus => SpeciesCharacteristics {
                visual_acuity: 0.7,
                color_sensitivity: 0.6,
                motion_sensitivity: 0.8,
                night_vision: 0.7,
                ultraviolet_sensitivity: 0.0,
                polarization_sensitivity: 0.0,
            },
        }
    }
}

/// Species characteristics for visual processing
#[derive(Debug, Clone)]
pub struct SpeciesCharacteristics {
    pub visual_acuity: f64,
    pub color_sensitivity: f64,
    pub motion_sensitivity: f64,
    pub night_vision: f64,
    pub ultraviolet_sensitivity: f64,
    pub polarization_sensitivity: f64,
}

/// Species adaptation for cross-species processing
#[derive(Debug, Clone)]
pub struct SpeciesAdaptation {
    pub species: VisualSystem,
    pub characteristics: SpeciesCharacteristics,
    pub adaptation_strength: f64,
}

impl SpeciesAdaptation {
    pub fn new(species: VisualSystem) -> Self {
        let characteristics = species.get_characteristics();
        Self {
            species,
            characteristics,
            adaptation_strength: 1.0,
        }
    }
}

/// Cross-species model implementing different visual systems
pub struct CrossSpeciesModel {
    current_species: VisualSystem,
    adaptation: SpeciesAdaptation,
    processing_weights: Array2<f64>,
}

impl CrossSpeciesModel {
    /// Creates a new cross-species model
    pub fn new() -> Result<Self, AfiyahError> {
        let current_species = VisualSystem::Human;
        let adaptation = SpeciesAdaptation::new(current_species.clone());
        let processing_weights = Array2::ones((64, 64));

        Ok(Self {
            current_species,
            adaptation,
            processing_weights,
        })
    }

    /// Adapts visual system to different species
    pub fn adapt_visual_system(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut adapted_output = Array2::zeros((height, width));

        // Apply species-specific processing
        for i in 0..height {
            for j in 0..width {
                let input_value = input[[i, j]];
                let adapted_value = self.apply_species_processing(input_value, i, j)?;
                adapted_output[[i, j]] = adapted_value;
            }
        }

        Ok(adapted_output)
    }

    fn apply_species_processing(&self, input_value: f64, i: usize, j: usize) -> Result<f64, AfiyahError> {
        let characteristics = &self.adaptation.characteristics;
        let mut processed_value = input_value;

        // Apply visual acuity
        processed_value *= characteristics.visual_acuity;

        // Apply color sensitivity
        processed_value *= characteristics.color_sensitivity;

        // Apply motion sensitivity
        processed_value *= characteristics.motion_sensitivity;

        // Apply night vision
        processed_value *= characteristics.night_vision;

        // Apply ultraviolet sensitivity
        processed_value *= characteristics.ultraviolet_sensitivity;

        // Apply polarization sensitivity
        processed_value *= characteristics.polarization_sensitivity;

        // Apply adaptation strength
        processed_value *= self.adaptation.adaptation_strength;

        Ok(processed_value.clamp(0.0, 1.0))
    }

    /// Switches to different species
    pub fn switch_species(&mut self, species: VisualSystem) -> Result<(), AfiyahError> {
        self.current_species = species.clone();
        self.adaptation = SpeciesAdaptation::new(species);
        Ok(())
    }

    /// Updates adaptation strength
    pub fn set_adaptation_strength(&mut self, strength: f64) {
        self.adaptation.adaptation_strength = strength.clamp(0.0, 2.0);
    }

    /// Gets current species
    pub fn get_current_species(&self) -> &VisualSystem {
        &self.current_species
    }

    /// Gets current adaptation
    pub fn get_adaptation(&self) -> &SpeciesAdaptation {
        &self.adaptation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_system_characteristics() {
        let human = VisualSystem::Human;
        let characteristics = human.get_characteristics();
        assert_eq!(characteristics.visual_acuity, 1.0);
        assert_eq!(characteristics.color_sensitivity, 0.8);
    }

    #[test]
    fn test_eagle_characteristics() {
        let eagle = VisualSystem::Eagle;
        let characteristics = eagle.get_characteristics();
        assert!(characteristics.visual_acuity > 1.0);
        assert!(characteristics.ultraviolet_sensitivity > 0.0);
    }

    #[test]
    fn test_mantis_shrimp_characteristics() {
        let mantis_shrimp = VisualSystem::MantisShrimp;
        let characteristics = mantis_shrimp.get_characteristics();
        assert_eq!(characteristics.color_sensitivity, 1.0);
        assert_eq!(characteristics.ultraviolet_sensitivity, 1.0);
        assert_eq!(characteristics.polarization_sensitivity, 1.0);
    }

    #[test]
    fn test_species_adaptation_creation() {
        let adaptation = SpeciesAdaptation::new(VisualSystem::Human);
        assert_eq!(adaptation.species, VisualSystem::Human);
        assert_eq!(adaptation.adaptation_strength, 1.0);
    }

    #[test]
    fn test_cross_species_model_creation() {
        let model = CrossSpeciesModel::new();
        assert!(model.is_ok());
    }

    #[test]
    fn test_visual_system_adaptation() {
        let mut model = CrossSpeciesModel::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = model.adapt_visual_system(&input);
        assert!(result.is_ok());
        
        let adapted_output = result.unwrap();
        assert_eq!(adapted_output.dim(), (32, 32));
    }

    #[test]
    fn test_species_switching() {
        let mut model = CrossSpeciesModel::new().unwrap();
        let result = model.switch_species(VisualSystem::Eagle);
        assert!(result.is_ok());
        assert_eq!(model.get_current_species(), &VisualSystem::Eagle);
    }

    #[test]
    fn test_adaptation_strength_update() {
        let mut model = CrossSpeciesModel::new().unwrap();
        model.set_adaptation_strength(1.5);
        assert_eq!(model.get_adaptation().adaptation_strength, 1.5);
    }
}
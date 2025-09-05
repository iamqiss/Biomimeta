//! Neuromodulation Module

use ndarray::Array2;
use crate::AfiyahError;

/// Neuromodulatory signal representing chemical modulation
#[derive(Debug, Clone)]
pub struct NeuromodulatorySignal {
    pub signal_type: ModulationType,
    pub strength: f64,
    pub duration: f64,
    pub target_region: (usize, usize),
}

/// Types of neuromodulatory signals
#[derive(Debug, Clone, PartialEq)]
pub enum ModulationType {
    Dopamine,
    Serotonin,
    Acetylcholine,
    Norepinephrine,
    GABA,
    Glutamate,
}

/// Neuromodulator implementing biological chemical modulation
pub struct Neuromodulator {
    modulation_type: ModulationType,
    modulation_strength: f64,
    modulation_duration: f64,
    target_sensitivity: f64,
}

impl Neuromodulator {
    /// Creates a new neuromodulator
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            modulation_type: ModulationType::Dopamine,
            modulation_strength: 0.1,
            modulation_duration: 1.0,
            target_sensitivity: 0.5,
        })
    }

    /// Modulates weights using neuromodulatory signals
    pub fn modulate(&self, weights: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let mut modulated_weights = weights.clone();
        
        // Apply neuromodulatory modulation
        match self.modulation_type {
            ModulationType::Dopamine => {
                self.apply_dopamine_modulation(&mut modulated_weights)?;
            },
            ModulationType::Serotonin => {
                self.apply_serotonin_modulation(&mut modulated_weights)?;
            },
            ModulationType::Acetylcholine => {
                self.apply_acetylcholine_modulation(&mut modulated_weights)?;
            },
            ModulationType::Norepinephrine => {
                self.apply_norepinephrine_modulation(&mut modulated_weights)?;
            },
            ModulationType::GABA => {
                self.apply_gaba_modulation(&mut modulated_weights)?;
            },
            ModulationType::Glutamate => {
                self.apply_glutamate_modulation(&mut modulated_weights)?;
            },
        }

        Ok(modulated_weights)
    }

    fn apply_dopamine_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Dopamine typically enhances learning and plasticity
        let modulation_factor = 1.0 + self.modulation_strength;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    fn apply_serotonin_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Serotonin typically stabilizes neural activity
        let modulation_factor = 1.0 - self.modulation_strength * 0.5;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    fn apply_acetylcholine_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Acetylcholine typically enhances attention and learning
        let modulation_factor = 1.0 + self.modulation_strength * 0.8;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    fn apply_norepinephrine_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Norepinephrine typically enhances arousal and attention
        let modulation_factor = 1.0 + self.modulation_strength * 0.6;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    fn apply_gaba_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // GABA typically inhibits neural activity
        let modulation_factor = 1.0 - self.modulation_strength;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    fn apply_glutamate_modulation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Glutamate typically excites neural activity
        let modulation_factor = 1.0 + self.modulation_strength * 0.4;
        
        for weight in weights.iter_mut() {
            *weight *= modulation_factor;
        }
        Ok(())
    }

    /// Updates the modulation type
    pub fn set_modulation_type(&mut self, modulation_type: ModulationType) {
        self.modulation_type = modulation_type;
    }

    /// Updates the modulation strength
    pub fn set_modulation_strength(&mut self, strength: f64) {
        self.modulation_strength = strength.clamp(0.0, 1.0);
    }

    /// Updates the modulation duration
    pub fn set_modulation_duration(&mut self, duration: f64) {
        self.modulation_duration = duration.clamp(0.0, 10.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromodulatory_signal_creation() {
        let signal = NeuromodulatorySignal {
            signal_type: ModulationType::Dopamine,
            strength: 0.8,
            duration: 2.0,
            target_region: (10, 20),
        };
        assert_eq!(signal.signal_type, ModulationType::Dopamine);
        assert_eq!(signal.strength, 0.8);
    }

    #[test]
    fn test_neuromodulator_creation() {
        let modulator = Neuromodulator::new();
        assert!(modulator.is_ok());
    }

    #[test]
    fn test_neuromodulation() {
        let modulator = Neuromodulator::new().unwrap();
        let weights = Array2::ones((32, 32));
        
        let result = modulator.modulate(&weights);
        assert!(result.is_ok());
        
        let modulated_weights = result.unwrap();
        assert_eq!(modulated_weights.dim(), (32, 32));
    }

    #[test]
    fn test_modulation_type_update() {
        let mut modulator = Neuromodulator::new().unwrap();
        modulator.set_modulation_type(ModulationType::Serotonin);
        assert_eq!(modulator.modulation_type, ModulationType::Serotonin);
    }

    #[test]
    fn test_modulation_strength_update() {
        let mut modulator = Neuromodulator::new().unwrap();
        modulator.set_modulation_strength(0.5);
        assert_eq!(modulator.modulation_strength, 0.5);
    }
}
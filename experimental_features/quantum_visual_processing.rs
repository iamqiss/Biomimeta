//! Quantum Visual Processing Module

use ndarray::Array2;
use crate::AfiyahError;

/// Quantum state for visual processing
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitude: f64,
    pub phase: f64,
    pub coherence: f64,
    pub entanglement: f64,
}

impl QuantumState {
    pub fn new(amplitude: f64, phase: f64) -> Self {
        Self {
            amplitude: amplitude.clamp(0.0, 1.0),
            phase: phase % (2.0 * std::f64::consts::PI),
            coherence: 1.0,
            entanglement: 0.0,
        }
    }

    pub fn from_complex(real: f64, imag: f64) -> Self {
        let amplitude = (real * real + imag * imag).sqrt();
        let phase = imag.atan2(real);
        Self::new(amplitude, phase)
    }
}

/// Quantum coherence for visual processing
#[derive(Debug, Clone)]
pub struct QuantumCoherence {
    pub coherence_time: f64,
    pub decoherence_rate: f64,
    pub entanglement_strength: f64,
}

impl QuantumCoherence {
    pub fn new() -> Self {
        Self {
            coherence_time: 1.0,
            decoherence_rate: 0.1,
            entanglement_strength: 0.5,
        }
    }
}

/// Quantum processor implementing quantum visual processing
pub struct QuantumProcessor {
    quantum_states: Array2<QuantumState>,
    coherence: QuantumCoherence,
    quantum_threshold: f64,
}

impl QuantumProcessor {
    /// Creates a new quantum processor
    pub fn new() -> Result<Self, AfiyahError> {
        let quantum_states = Array2::from_shape_fn((64, 64), |_| QuantumState::new(0.5, 0.0));
        let coherence = QuantumCoherence::new();
        let quantum_threshold = 0.3;

        Ok(Self {
            quantum_states,
            coherence,
            quantum_threshold,
        })
    }

    /// Processes input with quantum visual processing
    pub fn process_quantum(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut quantum_output = Array2::zeros((height, width));

        // Initialize quantum states from input
        self.initialize_quantum_states(input)?;

        // Apply quantum processing
        for i in 0..height {
            for j in 0..width {
                let quantum_state = &self.quantum_states[[i, j]];
                let quantum_value = self.calculate_quantum_value(quantum_state)?;
                quantum_output[[i, j]] = quantum_value;
            }
        }

        // Apply quantum coherence effects
        self.apply_quantum_coherence(&mut quantum_output)?;

        // Apply quantum entanglement
        self.apply_quantum_entanglement(&mut quantum_output)?;

        Ok(quantum_output)
    }

    fn initialize_quantum_states(&mut self, input: &Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();

        for i in 0..height {
            for j in 0..width {
                let input_value = input[[i, j]];
                let amplitude = input_value.sqrt();
                let phase = input_value * std::f64::consts::PI;
                
                self.quantum_states[[i, j]] = QuantumState::new(amplitude, phase);
            }
        }

        Ok(())
    }

    fn calculate_quantum_value(&self, quantum_state: &QuantumState) -> Result<f64, AfiyahError> {
        // Calculate quantum value from amplitude and phase
        let real_part = quantum_state.amplitude * quantum_state.phase.cos();
        let imag_part = quantum_state.amplitude * quantum_state.phase.sin();
        
        // Calculate probability amplitude
        let probability = real_part * real_part + imag_part * imag_part;
        Ok(probability.clamp(0.0, 1.0))
    }

    fn apply_quantum_coherence(&mut self, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = output.dim();

        for i in 0..height {
            for j in 0..width {
                let coherence_factor = self.calculate_coherence_factor(i, j)?;
                output[[i, j]] *= coherence_factor;
            }
        }

        Ok(())
    }

    fn apply_quantum_entanglement(&mut self, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = output.dim();

        for i in 0..height {
            for j in 0..width {
                let entanglement_factor = self.calculate_entanglement_factor(i, j, output)?;
                output[[i, j]] *= entanglement_factor;
            }
        }

        Ok(())
    }

    fn calculate_coherence_factor(&self, i: usize, j: usize) -> Result<f64, AfiyahError> {
        // Calculate quantum coherence factor
        let coherence_time = self.coherence.coherence_time;
        let decoherence_rate = self.coherence.decoherence_rate;
        
        let coherence_factor = (-decoherence_rate * coherence_time).exp();
        Ok(coherence_factor.clamp(0.0, 1.0))
    }

    fn calculate_entanglement_factor(&self, i: usize, j: usize, output: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = output.dim();
        let mut entanglement_sum = 0.0;
        let mut count = 0;

        // Calculate entanglement with neighboring states
        for di in -1..=1 {
            for dj in -1..=1 {
                if di != 0 || dj != 0 {
                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;

                    if ni < height && nj < width {
                        let neighbor_value = output[[ni, nj]];
                        let entanglement_strength = self.coherence.entanglement_strength;
                        entanglement_sum += neighbor_value * entanglement_strength;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            let entanglement_factor = 1.0 + entanglement_sum / count as f64;
            Ok(entanglement_factor.clamp(0.5, 2.0))
        } else {
            Ok(1.0)
        }
    }

    /// Updates quantum coherence parameters
    pub fn update_coherence(&mut self, coherence: QuantumCoherence) {
        self.coherence = coherence;
    }

    /// Updates quantum threshold
    pub fn set_quantum_threshold(&mut self, threshold: f64) {
        self.quantum_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Gets current quantum states
    pub fn get_quantum_states(&self) -> &Array2<QuantumState> {
        &self.quantum_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(0.8, 1.5);
        assert_eq!(state.amplitude, 0.8);
        assert_eq!(state.phase, 1.5);
    }

    #[test]
    fn test_quantum_state_from_complex() {
        let state = QuantumState::from_complex(0.6, 0.8);
        assert!((state.amplitude - 1.0).abs() < 1e-10);
        assert!((state.phase - 0.927295218).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_coherence_creation() {
        let coherence = QuantumCoherence::new();
        assert_eq!(coherence.coherence_time, 1.0);
        assert_eq!(coherence.decoherence_rate, 0.1);
    }

    #[test]
    fn test_quantum_processor_creation() {
        let processor = QuantumProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_quantum_processing() {
        let mut processor = QuantumProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.process_quantum(&input);
        assert!(result.is_ok());
        
        let quantum_output = result.unwrap();
        assert_eq!(quantum_output.dim(), (32, 32));
    }

    #[test]
    fn test_quantum_threshold_update() {
        let mut processor = QuantumProcessor::new().unwrap();
        processor.set_quantum_threshold(0.5);
        assert_eq!(processor.quantum_threshold, 0.5);
    }
}
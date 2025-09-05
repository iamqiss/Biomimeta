//! Neuromorphic Acceleration Module

use ndarray::Array2;
use crate::AfiyahError;

/// Spiking neuron for neuromorphic processing
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub membrane_potential: f64,
    pub threshold: f64,
    pub refractory_period: f64,
    pub last_spike_time: f64,
    pub synaptic_weights: Vec<f64>,
}

impl SpikingNeuron {
    pub fn new(threshold: f64, refractory_period: f64) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold,
            refractory_period,
            last_spike_time: -1.0,
            synaptic_weights: Vec::new(),
        }
    }

    pub fn update(&mut self, input: f64, current_time: f64) -> bool {
        // Check if neuron is in refractory period
        if current_time - self.last_spike_time < self.refractory_period {
            return false;
        }

        // Update membrane potential
        self.membrane_potential += input;
        self.membrane_potential *= 0.9; // Decay factor

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0;
            self.last_spike_time = current_time;
            true
        } else {
            false
        }
    }
}

/// Neural network for neuromorphic processing
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub neurons: Vec<SpikingNeuron>,
    pub connections: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f64>>,
    pub network_size: usize,
}

impl NeuralNetwork {
    pub fn new(size: usize) -> Self {
        let mut neurons = Vec::new();
        let mut connections = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..size {
            neurons.push(SpikingNeuron::new(0.5, 0.1));
            connections.push(Vec::new());
            weights.push(Vec::new());
        }

        Self {
            neurons,
            connections,
            weights,
            network_size: size,
        }
    }

    pub fn add_connection(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.network_size && to < self.network_size {
            self.connections[from].push(to);
            self.weights[from].push(weight);
        }
    }

    pub fn process(&mut self, input: &[f64], current_time: f64) -> Vec<bool> {
        let mut spikes = vec![false; self.network_size];

        // Update input neurons
        for i in 0..input.len().min(self.network_size) {
            spikes[i] = self.neurons[i].update(input[i], current_time);
        }

        // Propagate spikes through network
        for i in 0..self.network_size {
            if spikes[i] {
                for (j, &target) in self.connections[i].iter().enumerate() {
                    if target < self.network_size {
                        let weight = self.weights[i][j];
                        let spike_result = self.neurons[target].update(weight, current_time);
                        if spike_result {
                            spikes[target] = true;
                        }
                    }
                }
            }
        }

        spikes
    }
}

/// Neuromorphic accelerator implementing spiking neural networks
pub struct NeuromorphicAccelerator {
    neural_network: NeuralNetwork,
    spike_threshold: f64,
    temporal_window: f64,
    current_time: f64,
}

impl NeuromorphicAccelerator {
    /// Creates a new neuromorphic accelerator
    pub fn new() -> Result<Self, AfiyahError> {
        let neural_network = NeuralNetwork::new(64);
        let spike_threshold = 0.5;
        let temporal_window = 1.0;
        let current_time = 0.0;

        Ok(Self {
            neural_network,
            spike_threshold,
            temporal_window,
            current_time,
        })
    }

    /// Processes input with neuromorphic acceleration
    pub fn process_spiking(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut output = Array2::zeros((height, width));

        // Convert 2D input to 1D for neural network
        let mut input_1d = Vec::new();
        for i in 0..height {
            for j in 0..width {
                input_1d.push(input[[i, j]]);
            }
        }

        // Process with neural network
        let spikes = self.neural_network.process(&input_1d, self.current_time);

        // Convert spikes back to 2D output
        for i in 0..height {
            for j in 0..width {
                let index = i * width + j;
                if index < spikes.len() {
                    output[[i, j]] = if spikes[index] { 1.0 } else { 0.0 };
                }
            }
        }

        // Update time
        self.current_time += 0.01;

        Ok(output)
    }

    /// Creates connections in the neural network
    pub fn create_connections(&mut self) -> Result<(), AfiyahError> {
        let size = self.neural_network.network_size;

        // Create random connections
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    let weight = (rand::random::<f64>() - 0.5) * 2.0;
                    self.neural_network.add_connection(i, j, weight);
                }
            }
        }

        Ok(())
    }

    /// Updates spike threshold
    pub fn set_spike_threshold(&mut self, threshold: f64) {
        self.spike_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates temporal window
    pub fn set_temporal_window(&mut self, window: f64) {
        self.temporal_window = window.clamp(0.0, 10.0);
    }

    /// Gets current neural network
    pub fn get_neural_network(&self) -> &NeuralNetwork {
        &self.neural_network
    }

    /// Gets current time
    pub fn get_current_time(&self) -> f64 {
        self.current_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_neuron_creation() {
        let neuron = SpikingNeuron::new(0.5, 0.1);
        assert_eq!(neuron.threshold, 0.5);
        assert_eq!(neuron.refractory_period, 0.1);
        assert_eq!(neuron.membrane_potential, 0.0);
    }

    #[test]
    fn test_spiking_neuron_update() {
        let mut neuron = SpikingNeuron::new(0.5, 0.1);
        let spike = neuron.update(0.6, 0.0);
        assert!(spike);
        assert_eq!(neuron.membrane_potential, 0.0);
    }

    #[test]
    fn test_neural_network_creation() {
        let network = NeuralNetwork::new(10);
        assert_eq!(network.network_size, 10);
        assert_eq!(network.neurons.len(), 10);
    }

    #[test]
    fn test_neural_network_connection() {
        let mut network = NeuralNetwork::new(10);
        network.add_connection(0, 1, 0.5);
        assert_eq!(network.connections[0].len(), 1);
        assert_eq!(network.weights[0].len(), 1);
    }

    #[test]
    fn test_neuromorphic_accelerator_creation() {
        let accelerator = NeuromorphicAccelerator::new();
        assert!(accelerator.is_ok());
    }

    #[test]
    fn test_spiking_processing() {
        let mut accelerator = NeuromorphicAccelerator::new().unwrap();
        let input = Array2::ones((8, 8));
        
        let result = accelerator.process_spiking(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.dim(), (8, 8));
    }

    #[test]
    fn test_spike_threshold_update() {
        let mut accelerator = NeuromorphicAccelerator::new().unwrap();
        accelerator.set_spike_threshold(0.7);
        assert_eq!(accelerator.spike_threshold, 0.7);
    }
}
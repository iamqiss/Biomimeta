//! Neuromorphic Interfaces Module

use ndarray::Array2;
use crate::AfiyahError;

/// Neuromorphic hardware type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuromorphicHardware {
    SpiNNaker,
    TrueNorth,
    Loihi,
    BrainScaleS,
    NeuroGrid,
    DYNAP,
    SpiNNaker2,
    Loihi2,
}

/// Neuromorphic configuration
#[derive(Debug, Clone)]
pub struct NeuromorphicConfig {
    pub hardware: NeuromorphicHardware,
    pub num_cores: usize,
    pub num_neurons: usize,
    pub num_synapses: usize,
    pub time_step: f64,
    pub voltage_threshold: f64,
    pub refractory_period: f64,
}

impl NeuromorphicConfig {
    pub fn new(hardware: NeuromorphicHardware) -> Self {
        let (num_cores, num_neurons, num_synapses) = match hardware {
            NeuromorphicHardware::SpiNNaker => (18, 1000, 10000),
            NeuromorphicHardware::TrueNorth => (4096, 1000000, 256000000),
            NeuromorphicHardware::Loihi => (128, 130000, 130000000),
            NeuromorphicHardware::BrainScaleS => (384, 200000, 50000000),
            NeuromorphicHardware::NeuroGrid => (16, 1000000, 1000000000),
            NeuromorphicHardware::DYNAP => (4, 1000, 10000),
            NeuromorphicHardware::SpiNNaker2 => (144, 10000000, 1000000000),
            NeuromorphicHardware::Loihi2 => (128, 1000000, 1000000000),
        };

        Self {
            hardware,
            num_cores,
            num_neurons,
            num_synapses,
            time_step: 1.0,
            voltage_threshold: 1.0,
            refractory_period: 2.0,
        }
    }
}

/// Neuromorphic neuron model
#[derive(Debug, Clone)]
pub struct NeuromorphicNeuron {
    pub id: usize,
    pub voltage: f64,
    pub threshold: f64,
    pub refractory_time: f64,
    pub last_spike_time: f64,
    pub input_current: f64,
    pub membrane_time_constant: f64,
    pub reset_voltage: f64,
}

impl NeuromorphicNeuron {
    pub fn new(id: usize, threshold: f64) -> Self {
        Self {
            id,
            voltage: 0.0,
            threshold,
            refractory_time: 0.0,
            last_spike_time: -1.0,
            input_current: 0.0,
            membrane_time_constant: 10.0,
            reset_voltage: 0.0,
        }
    }

    pub fn update(&mut self, dt: f64) -> bool {
        if self.refractory_time > 0.0 {
            self.refractory_time -= dt;
            return false;
        }

        // Leaky integrate-and-fire model
        self.voltage += (self.input_current - self.voltage) * dt / self.membrane_time_constant;

        if self.voltage >= self.threshold {
            self.voltage = self.reset_voltage;
            self.refractory_time = 2.0; // 2ms refractory period
            self.last_spike_time = 0.0; // Current time
            return true;
        }

        false
    }

    pub fn add_input(&mut self, current: f64) {
        self.input_current += current;
    }

    pub fn reset_input(&mut self) {
        self.input_current = 0.0;
    }
}

/// Neuromorphic synapse model
#[derive(Debug, Clone)]
pub struct NeuromorphicSynapse {
    pub id: usize,
    pub pre_neuron: usize,
    pub post_neuron: usize,
    pub weight: f64,
    pub delay: f64,
    pub plasticity: bool,
    pub learning_rate: f64,
    pub last_activity: f64,
}

impl NeuromorphicSynapse {
    pub fn new(id: usize, pre_neuron: usize, post_neuron: usize, weight: f64) -> Self {
        Self {
            id,
            pre_neuron,
            post_neuron,
            weight,
            delay: 1.0,
            plasticity: true,
            learning_rate: 0.01,
            last_activity: 0.0,
        }
    }

    pub fn update_weight(&mut self, pre_spike: bool, post_spike: bool, dt: f64) {
        if !self.plasticity {
            return;
        }

        // STDP (Spike-Timing Dependent Plasticity)
        if pre_spike && post_spike {
            self.weight += self.learning_rate * dt;
        } else if pre_spike && !post_spike {
            self.weight -= self.learning_rate * dt * 0.5;
        }

        // Weight bounds
        self.weight = self.weight.max(0.0).min(1.0);
    }
}

/// Neuromorphic interface for hardware acceleration
pub struct NeuromorphicInterface {
    config: NeuromorphicConfig,
    neurons: Vec<NeuromorphicNeuron>,
    synapses: Vec<NeuromorphicSynapse>,
    spike_times: Vec<f64>,
    current_time: f64,
}

impl NeuromorphicInterface {
    /// Creates a new neuromorphic interface
    pub fn new(config: NeuromorphicConfig) -> Self {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        // Initialize neurons
        for i in 0..config.num_neurons {
            let threshold = 1.0 + (i as f64 * 0.1) % 0.5;
            neurons.push(NeuromorphicNeuron::new(i, threshold));
        }

        // Initialize synapses
        for i in 0..config.num_synapses {
            let pre_neuron = i % config.num_neurons;
            let post_neuron = (i + 1) % config.num_neurons;
            let weight = 0.1 + (i as f64 * 0.01) % 0.8;
            synapses.push(NeuromorphicSynapse::new(i, pre_neuron, post_neuron, weight));
        }

        Self {
            config,
            neurons,
            synapses,
            spike_times: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Processes input through neuromorphic hardware
    pub fn process_input(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut output = Array2::zeros((height, width));

        // Convert input to spike trains
        self.convert_to_spikes(input)?;

        // Process through neuromorphic network
        self.process_network()?;

        // Convert spikes back to output
        self.convert_from_spikes(&mut output)?;

        Ok(output)
    }

    fn convert_to_spikes(&mut self, input: &Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();
        let total_pixels = height * width;

        if total_pixels > self.neurons.len() {
            return Err(AfiyahError::HardwareAcceleration { 
                message: "Input too large for neuromorphic hardware".to_string() 
            });
        }

        for i in 0..total_pixels {
            let row = i / width;
            let col = i % width;
            let intensity = input[[row, col]];

            // Convert intensity to input current
            let current = intensity * 10.0; // Scale factor
            self.neurons[i].add_input(current);
        }

        Ok(())
    }

    fn process_network(&mut self) -> Result<(), AfiyahError> {
        let dt = self.config.time_step;
        let mut spikes = Vec::new();

        // Update neurons
        for neuron in &mut self.neurons {
            if neuron.update(dt) {
                spikes.push(neuron.id);
            }
        }

        // Update synapses
        for synapse in &mut self.synapses {
            let pre_spike = spikes.contains(&synapse.pre_neuron);
            let post_spike = spikes.contains(&synapse.post_neuron);
            synapse.update_weight(pre_spike, post_spike, dt);
        }

        // Propagate spikes
        for &spike_id in &spikes {
            self.propagate_spike(spike_id)?;
        }

        // Reset input currents
        for neuron in &mut self.neurons {
            neuron.reset_input();
        }

        // Update time
        self.current_time += dt;

        Ok(())
    }

    fn propagate_spike(&mut self, spike_id: usize) -> Result<(), AfiyahError> {
        for synapse in &mut self.synapses {
            if synapse.pre_neuron == spike_id {
                let post_neuron = &mut self.neurons[synapse.post_neuron];
                post_neuron.add_input(synapse.weight);
            }
        }
        Ok(())
    }

    fn convert_from_spikes(&mut self, output: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = output.dim();
        let total_pixels = height * width;

        for i in 0..total_pixels {
            let row = i / width;
            let col = i % width;
            
            // Convert neuron voltage to output intensity
            let intensity = self.neurons[i].voltage.min(1.0).max(0.0);
            output[[row, col]] = intensity;
        }

        Ok(())
    }

    /// Gets the neuromorphic configuration
    pub fn get_config(&self) -> &NeuromorphicConfig {
        &self.config
    }

    /// Gets the number of neurons
    pub fn get_num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Gets the number of synapses
    pub fn get_num_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Gets the current time
    pub fn get_current_time(&self) -> f64 {
        self.current_time
    }

    /// Updates the neuromorphic configuration
    pub fn update_config(&mut self, config: NeuromorphicConfig) {
        let num_neurons = config.num_neurons;
        let num_synapses = config.num_synapses;
        self.config = config;
        self.neurons.clear();
        self.synapses.clear();

        // Reinitialize with new config
        for i in 0..num_neurons {
            let threshold = 1.0 + (i as f64 * 0.1) % 0.5;
            self.neurons.push(NeuromorphicNeuron::new(i, threshold));
        }

        for i in 0..num_synapses {
            let pre_neuron = i % num_neurons;
            let post_neuron = (i + 1) % num_neurons;
            let weight = 0.1 + (i as f64 * 0.01) % 0.8;
            self.synapses.push(NeuromorphicSynapse::new(i, pre_neuron, post_neuron, weight));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromorphic_config_creation() {
        let config = NeuromorphicConfig::new(NeuromorphicHardware::Loihi);
        assert_eq!(config.hardware, NeuromorphicHardware::Loihi);
        assert_eq!(config.num_cores, 128);
        assert_eq!(config.num_neurons, 130000);
    }

    #[test]
    fn test_neuromorphic_neuron_creation() {
        let neuron = NeuromorphicNeuron::new(0, 1.0);
        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.threshold, 1.0);
        assert_eq!(neuron.voltage, 0.0);
    }

    #[test]
    fn test_neuromorphic_synapse_creation() {
        let synapse = NeuromorphicSynapse::new(0, 1, 2, 0.5);
        assert_eq!(synapse.id, 0);
        assert_eq!(synapse.pre_neuron, 1);
        assert_eq!(synapse.post_neuron, 2);
        assert_eq!(synapse.weight, 0.5);
    }

    #[test]
    fn test_neuromorphic_interface_creation() {
        let config = NeuromorphicConfig::new(NeuromorphicHardware::Loihi);
        let interface = NeuromorphicInterface::new(config);
        assert_eq!(interface.get_num_neurons(), 130000);
        assert_eq!(interface.get_num_synapses(), 130000000);
    }

    #[test]
    fn test_neuromorphic_processing() {
        let config = NeuromorphicConfig::new(NeuromorphicHardware::Loihi);
        let mut interface = NeuromorphicInterface::new(config);
        let input = Array2::ones((32, 32));
        
        let result = interface.process_input(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.dim(), (32, 32));
    }

    #[test]
    fn test_different_hardware_types() {
        let hardware_types = vec![
            NeuromorphicHardware::SpiNNaker,
            NeuromorphicHardware::TrueNorth,
            NeuromorphicHardware::Loihi,
            NeuromorphicHardware::BrainScaleS,
            NeuromorphicHardware::NeuroGrid,
            NeuromorphicHardware::DYNAP,
            NeuromorphicHardware::SpiNNaker2,
            NeuromorphicHardware::Loihi2,
        ];

        for hardware in hardware_types {
            let config = NeuromorphicConfig::new(hardware);
            let mut interface = NeuromorphicInterface::new(config);
            let input = Array2::ones((16, 16));
            
            let result = interface.process_input(&input);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_config_update() {
        let config = NeuromorphicConfig::new(NeuromorphicHardware::Loihi);
        let mut interface = NeuromorphicInterface::new(config);
        
        let new_config = NeuromorphicConfig::new(NeuromorphicHardware::SpiNNaker2);
        interface.update_config(new_config);
        
        assert_eq!(interface.get_config().hardware, NeuromorphicHardware::SpiNNaker2);
        assert_eq!(interface.get_num_neurons(), 10000000);
    }
}
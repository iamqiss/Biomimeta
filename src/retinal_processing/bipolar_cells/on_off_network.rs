//! ON/OFF Network Implementation
//! 
//! This module implements the ON/OFF center-surround network of bipolar cells.
//! ON cells respond to light increments, OFF cells respond to light decrements.
//! This creates the fundamental center-surround antagonism in retinal processing.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::super::photoreceptors::PhotoreceptorResponse;

/// ON/OFF network for center-surround processing
pub struct OnOffNetwork {
    on_cells: OnCells,
    off_cells: OffCells,
    center_surround_filter: CenterSurroundFilter,
    adaptation_state: OnOffAdaptation,
}

impl OnOffNetwork {
    /// Creates new ON/OFF network with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            on_cells: OnCells::new()?,
            off_cells: OffCells::new()?,
            center_surround_filter: CenterSurroundFilter::new()?,
            adaptation_state: OnOffAdaptation::default(),
        })
    }

    /// Processes photoreceptor response through ON/OFF network
    pub fn process(&self, input: &PhotoreceptorResponse) -> Result<OnOffResponse, AfiyahError> {
        // Combine rod and cone signals for processing
        let combined_signals = self.combine_photoreceptor_signals(input)?;
        
        // Apply center-surround filtering
        let filtered_signals = self.center_surround_filter.filter(&combined_signals)?;
        
        // Process through ON cells
        let on_response = self.on_cells.process(&filtered_signals)?;
        
        // Process through OFF cells
        let off_response = self.off_cells.process(&filtered_signals)?;
        
        Ok(OnOffResponse {
            on_response,
            off_response,
            center_surround_ratio: self.center_surround_filter.center_surround_ratio,
            adaptation_level: self.adaptation_state.current_level,
        })
    }

    /// Calibrates ON/OFF network based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.on_cells.calibrate(params)?;
        self.off_cells.calibrate(params)?;
        self.center_surround_filter.calibrate(params)?;
        Ok(())
    }

    fn combine_photoreceptor_signals(&self, input: &PhotoreceptorResponse) -> Result<Vec<f64>, AfiyahError> {
        let mut combined = Vec::new();
        
        // Combine rod and cone signals with appropriate weighting
        let rod_weight = 0.3; // Rods contribute 30% in mixed lighting
        let cone_weight = 0.7; // Cones contribute 70% in mixed lighting
        
        for i in 0..input.rod_signals.len() {
            let rod_signal = input.rod_signals[i];
            let cone_signal = if i < input.cone_signals.len() {
                input.cone_signals[i]
            } else {
                0.0
            };
            
            let combined_signal = (rod_signal * rod_weight) + (cone_signal * cone_weight);
            combined.push(combined_signal);
        }
        
        Ok(combined)
    }
}

/// ON cells that respond to light increments
#[derive(Debug, Clone)]
pub struct OnCells {
    activation_threshold: f64,
    response_gain: f64,
    temporal_integration: f64,
}

impl OnCells {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            activation_threshold: 0.1,
            response_gain: 1.2,
            temporal_integration: 0.8,
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut on_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // ON cells respond to positive changes
            if signal > self.activation_threshold {
                let response = (signal - self.activation_threshold) * self.response_gain * self.temporal_integration;
                on_response.push(response.min(1.0));
            } else {
                on_response.push(0.0);
            }
        }
        
        Ok(on_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust response gain based on adaptation rate
        self.response_gain = 1.2 * params.adaptation_rate;
        Ok(())
    }
}

/// OFF cells that respond to light decrements
#[derive(Debug, Clone)]
pub struct OffCells {
    activation_threshold: f64,
    response_gain: f64,
    temporal_integration: f64,
}

impl OffCells {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            activation_threshold: 0.1,
            response_gain: 1.1, // Slightly lower gain than ON cells
            temporal_integration: 0.8,
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut off_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // OFF cells respond to negative changes (darkness)
            if signal < self.activation_threshold {
                let response = (self.activation_threshold - signal) * self.response_gain * self.temporal_integration;
                off_response.push(response.min(1.0));
            } else {
                off_response.push(0.0);
            }
        }
        
        Ok(off_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust response gain based on adaptation rate
        self.response_gain = 1.1 * params.adaptation_rate;
        Ok(())
    }
}

/// Center-surround filter for spatial processing
#[derive(Debug, Clone)]
pub struct CenterSurroundFilter {
    center_radius: f64,
    surround_radius: f64,
    center_surround_ratio: f64,
    spatial_frequency_tuning: f64,
}

impl CenterSurroundFilter {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            center_radius: 0.1,      // Center radius (normalized)
            surround_radius: 0.3,    // Surround radius (normalized)
            center_surround_ratio: 0.3, // Center is 30% of surround
            spatial_frequency_tuning: 0.5, // Optimal spatial frequency
        })
    }

    pub fn filter(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut filtered = Vec::with_capacity(input.len());
        
        // Simple center-surround filtering
        for (i, &signal) in input.iter().enumerate() {
            // Calculate center response
            let center_response = signal * self.center_surround_ratio;
            
            // Calculate surround response (simplified)
            let surround_response = signal * (1.0 - self.center_surround_ratio);
            
            // Center-surround antagonism
            let filtered_signal = center_response - surround_response;
            filtered.push(filtered_signal.max(0.0).min(1.0));
        }
        
        Ok(filtered)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust center-surround ratio based on adaptation
        self.center_surround_ratio = 0.3 * params.adaptation_rate;
        Ok(())
    }
}

/// ON/OFF adaptation state
#[derive(Debug, Clone)]
pub struct OnOffAdaptation {
    pub current_level: f64,
    pub adaptation_rate: f64,
    pub on_off_balance: f64,
}

impl Default for OnOffAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_rate: 0.1,
            on_off_balance: 0.5, // Balanced ON/OFF response
        }
    }
}

/// Response from ON/OFF network processing
#[derive(Debug, Clone)]
pub struct OnOffResponse {
    pub on_response: Vec<f64>,
    pub off_response: Vec<f64>,
    pub center_surround_ratio: f64,
    pub adaptation_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_off_network_creation() {
        let network = OnOffNetwork::new();
        assert!(network.is_ok());
    }

    #[test]
    fn test_on_cells() {
        let on_cells = OnCells::new().unwrap();
        let input = vec![0.0, 0.2, 0.5, 0.8];
        let result = on_cells.process(&input);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.len(), 4);
    }

    #[test]
    fn test_off_cells() {
        let off_cells = OffCells::new().unwrap();
        let input = vec![0.0, 0.2, 0.5, 0.8];
        let result = off_cells.process(&input);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.len(), 4);
    }

    #[test]
    fn test_center_surround_filter() {
        let filter = CenterSurroundFilter::new().unwrap();
        let input = vec![0.1, 0.5, 0.9];
        let result = filter.filter(&input);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_center_surround_ratio() {
        let filter = CenterSurroundFilter::new().unwrap();
        assert_eq!(filter.center_surround_ratio, 0.3);
    }
}
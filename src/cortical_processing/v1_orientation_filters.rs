//! V1 Orientation Filters Module
//! 
//! This module implements orientation-selective filters in primary visual cortex (V1)
//! based on Hubel & Wiesel's classical studies of simple and complex cells.

use crate::AfiyahError;
use std::f64::consts::PI;

/// Simple cell orientation filter
#[derive(Debug, Clone)]
pub struct SimpleCellFilter {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub adaptation_rate: f64,
    pub current_response: f64,
}

impl SimpleCellFilter {
    pub fn new(orientation: f64, spatial_frequency: f64) -> Result<Self, AfiyahError> {
        if orientation < 0.0 || orientation > PI {
            return Err(AfiyahError::BiologicalValidation(
                "Orientation must be between 0 and π".to_string()
            ));
        }
        
        Ok(Self {
            orientation,
            spatial_frequency,
            adaptation_rate: 0.1,
            current_response: 0.0,
        })
    }

    pub fn compute_response(&mut self, input: &[f64], width: u32, height: u32) -> Result<f64, AfiyahError> {
        let mut response = 0.0;
        
        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) as usize;
                if pixel_index < input.len() {
                    let gabor_value = self.gabor_function(x as f64, y as f64);
                    response += input[pixel_index] * gabor_value;
                }
            }
        }
        
        self.adapt_response(response);
        Ok(response)
    }

    fn gabor_function(&self, x: f64, y: f64) -> f64 {
        let cos_theta = self.orientation.cos();
        let sin_theta = self.orientation.sin();
        
        let x_rot = x * cos_theta + y * sin_theta;
        let y_rot = -x * sin_theta + y * cos_theta;
        
        let gaussian = (-(x_rot * x_rot) / 2.0 - (y_rot * y_rot) / 2.0).exp();
        let carrier = (2.0 * PI * self.spatial_frequency * x_rot).cos();
        
        gaussian * carrier
    }

    fn adapt_response(&mut self, new_response: f64) {
        let adaptation_factor = 1.0 - self.adaptation_rate * self.current_response.abs();
        self.current_response = new_response * adaptation_factor.max(0.1);
    }
}

/// Complex cell orientation filter
#[derive(Debug, Clone)]
pub struct ComplexCellFilter {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub adaptation_rate: f64,
    pub current_response: f64,
}

impl ComplexCellFilter {
    pub fn new(orientation: f64, spatial_frequency: f64) -> Result<Self, AfiyahError> {
        if orientation < 0.0 || orientation > PI {
            return Err(AfiyahError::BiologicalValidation(
                "Orientation must be between 0 and π".to_string()
            ));
        }
        
        Ok(Self {
            orientation,
            spatial_frequency,
            adaptation_rate: 0.1,
            current_response: 0.0,
        })
    }

    pub fn compute_response(&mut self, input: &[f64], width: u32, height: u32) -> Result<f64, AfiyahError> {
        let mut simple_cell_1 = SimpleCellFilter::new(self.orientation, self.spatial_frequency)?;
        let mut simple_cell_2 = SimpleCellFilter::new(self.orientation, self.spatial_frequency)?;
        
        let response_1 = simple_cell_1.compute_response(input, width, height)?;
        let response_2 = simple_cell_2.compute_response(input, width, height)?;
        
        let energy_response = (response_1 * response_1 + response_2 * response_2).sqrt();
        self.adapt_response(energy_response);
        
        Ok(energy_response)
    }

    fn adapt_response(&mut self, new_response: f64) {
        let adaptation_factor = 1.0 - self.adaptation_rate * self.current_response.abs();
        self.current_response = new_response * adaptation_factor.max(0.1);
    }
}

/// Bank of orientation filters
#[derive(Debug)]
pub struct OrientationFilterBank {
    pub simple_cells: Vec<SimpleCellFilter>,
    pub complex_cells: Vec<ComplexCellFilter>,
    pub num_orientations: u32,
}

impl OrientationFilterBank {
    pub fn new(num_orientations: u32, spatial_frequency: f64) -> Result<Self, AfiyahError> {
        let mut simple_cells = Vec::new();
        let mut complex_cells = Vec::new();
        
        for i in 0..num_orientations {
            let orientation = (i as f64 * PI) / (num_orientations as f64);
            simple_cells.push(SimpleCellFilter::new(orientation, spatial_frequency)?);
            complex_cells.push(ComplexCellFilter::new(orientation, spatial_frequency)?);
        }
        
        Ok(Self {
            simple_cells,
            complex_cells,
            num_orientations,
        })
    }

    pub fn process(&mut self, input: &[f64], width: u32, height: u32) -> Result<OrientationResponse, AfiyahError> {
        let mut simple_responses = Vec::new();
        let mut complex_responses = Vec::new();
        
        for cell in &mut self.simple_cells {
            let response = cell.compute_response(input, width, height)?;
            simple_responses.push(response);
        }
        
        for cell in &mut self.complex_cells {
            let response = cell.compute_response(input, width, height)?;
            complex_responses.push(response);
        }
        
        let dominant_orientation = self.find_dominant_orientation(&complex_responses)?;
        let orientation_energy = self.compute_orientation_energy(&complex_responses);
        
        Ok(OrientationResponse {
            simple_responses,
            complex_responses,
            dominant_orientation,
            orientation_energy,
        })
    }

    fn find_dominant_orientation(&self, responses: &[f64]) -> Result<f64, AfiyahError> {
        if responses.is_empty() {
            return Err(AfiyahError::InputError("No responses to analyze".to_string()));
        }
        
        let max_index = responses.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let orientation = (max_index as f64 * PI) / (self.num_orientations as f64);
        Ok(orientation)
    }

    fn compute_orientation_energy(&self, responses: &[f64]) -> f64 {
        responses.iter().map(|&r| r * r).sum::<f64>().sqrt()
    }
}

/// Response from orientation filter bank
#[derive(Debug, Clone)]
pub struct OrientationResponse {
    pub simple_responses: Vec<f64>,
    pub complex_responses: Vec<f64>,
    pub dominant_orientation: f64,
    pub orientation_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_cell_filter_creation() {
        let filter = SimpleCellFilter::new(PI / 4.0, 2.0);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_complex_cell_filter_creation() {
        let filter = ComplexCellFilter::new(PI / 2.0, 1.5);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_orientation_filter_bank() {
        let bank = OrientationFilterBank::new(8, 2.0);
        assert!(bank.is_ok());
    }
}

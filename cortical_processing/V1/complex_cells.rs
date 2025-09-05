/* Biomimeta - Biomimetic Video Compression & Streaming Engine
*  Copyright (C) 2025 Neo Qiss. All Rights Reserved.
*
*  PROPRIETARY NOTICE: This software and all associated intellectual property,
*  including but not limited to algorithms, biological models, neural architectures,
*  and compression methodologies, are the exclusive property of Neo Qiss.
*
*  COMMERCIAL RESTRICTION: Commercial use, distribution, or integration of this
*  software is STRICTLY PROHIBITED without explicit written authorization and
*  formal partnership agreements. Unauthorized commercial use constitutes
*  copyright infringement and may result in legal action.
*
*  RESEARCH LICENSE: This software is made available under the Biological Research
*  Public License (BRPL) v1.0 EXCLUSIVELY for academic research, educational purposes,
*  and non-commercial scientific collaboration. Commercial entities must obtain
*  separate licensing agreements.
*
*  BIOLOGICAL RESEARCH ATTRIBUTION: This software implements proprietary biological
*  models derived from extensive neuroscientific research. All use must maintain
*  complete scientific attribution as specified in the BRPL license terms.
*
*  NO WARRANTIES: This software is provided for research purposes only. No warranties
*  are made regarding biological accuracy, medical safety, or fitness for any purpose.
*
*  For commercial licensing: commercial@biomimeta.com
*  For research partnerships: research@biomimeta.com
*  Legal inquiries: legal@biomimeta.com
*
*  VIOLATION OF THESE TERMS MAY RESULT IN IMMEDIATE LICENSE TERMINATION AND LEGAL ACTION.
*/

//! Complex Cells Implementation for V1 Primary Visual Cortex
//! 
//! Complex cells are motion-invariant feature detectors with non-linear responses.
//! They respond to oriented features regardless of their exact position within the receptive field.

use crate::AfiyahError;
use ndarray::{Array2, Array3};

/// Complex cell for motion-invariant feature detection
#[derive(Debug, Clone)]
pub struct ComplexCell {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub receptive_field_size: usize,
    pub phase_invariance: f64,
    pub motion_sensitivity: f64,
}

/// Bank of complex cells for motion-invariant processing
pub struct ComplexCellBank {
    cells: Vec<ComplexCell>,
    orientations: Vec<f64>,
    spatial_frequencies: Vec<f64>,
}

impl ComplexCell {
    /// Creates a new complex cell with specified parameters
    pub fn new(orientation: f64, spatial_frequency: f64, field_size: usize) -> Result<Self, AfiyahError> {
        Ok(Self {
            orientation,
            spatial_frequency,
            receptive_field_size: field_size,
            phase_invariance: 0.8,
            motion_sensitivity: 1.0,
        })
    }

    /// Computes response to simple cell inputs
    pub fn compute_response(&self, simple_responses: &[f64]) -> Result<f64, AfiyahError> {
        // Complex cell response is phase-invariant version of simple cell responses
        // This is a simplified implementation - in reality, it would involve
        // pooling over multiple simple cells with different phases
        
        let mut response = 0.0;
        let mut count = 0.0;
        
        for &simple_response in simple_responses {
            // Use absolute value for phase invariance
            response += simple_response.abs();
            count += 1.0;
        }
        
        if count > 0.0 {
            response = (response / count) * self.phase_invariance * self.motion_sensitivity;
        }
        
        Ok(response)
    }

    /// Gets orientation selectivity
    pub fn get_orientation_selectivity(&self) -> f64 {
        // Complex cells have broader orientation tuning than simple cells
        25.0 // degrees
    }

    /// Gets spatial frequency bandwidth
    pub fn get_spatial_frequency_bandwidth(&self) -> f64 {
        // Complex cells have broader spatial frequency tuning
        2.0 // octaves
    }
}

impl ComplexCellBank {
    /// Creates a new complex cell bank
    pub fn new() -> Result<Self, AfiyahError> {
        let mut cells = Vec::new();
        let orientations = vec![0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5];
        let spatial_frequencies = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        
        for &orientation in &orientations {
            for &spatial_freq in &spatial_frequencies {
                let cell = ComplexCell::new(orientation, spatial_freq, 7)?;
                cells.push(cell);
            }
        }
        
        Ok(Self {
            cells,
            orientations,
            spatial_frequencies,
        })
    }

    /// Processes simple cell responses through complex cells
    pub fn process(&self, simple_responses: &Array3<f64>) -> Result<Array3<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = simple_responses.dim();
        let mut complex_responses = Array3::zeros((orientations, spatial_freqs, spatial_size));
        
        let mut cell_idx = 0;
        for orientation_idx in 0..orientations {
            for spatial_freq_idx in 0..spatial_freqs {
                if cell_idx < self.cells.len() {
                    let cell = &self.cells[cell_idx];
                    
                    for spatial_idx in 0..spatial_size {
                        // Get simple cell response for this location
                        let simple_response = simple_responses[[orientation_idx, spatial_freq_idx, spatial_idx]];
                        
                        // Compute complex cell response
                        let complex_response = cell.compute_response(&[simple_response])?;
                        complex_responses[[orientation_idx, spatial_freq_idx, spatial_idx]] = complex_response;
                    }
                    
                    cell_idx += 1;
                }
            }
        }
        
        Ok(complex_responses)
    }

    /// Gets motion sensitivity for specific orientation
    pub fn get_motion_sensitivity(&self, orientation: f64) -> Result<f64, AfiyahError> {
        for cell in &self.cells {
            if (cell.orientation - orientation).abs() < 1.0 {
                return Ok(cell.motion_sensitivity);
            }
        }
        Ok(0.0)
    }

    /// Updates motion sensitivity based on temporal dynamics
    pub fn update_motion_sensitivity(&mut self, temporal_input: &[f64]) -> Result<(), AfiyahError> {
        // Update motion sensitivity based on temporal correlation
        if temporal_input.len() > 1 {
            let temporal_correlation = self.calculate_temporal_correlation(temporal_input);
            for cell in &mut self.cells {
                cell.motion_sensitivity = (1.0 + temporal_correlation).max(0.1).min(2.0);
            }
        }
        Ok(())
    }

    fn calculate_temporal_correlation(&self, temporal_input: &[f64]) -> f64 {
        if temporal_input.len() < 2 {
            return 0.0;
        }
        
        let mut correlation = 0.0;
        for i in 1..temporal_input.len() {
            correlation += temporal_input[i] * temporal_input[i-1];
        }
        
        correlation / (temporal_input.len() - 1) as f64
    }
}
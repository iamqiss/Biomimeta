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

//! Edge Detection Implementation for V1 Primary Visual Cortex
//! 
//! Edge detection using multiple orientation channels and biological constraints.
//! Implements non-maximum suppression and edge linking algorithms.

use crate::AfiyahError;
use ndarray::{Array2, Array3};

/// Edge detector using multiple orientation channels
pub struct EdgeDetector {
    orientation_threshold: f64,
    edge_threshold: f64,
    non_maximum_suppression: bool,
    edge_linking: bool,
}

/// Edge detection result
#[derive(Debug, Clone)]
pub struct EdgeMap {
    pub edge_strength: Array2<f64>,
    pub edge_orientation: Array2<f64>,
    pub edge_direction: Array2<f64>,
    pub edge_confidence: Array2<f64>,
}

impl EdgeDetector {
    /// Creates a new edge detector
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            orientation_threshold: 0.1,
            edge_threshold: 0.2,
            non_maximum_suppression: true,
            edge_linking: true,
        })
    }

    /// Detects edges from simple cell responses
    pub fn detect_edges(&self, simple_responses: &Array3<f64>) -> Result<EdgeMap, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = simple_responses.dim();
        let size = (spatial_size as f64).sqrt() as usize;
        
        let mut edge_strength = Array2::zeros((size, size));
        let mut edge_orientation = Array2::zeros((size, size));
        let mut edge_direction = Array2::zeros((size, size));
        let mut edge_confidence = Array2::zeros((size, size));
        
        // Use first spatial frequency for edge detection
        for h in 0..size {
            for w in 0..size {
                let spatial_idx = h * size + w;
                let mut max_response = 0.0;
                let mut dominant_orientation = 0.0;
                let mut orientation_variance = 0.0;
                
                // Find dominant orientation and calculate edge strength
                for o in 0..orientations {
                    let response = simple_responses[[o, 0, spatial_idx]].abs();
                    if response > max_response {
                        max_response = response;
                        dominant_orientation = (o as f64) * (180.0 / orientations as f64);
                    }
                }
                
                // Calculate orientation variance for confidence
                let mut responses = Vec::new();
                for o in 0..orientations {
                    responses.push(simple_responses[[o, 0, spatial_idx]].abs());
                }
                orientation_variance = self.calculate_variance(&responses);
                
                edge_strength[[h, w]] = max_response;
                edge_orientation[[h, w]] = dominant_orientation;
                edge_direction[[h, w]] = dominant_orientation + 90.0; // Perpendicular to edge
                edge_confidence[[h, w]] = 1.0 / (1.0 + orientation_variance);
            }
        }
        
        // Apply non-maximum suppression
        if self.non_maximum_suppression {
            edge_strength = self.apply_non_maximum_suppression(&edge_strength, &edge_direction)?;
        }
        
        // Apply edge linking
        if self.edge_linking {
            edge_strength = self.apply_edge_linking(&edge_strength, &edge_orientation)?;
        }
        
        Ok(EdgeMap {
            edge_strength,
            edge_orientation,
            edge_direction,
            edge_confidence,
        })
    }

    /// Applies non-maximum suppression to thin edges
    fn apply_non_maximum_suppression(&self, edge_strength: &Array2<f64>, edge_direction: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = edge_strength.dim();
        let mut suppressed = edge_strength.clone();
        
        for h in 1..height-1 {
            for w in 1..width-1 {
                let direction = edge_direction[[h, w]];
                let current_strength = edge_strength[[h, w]];
                
                // Determine neighboring pixels based on edge direction
                let (neighbor1, neighbor2) = self.get_neighbors(h, w, direction, height, width);
                
                if let (Some(n1), Some(n2)) = (neighbor1, neighbor2) {
                    let strength1 = edge_strength[[n1.0, n1.1]];
                    let strength2 = edge_strength[[n2.0, n2.1]];
                    
                    // Suppress if current pixel is not a local maximum
                    if current_strength <= strength1 || current_strength <= strength2 {
                        suppressed[[h, w]] = 0.0;
                    }
                }
            }
        }
        
        Ok(suppressed)
    }

    /// Applies edge linking to connect broken edges
    fn apply_edge_linking(&self, edge_strength: &Array2<f64>, edge_orientation: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = edge_strength.dim();
        let mut linked = edge_strength.clone();
        
        // Simple edge linking - connect nearby strong edges with similar orientation
        for h in 1..height-1 {
            for w in 1..width-1 {
                if edge_strength[[h, w]] > self.edge_threshold {
                    let current_orientation = edge_orientation[[h, w]];
                    
                    // Check 8-connected neighbors
                    for dh in -1..=1 {
                        for dw in -1..=1 {
                            if dh == 0 && dw == 0 { continue; }
                            
                            let nh = (h as i32 + dh) as usize;
                            let nw = (w as i32 + dw) as usize;
                            
                            if nh < height && nw < width {
                                let neighbor_strength = edge_strength[[nh, nw]];
                                let neighbor_orientation = edge_orientation[[nh, nw]];
                                
                                // Link if orientations are similar and neighbor is weak
                                let orientation_diff = (current_orientation - neighbor_orientation).abs();
                                if orientation_diff < 30.0 && neighbor_strength < self.edge_threshold {
                                    linked[[nh, nw]] = neighbor_strength * 0.5; // Reduce strength for linked edges
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(linked)
    }

    /// Gets neighboring pixels based on edge direction
    fn get_neighbors(&self, h: usize, w: usize, direction: f64, height: usize, width: usize) -> (Option<(usize, usize)>, Option<(usize, usize)>) {
        let angle = direction.to_radians();
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        // Calculate neighbor positions
        let h1 = (h as f64 + sin_angle) as usize;
        let w1 = (w as f64 + cos_angle) as usize;
        let h2 = (h as f64 - sin_angle) as usize;
        let w2 = (w as f64 - cos_angle) as usize;
        
        let neighbor1 = if h1 < height && w1 < width { Some((h1, w1)) } else { None };
        let neighbor2 = if h2 < height && w2 < width { Some((h2, w2)) } else { None };
        
        (neighbor1, neighbor2)
    }

    /// Calculates variance of responses
    fn calculate_variance(&self, responses: &[f64]) -> f64 {
        if responses.is_empty() {
            return 0.0;
        }
        
        let mean = responses.iter().sum::<f64>() / responses.len() as f64;
        let variance = responses.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / responses.len() as f64;
        
        variance
    }

    /// Sets edge detection threshold
    pub fn set_edge_threshold(&mut self, threshold: f64) {
        self.edge_threshold = threshold;
    }

    /// Sets orientation threshold
    pub fn set_orientation_threshold(&mut self, threshold: f64) {
        self.orientation_threshold = threshold;
    }

    /// Enables or disables non-maximum suppression
    pub fn set_non_maximum_suppression(&mut self, enabled: bool) {
        self.non_maximum_suppression = enabled;
    }

    /// Enables or disables edge linking
    pub fn set_edge_linking(&mut self, enabled: bool) {
        self.edge_linking = enabled;
    }
}
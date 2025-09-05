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

//! Motion Processing Module
//! 
//! This module implements sophisticated motion detection and processing algorithms
//! based on neurophysiological studies of V5/MT (middle temporal) area.
//! 
//! Biological Basis:
//! - Newsome & Pare (1988): MT motion selectivity and direction tuning
//! - Movshon et al. (1985): Complex motion patterns in MT
//! - Born & Bradley (2005): Motion integration and segmentation
//! 
//! Key Features:
//! - Direction-selective motion detection
//! - Speed tuning and velocity estimation
//! - Motion coherence analysis
//! - Motion segmentation and grouping

use ndarray::{Array2, Array3, ArrayView2};
use crate::AfiyahError;

/// Motion vector representing direction and magnitude
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    pub x: f64,
    pub y: f64,
    pub magnitude: f64,
    pub direction: f64, // radians
    pub confidence: f64,
}

impl MotionVector {
    pub fn new(x: f64, y: f64, confidence: f64) -> Self {
        let magnitude = (x * x + y * y).sqrt();
        let direction = y.atan2(x);
        Self {
            x,
            y,
            magnitude,
            direction,
            confidence,
        }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            magnitude: 0.0,
            direction: 0.0,
            confidence: 0.0,
        }
    }
}

/// Motion field containing vectors for each spatial location
#[derive(Debug, Clone)]
pub struct MotionField {
    pub vectors: Array2<MotionVector>,
    pub coherence: f64,
    pub global_motion: MotionVector,
}

impl MotionField {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            vectors: Array2::from_shape_fn((height, width), |_| MotionVector::zero()),
            coherence: 0.0,
            global_motion: MotionVector::zero(),
        }
    }

    pub fn calculate_coherence(&self) -> Result<f64, AfiyahError> {
        // Calculate motion coherence based on vector alignment
        let mut total_magnitude = 0.0;
        let mut aligned_magnitude = 0.0;
        
        for vector in self.vectors.iter() {
            total_magnitude += vector.magnitude;
            if vector.confidence > 0.5 {
                aligned_magnitude += vector.magnitude;
            }
        }
        
        if total_magnitude > 0.0 {
            Ok(aligned_magnitude / total_magnitude)
        } else {
            Ok(0.0)
        }
    }

    pub fn calculate_global_motion(&mut self) -> Result<(), AfiyahError> {
        // Calculate global motion vector as weighted average
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut total_weight = 0.0;
        
        for vector in self.vectors.iter() {
            let weight = vector.confidence * vector.magnitude;
            weighted_x += vector.x * weight;
            weighted_y += vector.y * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            self.global_motion = MotionVector::new(
                weighted_x / total_weight,
                weighted_y / total_weight,
                self.coherence,
            );
        }
        
        Ok(())
    }
}

impl Default for MotionField {
    fn default() -> Self {
        Self::new(64, 64)
    }
}

/// Motion detector implementing biological motion processing
pub struct MotionDetector {
    direction_filters: Array3<f64>,
    speed_tuning: Vec<f64>,
    temporal_window: usize,
    spatial_window: usize,
}

impl MotionDetector {
    /// Creates a new motion detector with biological parameters
    pub fn new() -> Result<Self, AfiyahError> {
        let num_directions = 8;
        let num_speeds = 4;
        let spatial_size = 7;
        
        // Initialize direction-selective filters (Gabor-like)
        let mut direction_filters = Array3::zeros((num_directions, num_speeds, spatial_size * spatial_size));
        
        for dir in 0..num_directions {
            for speed in 0..num_speeds {
                let direction = (dir as f64) * std::f64::consts::PI / 4.0;
                let speed_val = (speed as f64 + 1.0) * 0.5; // 0.5, 1.0, 1.5, 2.0 pixels/frame
                
                for i in 0..spatial_size {
                    for j in 0..spatial_size {
                        let x = (j as f64) - (spatial_size as f64) / 2.0;
                        let y = (i as f64) - (spatial_size as f64) / 2.0;
                        
                        // Create direction-selective filter
                        let filter_val = Self::create_direction_filter(x, y, direction, speed_val);
                        direction_filters[[dir, speed, i * spatial_size + j]] = filter_val;
                    }
                }
            }
        }
        
        Ok(Self {
            direction_filters,
            speed_tuning: vec![0.5, 1.0, 1.5, 2.0],
            temporal_window: 3,
            spatial_window: spatial_size,
        })
    }

    /// Detects motion between two frames
    pub fn detect_motion(&self, current: &Array2<f64>, previous: &Array2<f64>) -> Result<MotionField, AfiyahError> {
        let (height, width) = current.dim();
        let mut motion_field = MotionField::new(height, width);
        
        // Calculate frame difference
        let frame_diff = current - previous;
        
        // Process each spatial location
        for i in 0..height {
            for j in 0..width {
                if i >= self.spatial_window / 2 && i < height - self.spatial_window / 2 &&
                   j >= self.spatial_window / 2 && j < width - self.spatial_window / 2 {
                    
                    let motion_vector = self.calculate_motion_vector(&frame_diff, i, j)?;
                    motion_field.vectors[[i, j]] = motion_vector;
                }
            }
        }
        
        // Calculate coherence and global motion
        motion_field.coherence = motion_field.calculate_coherence()?;
        motion_field.calculate_global_motion()?;
        
        Ok(motion_field)
    }

    fn create_direction_filter(x: f64, y: f64, direction: f64, speed: f64) -> f64 {
        // Create direction-selective Gabor-like filter
        let sigma = 1.5;
        let lambda = 2.0;
        
        // Rotate coordinates
        let x_rot = x * direction.cos() + y * direction.sin();
        let y_rot = -x * direction.sin() + y * direction.cos();
        
        // Gaussian envelope
        let gaussian = (-(x_rot * x_rot + y_rot * y_rot) / (2.0 * sigma * sigma)).exp();
        
        // Sinusoidal carrier
        let carrier = (2.0 * std::f64::consts::PI * x_rot / lambda).sin();
        
        // Speed-dependent modulation
        let speed_mod = (-(y_rot * y_rot) / (2.0 * speed * speed)).exp();
        
        gaussian * carrier * speed_mod
    }

    fn calculate_motion_vector(&self, frame_diff: &Array2<f64>, i: usize, j: usize) -> Result<MotionVector, AfiyahError> {
        let mut best_response = 0.0;
        let mut best_direction = 0;
        let mut best_speed = 0;
        
        // Extract local patch
        let patch_size = self.spatial_window;
        let mut patch = Array2::zeros((patch_size, patch_size));
        
        for pi in 0..patch_size {
            for pj in 0..patch_size {
                let gi = i + pi - patch_size / 2;
                let gj = j + pj - patch_size / 2;
                if gi < frame_diff.nrows() && gj < frame_diff.ncols() {
                    patch[[pi, pj]] = frame_diff[[gi, gj]];
                }
            }
        }
        
        // Convolve with direction-selective filters
        for dir in 0..self.direction_filters.nrows() {
            for speed in 0..self.direction_filters.ncols() {
                let mut response = 0.0;
                
                for pi in 0..patch_size {
                    for pj in 0..patch_size {
                        let filter_idx = pi * patch_size + pj;
                        response += patch[[pi, pj]] * self.direction_filters[[dir, speed, filter_idx]];
                    }
                }
                
                if response.abs() > best_response {
                    best_response = response.abs();
                    best_direction = dir;
                    best_speed = speed;
                }
            }
        }
        
        // Convert to motion vector
        let direction = (best_direction as f64) * std::f64::consts::PI / 4.0;
        let speed = self.speed_tuning[best_speed];
        let confidence = (best_response / 10.0).min(1.0); // Normalize confidence
        
        let x = speed * direction.cos();
        let y = speed * direction.sin();
        
        Ok(MotionVector::new(x, y, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_creation() {
        let vector = MotionVector::new(1.0, 1.0, 0.8);
        assert!((vector.magnitude - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((vector.direction - std::f64::consts::PI / 4.0).abs() < 1e-10);
        assert_eq!(vector.confidence, 0.8);
    }

    #[test]
    fn test_motion_field_creation() {
        let field = MotionField::new(32, 32);
        assert_eq!(field.vectors.dim(), (32, 32));
        assert_eq!(field.coherence, 0.0);
    }

    #[test]
    fn test_motion_detector_creation() {
        let detector = MotionDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_motion_detection() {
        let detector = MotionDetector::new().unwrap();
        
        // Create test frames with horizontal motion
        let mut current = Array2::zeros((32, 32));
        let mut previous = Array2::zeros((32, 32));
        
        // Add a moving pattern
        for i in 10..20 {
            for j in 10..20 {
                current[[i, j]] = 1.0;
                previous[[i, j - 1]] = 1.0; // Moved one pixel to the right
            }
        }
        
        let result = detector.detect_motion(&current, &previous);
        assert!(result.is_ok());
        
        let motion_field = result.unwrap();
        assert!(motion_field.coherence > 0.0);
    }
}
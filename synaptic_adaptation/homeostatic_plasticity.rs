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

//! Homeostatic Plasticity Implementation

use crate::AfiyahError;
use ndarray::Array3;

/// Homeostatic plasticity controller
pub struct HomeostaticController {
    target_activity: f64,
    adaptation_rate: f64,
    scaling_factor: f64,
    activity_history: Vec<f64>,
}

impl HomeostaticController {
    /// Creates a new homeostatic controller
    pub fn new(target_activity: f64) -> Result<Self, AfiyahError> {
        Ok(Self {
            target_activity,
            adaptation_rate: 0.01,
            scaling_factor: 1.0,
            activity_history: Vec::new(),
        })
    }

    /// Adjusts weights to maintain target activity
    pub fn adjust_weights(&mut self, weights: &Array3<f64>, activity_patterns: &Array3<f64>) -> Result<Array3<f64>, AfiyahError> {
        let current_activity = activity_patterns.mean().unwrap_or(0.0);
        self.activity_history.push(current_activity);
        
        if self.activity_history.len() > 100 {
            self.activity_history.remove(0);
        }
        
        // Calculate scaling factor to maintain target activity
        let activity_error = self.target_activity - current_activity;
        self.scaling_factor += self.adaptation_rate * activity_error;
        self.scaling_factor = self.scaling_factor.max(0.1).min(2.0);
        
        // Apply scaling to weights
        let adjusted_weights = weights * self.scaling_factor;
        Ok(adjusted_weights)
    }

    /// Gets current error from target
    pub fn get_error(&self) -> f64 {
        if !self.activity_history.is_empty() {
            let current_activity = self.activity_history.last().unwrap_or(&0.0);
            (self.target_activity - current_activity).abs()
        } else {
            0.0
        }
    }
}
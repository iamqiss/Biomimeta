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

//! Hebbian Learning Implementation

use crate::AfiyahError;
use ndarray::Array3;

/// Hebbian learning processor
pub struct HebbianLearner {
    learning_rate: f64,
    weight_decay: f64,
    correlation_threshold: f64,
    weight_history: Vec<f64>,
}

impl HebbianLearner {
    /// Creates a new Hebbian learner
    pub fn new(learning_rate: f64) -> Result<Self, AfiyahError> {
        Ok(Self {
            learning_rate,
            weight_decay: 0.001,
            correlation_threshold: 0.1,
            weight_history: Vec::new(),
        })
    }

    /// Updates synaptic weights using Hebbian learning
    pub fn update_weights(&mut self, activity_patterns: &Array3<f64>) -> Result<Array3<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = activity_patterns.dim();
        let mut weights = Array3::ones((orientations, spatial_freqs, spatial_size)) * 0.5;
        
        for o in 0..orientations {
            for s in 0..spatial_freqs {
                for p in 0..spatial_size {
                    let activity = activity_patterns[[o, s, p]];
                    let current_weight = weights[[o, s, p]];
                    
                    // Standard Hebbian: Δw = η * pre * post
                    let weight_change = self.learning_rate * activity * current_weight;
                    let new_weight: f64 = current_weight + weight_change - self.weight_decay * current_weight;
                    
                    weights[[o, s, p]] = new_weight.max(0.0).min(1.0);
                }
            }
        }
        
        // Store weight history
        let avg_weight = weights.mean().unwrap_or(0.0);
        self.weight_history.push(avg_weight);
        if self.weight_history.len() > 100 {
            self.weight_history.remove(0);
        }
        
        Ok(weights)
    }

    /// Gets activity level
    pub fn get_activity_level(&self) -> f64 {
        if self.weight_history.len() > 1 {
            let recent_avg = self.weight_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older_avg = self.weight_history.iter().take(10).sum::<f64>() / 10.0;
            (recent_avg - older_avg).abs()
        } else {
            0.0
        }
    }
}
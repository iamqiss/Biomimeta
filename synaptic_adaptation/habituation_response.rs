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

//! Habituation Response Implementation

use crate::AfiyahError;
use ndarray::Array3;

/// Habituation response controller
pub struct HabituationController {
    habituation_rate: f64,
    recovery_rate: f64,
    habituation_threshold: f64,
    response_history: Vec<f64>,
}

impl HabituationController {
    /// Creates a new habituation controller
    pub fn new(habituation_rate: f64) -> Result<Self, AfiyahError> {
        Ok(Self {
            habituation_rate,
            recovery_rate: 0.01,
            habituation_threshold: 0.1,
            response_history: Vec::new(),
        })
    }

    /// Applies habituation to weights
    pub fn apply_habituation(&mut self, weights: &Array3<f64>, activity_patterns: &Array3<f64>) -> Result<Array3<f64>, AfiyahError> {
        let current_activity = activity_patterns.mean().unwrap_or(0.0);
        self.response_history.push(current_activity);
        
        if self.response_history.len() > 50 {
            self.response_history.remove(0);
        }
        
        // Calculate habituation level
        let habituation_level = self.calculate_habituation_level();
        
        // Apply habituation to weights
        let habituation_factor = 1.0 - habituation_level;
        let habituated_weights = weights * habituation_factor;
        
        Ok(habituated_weights)
    }

    /// Calculates habituation level
    fn calculate_habituation_level(&self) -> f64 {
        if self.response_history.len() < 10 {
            return 0.0;
        }
        
        let recent_responses = &self.response_history[self.response_history.len()-10..];
        let older_responses = &self.response_history[..10];
        
        let recent_avg = recent_responses.iter().sum::<f64>() / recent_responses.len() as f64;
        let older_avg = older_responses.iter().sum::<f64>() / older_responses.len() as f64;
        
        let habituation = (older_avg - recent_avg).max(0.0);
        habituation.min(1.0)
    }

    /// Gets habituation level
    pub fn get_habituation_level(&self) -> f64 {
        self.calculate_habituation_level()
    }
}
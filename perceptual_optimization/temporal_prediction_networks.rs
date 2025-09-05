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

//! Temporal Prediction Networks Module
//! 
//! This module implements sophisticated temporal prediction networks based on
//! biological motion processing and predictive coding mechanisms.
//! 
//! Biological Basis:
//! - Rao & Ballard (1999): Predictive coding in visual cortex
//! - Friston (2005): Predictive coding and free energy
//! - Spratling (2017): Predictive coding in visual processing
//! - Clark (2013): Predictive processing and perception

use ndarray::{Array2, Array3};
use crate::AfiyahError;

/// Temporal prediction network implementing biological predictive coding
pub struct TemporalPredictionNetwork {
    prediction_window: usize,
    learning_rate: f64,
    prediction_weights: Array3<f64>,
    temporal_filters: Array2<f64>,
    prediction_threshold: f64,
}

impl TemporalPredictionNetwork {
    /// Creates a new temporal prediction network
    pub fn new() -> Result<Self, AfiyahError> {
        let prediction_window = 5;
        let learning_rate = 0.01;
        let prediction_weights = Self::create_prediction_weights(prediction_window)?;
        let temporal_filters = Self::create_temporal_filters()?;
        let prediction_threshold = 0.3;

        Ok(Self {
            prediction_window,
            learning_rate,
            prediction_weights,
            temporal_filters,
            prediction_threshold,
        })
    }

    /// Predicts next frame based on temporal sequence
    pub fn predict_next_frame(&mut self, sequence: &[Array2<f64>]) -> Result<Array2<f64>, AfiyahError> {
        if sequence.is_empty() {
            return Err(AfiyahError::InputError { message: "Empty sequence".to_string() });
        }

        let (height, width) = sequence[0].dim();
        let mut prediction = Array2::zeros((height, width));

        // Use temporal filters to predict next frame
        for i in 0..height {
            for j in 0..width {
                let mut predicted_value = 0.0;
                let mut total_weight = 0.0;

                for t in 0..sequence.len().min(self.prediction_window) {
                    let frame_idx = sequence.len() - 1 - t;
                    let temporal_weight = self.temporal_filters[[t, 0]];
                    let spatial_weight = self.calculate_spatial_weight(sequence[frame_idx], i, j)?;

                    predicted_value += sequence[frame_idx][[i, j]] * temporal_weight * spatial_weight;
                    total_weight += temporal_weight * spatial_weight;
                }

                if total_weight > 0.0 {
                    prediction[[i, j]] = predicted_value / total_weight;
                } else {
                    prediction[[i, j]] = sequence.last().unwrap()[[i, j]];
                }
            }
        }

        // Update prediction weights based on accuracy
        self.update_prediction_weights(sequence, &prediction)?;

        Ok(prediction)
    }

    /// Calculates prediction error between predicted and actual frame
    pub fn calculate_prediction_error(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = predicted.dim();
        let mut error = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                error[[i, j]] = actual[[i, j]] - predicted[[i, j]];
            }
        }

        Ok(error)
    }

    /// Updates prediction weights based on prediction error
    fn update_prediction_weights(&mut self, sequence: &[Array2<f64>], prediction: &Array2<f64>) -> Result<(), AfiyahError> {
        if sequence.len() < 2 {
            return Ok(());
        }

        let actual = &sequence[sequence.len() - 1];
        let error = self.calculate_prediction_error(prediction, actual)?;

        // Update weights based on prediction error
        for i in 0..self.prediction_weights.nrows() {
            for j in 0..self.prediction_weights.ncols() {
                for t in 0..self.prediction_weights.nslices() {
                    if i < actual.nrows() && j < actual.ncols() {
                        let weight_delta = self.learning_rate * error[[i, j]] * actual[[i, j]];
                        self.prediction_weights[[i, j, t]] += weight_delta;
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculates spatial weight for prediction
    fn calculate_spatial_weight(&self, frame: &Array2<f64>, i: usize, j: usize) -> Result<f64, AfiyahError> {
        let (height, width) = frame.dim();
        let mut spatial_weight = 0.0;
        let mut count = 0;

        // Calculate local spatial correlation
        for di in -1..=1 {
            for dj in -1..=1 {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;

                if ni < height && nj < width {
                    spatial_weight += frame[[ni, nj]];
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(spatial_weight / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Creates prediction weights
    fn create_prediction_weights(prediction_window: usize) -> Result<Array3<f64>, AfiyahError> {
        let height = 64;
        let width = 64;
        let mut weights = Array3::zeros((height, width, prediction_window));

        for i in 0..height {
            for j in 0..width {
                for t in 0..prediction_window {
                    weights[[i, j, t]] = (rand::random::<f64>() - 0.5) * 0.1;
                }
            }
        }

        Ok(weights)
    }

    /// Creates temporal filters
    fn create_temporal_filters() -> Result<Array2<f64>, AfiyahError> {
        let temporal_size = 5;
        let mut filters = Array2::zeros((temporal_size, 1));

        // Create exponential decay temporal filter
        for t in 0..temporal_size {
            filters[[t, 0]] = (-t as f64 * 0.5).exp();
        }

        // Normalize filters
        let sum: f64 = filters.iter().sum();
        for filter in filters.iter_mut() {
            *filter /= sum;
        }

        Ok(filters)
    }

    /// Updates the learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Updates the prediction threshold
    pub fn set_prediction_threshold(&mut self, threshold: f64) {
        self.prediction_threshold = threshold.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_prediction_network_creation() {
        let network = TemporalPredictionNetwork::new();
        assert!(network.is_ok());
    }

    #[test]
    fn test_prediction_next_frame() {
        let mut network = TemporalPredictionNetwork::new().unwrap();
        let sequence = vec![
            Array2::ones((32, 32)),
            Array2::ones((32, 32)),
            Array2::ones((32, 32)),
        ];
        
        let result = network.predict_next_frame(&sequence);
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert_eq!(prediction.dim(), (32, 32));
    }

    #[test]
    fn test_prediction_error_calculation() {
        let network = TemporalPredictionNetwork::new().unwrap();
        let predicted = Array2::ones((32, 32));
        let actual = Array2::ones((32, 32));
        
        let result = network.calculate_prediction_error(&predicted, &actual);
        assert!(result.is_ok());
        
        let error = result.unwrap();
        assert_eq!(error.dim(), (32, 32));
    }

    #[test]
    fn test_learning_rate_update() {
        let mut network = TemporalPredictionNetwork::new().unwrap();
        network.set_learning_rate(0.05);
        assert_eq!(network.learning_rate, 0.05);
    }

    #[test]
    fn test_prediction_threshold_update() {
        let mut network = TemporalPredictionNetwork::new().unwrap();
        network.set_prediction_threshold(0.5);
        assert_eq!(network.prediction_threshold, 0.5);
    }
}
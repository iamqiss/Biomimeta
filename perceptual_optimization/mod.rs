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

//! Perceptual Optimization Module
//! 
//! This module implements sophisticated perceptual optimization algorithms based on
//! human visual system limitations and psychophysical studies.

use ndarray::Array2;
use crate::AfiyahError;
use crate::cortical_processing::CorticalOutput;

// Re-export all sub-modules
pub mod masking_algorithms;
pub mod quality_metrics;
pub mod temporal_prediction_networks;
pub mod foveal_sampling;
pub mod perceptual_error_model;

// Re-export the main types
pub use masking_algorithms::{MaskingAlgorithm, MaskingParams};
pub use quality_metrics::{QualityCalculator, QualityMetrics};
pub use temporal_prediction_networks::TemporalPredictionNetwork;

/// Perceptual optimizer implementing biological perceptual optimization
pub struct PerceptualOptimizer {
    masking_algorithm: MaskingAlgorithm,
    quality_calculator: QualityCalculator,
    temporal_predictor: TemporalPredictionNetwork,
    optimization_params: OptimizationParams,
}

/// Optimization parameters for perceptual processing
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    pub masking_enabled: bool,
    pub quality_threshold: f64,
    pub prediction_enabled: bool,
    pub optimization_strength: f64,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            masking_enabled: true,
            quality_threshold: 0.8,
            prediction_enabled: true,
            optimization_strength: 0.9,
        }
    }
}

impl PerceptualOptimizer {
    /// Creates a new perceptual optimizer
    pub fn new() -> Result<Self, AfiyahError> {
        let masking_algorithm = MaskingAlgorithm::new()?;
        let quality_calculator = QualityCalculator::new()?;
        let temporal_predictor = TemporalPredictionNetwork::new()?;
        let optimization_params = OptimizationParams::default();

        Ok(Self {
            masking_algorithm,
            quality_calculator,
            temporal_predictor,
            optimization_params,
        })
    }
    
    /// Optimizes cortical output using perceptual optimization
    pub fn optimize(&mut self, input: &CorticalOutput) -> Result<CorticalOutput, AfiyahError> {
        // Convert cortical output to 2D array for processing
        let (height, width) = (32, 32); // Assume square output
        let mut data_2d = Array2::zeros((height, width));
        
        // Fill 2D array with cortical data
        for (i, &value) in input.data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                data_2d[[row, col]] = value;
            }
        }

        let mut optimized_data = data_2d.clone();

        // Apply masking if enabled
        if self.optimization_params.masking_enabled {
            optimized_data = self.masking_algorithm.apply_masking(&optimized_data)?;
        }

        // Apply temporal prediction if enabled
        if self.optimization_params.prediction_enabled {
            let sequence = vec![data_2d.clone()];
            let prediction = self.temporal_predictor.predict_next_frame(&sequence)?;
            
            // Blend original and predicted data
            for i in 0..height {
                for j in 0..width {
                    optimized_data[[i, j]] = optimized_data[[i, j]] * (1.0 - self.optimization_params.optimization_strength) +
                                           prediction[[i, j]] * self.optimization_params.optimization_strength;
                }
            }
        }

        // Convert back to 1D vector
        let mut optimized_vector = Vec::new();
        for i in 0..height {
            for j in 0..width {
                optimized_vector.push(optimized_data[[i, j]]);
            }
        }

        Ok(CorticalOutput {
            data: optimized_vector,
        })
    }
    
    /// Calculates VMAF score
    pub fn calculate_vmaf(&self, input: &crate::VisualInput, output: &CorticalOutput) -> Result<f64, AfiyahError> {
        // Convert input to 2D array
        let (height, width) = input.spatial_resolution;
        let mut input_2d = Array2::zeros((height, width));
        
        for (i, &value) in input.luminance_data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                input_2d[[row, col]] = value;
            }
        }

        // Convert output to 2D array
        let mut output_2d = Array2::zeros((height, width));
        for (i, &value) in output.data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                output_2d[[row, col]] = value;
            }
        }

        // Calculate VMAF
        let quality_metrics = self.quality_calculator.calculate_quality(&input_2d, &output_2d)?;
        Ok(quality_metrics.vmaf)
    }
    
    /// Calculates PSNR score
    pub fn calculate_psnr(&self, input: &crate::VisualInput, output: &CorticalOutput) -> Result<f64, AfiyahError> {
        // Convert input to 2D array
        let (height, width) = input.spatial_resolution;
        let mut input_2d = Array2::zeros((height, width));
        
        for (i, &value) in input.luminance_data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                input_2d[[row, col]] = value;
            }
        }

        // Convert output to 2D array
        let mut output_2d = Array2::zeros((height, width));
        for (i, &value) in output.data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                output_2d[[row, col]] = value;
            }
        }

        // Calculate PSNR
        let quality_metrics = self.quality_calculator.calculate_quality(&input_2d, &output_2d)?;
        Ok(quality_metrics.psnr)
    }
    
    /// Calculates SSIM score
    pub fn calculate_ssim(&self, input: &crate::VisualInput, output: &CorticalOutput) -> Result<f64, AfiyahError> {
        // Convert input to 2D array
        let (height, width) = input.spatial_resolution;
        let mut input_2d = Array2::zeros((height, width));
        
        for (i, &value) in input.luminance_data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                input_2d[[row, col]] = value;
            }
        }

        // Convert output to 2D array
        let mut output_2d = Array2::zeros((height, width));
        for (i, &value) in output.data.iter().enumerate() {
            if i < height * width {
                let row = i / width;
                let col = i % width;
                output_2d[[row, col]] = value;
            }
        }

        // Calculate SSIM
        let quality_metrics = self.quality_calculator.calculate_quality(&input_2d, &output_2d)?;
        Ok(quality_metrics.ssim)
    }

    /// Updates the optimization parameters
    pub fn update_params(&mut self, params: OptimizationParams) {
        self.optimization_params = params;
    }

    /// Gets current optimization parameters
    pub fn get_params(&self) -> &OptimizationParams {
        &self.optimization_params
    }
}
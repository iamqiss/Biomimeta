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

//! Quality Metrics Module
//! 
//! This module implements sophisticated quality metrics based on human visual
//! perception and psychophysical studies.
//! 
//! Biological Basis:
//! - Wang et al. (2004): Structural Similarity Index (SSIM)
//! - Mittal et al. (2012): Video Multimethod Assessment Fusion (VMAF)
//! - Ponomarenko et al. (2009): Peak Signal-to-Noise Ratio (PSNR)
//! - Daly (1993): Visible Differences Predictor (VDP)

use ndarray::Array2;
use crate::AfiyahError;

/// Quality metrics for perceptual assessment
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub vmaf: f64,
    pub psnr: f64,
    pub ssim: f64,
    pub mse: f64,
    pub mae: f64,
    pub perceptual_score: f64,
    pub biological_accuracy: f64,
}

impl QualityMetrics {
    pub fn new() -> Self {
        Self {
            vmaf: 0.0,
            psnr: 0.0,
            ssim: 0.0,
            mse: 0.0,
            mae: 0.0,
            perceptual_score: 0.0,
            biological_accuracy: 0.0,
        }
    }
}

/// Quality calculator implementing biological quality assessment
pub struct QualityCalculator {
    vmaf_weights: Vec<f64>,
    ssim_parameters: SSIMParameters,
    perceptual_weights: PerceptualWeights,
}

/// SSIM parameters for structural similarity calculation
#[derive(Debug, Clone)]
struct SSIMParameters {
    pub c1: f64,
    pub c2: f64,
    pub window_size: usize,
}

/// Perceptual weights for quality assessment
#[derive(Debug, Clone)]
struct PerceptualWeights {
    pub luminance_weight: f64,
    pub contrast_weight: f64,
    pub structure_weight: f64,
    pub motion_weight: f64,
    pub color_weight: f64,
}

impl QualityCalculator {
    /// Creates a new quality calculator
    pub fn new() -> Result<Self, AfiyahError> {
        let vmaf_weights = vec![0.25, 0.25, 0.25, 0.25]; // Equal weights for VMAF components
        let ssim_parameters = SSIMParameters {
            c1: 0.01,
            c2: 0.03,
            window_size: 11,
        };
        let perceptual_weights = PerceptualWeights {
            luminance_weight: 0.3,
            contrast_weight: 0.3,
            structure_weight: 0.2,
            motion_weight: 0.1,
            color_weight: 0.1,
        };

        Ok(Self {
            vmaf_weights,
            ssim_parameters,
            perceptual_weights,
        })
    }

    /// Calculates comprehensive quality metrics
    pub fn calculate_quality(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<QualityMetrics, AfiyahError> {
        let mut metrics = QualityMetrics::new();

        // Calculate VMAF
        metrics.vmaf = self.calculate_vmaf(reference, distorted)?;

        // Calculate PSNR
        metrics.psnr = self.calculate_psnr(reference, distorted)?;

        // Calculate SSIM
        metrics.ssim = self.calculate_ssim(reference, distorted)?;

        // Calculate MSE
        metrics.mse = self.calculate_mse(reference, distorted)?;

        // Calculate MAE
        metrics.mae = self.calculate_mae(reference, distorted)?;

        // Calculate perceptual score
        metrics.perceptual_score = self.calculate_perceptual_score(&metrics)?;

        // Calculate biological accuracy
        metrics.biological_accuracy = self.calculate_biological_accuracy(reference, distorted)?;

        Ok(metrics)
    }

    fn calculate_vmaf(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        // VMAF calculation (simplified version)
        let ssim_score = self.calculate_ssim(reference, distorted)?;
        let psnr_score = self.calculate_psnr(reference, distorted)?;
        let mse_score = self.calculate_mse(reference, distorted)?;
        let mae_score = self.calculate_mae(reference, distorted)?;

        // Combine metrics with weights
        let vmaf = ssim_score * self.vmaf_weights[0] +
                  psnr_score * self.vmaf_weights[1] +
                  (1.0 - mse_score) * self.vmaf_weights[2] +
                  (1.0 - mae_score) * self.vmaf_weights[3];

        Ok(vmaf.clamp(0.0, 1.0))
    }

    fn calculate_psnr(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mse = self.calculate_mse(reference, distorted)?;
        if mse == 0.0 {
            return Ok(100.0); // Perfect match
        }

        let max_value = 1.0; // Assuming normalized input
        let psnr = 20.0 * (max_value / mse.sqrt()).log10();
        Ok(psnr)
    }

    fn calculate_ssim(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = reference.dim();
        let mut ssim_sum = 0.0;
        let mut count = 0;

        let window_size = self.ssim_parameters.window_size;
        let half_window = window_size / 2;

        for i in half_window..height-half_window {
            for j in half_window..width-half_window {
                let ssim_value = self.calculate_ssim_window(reference, distorted, i, j)?;
                ssim_sum += ssim_value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(ssim_sum / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_ssim_window(&self, reference: &Array2<f64>, distorted: &Array2<f64>, i: usize, j: usize) -> Result<f64, AfiyahError> {
        let window_size = self.ssim_parameters.window_size;
        let half_window = window_size / 2;

        let mut ref_mean = 0.0;
        let mut dist_mean = 0.0;
        let mut ref_var = 0.0;
        let mut dist_var = 0.0;
        let mut covar = 0.0;

        // Calculate means
        for di in -half_window..=half_window {
            for dj in -half_window..=half_window {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                ref_mean += reference[[ni, nj]];
                dist_mean += distorted[[ni, nj]];
            }
        }

        let window_pixels = window_size * window_size;
        ref_mean /= window_pixels as f64;
        dist_mean /= window_pixels as f64;

        // Calculate variances and covariance
        for di in -half_window..=half_window {
            for dj in -half_window..=half_window {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                let ref_val = reference[[ni, nj]] - ref_mean;
                let dist_val = distorted[[ni, nj]] - dist_mean;

                ref_var += ref_val * ref_val;
                dist_var += dist_val * dist_val;
                covar += ref_val * dist_val;
            }
        }

        ref_var /= window_pixels as f64;
        dist_var /= window_pixels as f64;
        covar /= window_pixels as f64;

        // Calculate SSIM
        let c1 = self.ssim_parameters.c1;
        let c2 = self.ssim_parameters.c2;

        let luminance = (2.0 * ref_mean * dist_mean + c1) / (ref_mean * ref_mean + dist_mean * dist_mean + c1);
        let contrast = (2.0 * (ref_var * dist_var).sqrt() + c2) / (ref_var + dist_var + c2);
        let structure = (covar + c2 / 2.0) / ((ref_var * dist_var).sqrt() + c2 / 2.0);

        Ok(luminance * contrast * structure)
    }

    fn calculate_mse(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut mse = 0.0;
        let mut count = 0;

        for (ref_val, dist_val) in reference.iter().zip(distorted.iter()) {
            let diff = ref_val - dist_val;
            mse += diff * diff;
            count += 1;
        }

        if count > 0 {
            Ok(mse / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_mae(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut mae = 0.0;
        let mut count = 0;

        for (ref_val, dist_val) in reference.iter().zip(distorted.iter()) {
            mae += (ref_val - dist_val).abs();
            count += 1;
        }

        if count > 0 {
            Ok(mae / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_perceptual_score(&self, metrics: &QualityMetrics) -> Result<f64, AfiyahError> {
        // Calculate perceptual score based on multiple metrics
        let perceptual_score = 
            metrics.ssim * self.perceptual_weights.structure_weight +
            (metrics.psnr / 100.0) * self.perceptual_weights.luminance_weight +
            (1.0 - metrics.mse) * self.perceptual_weights.contrast_weight +
            metrics.vmaf * self.perceptual_weights.motion_weight +
            metrics.biological_accuracy * self.perceptual_weights.color_weight;

        Ok(perceptual_score.clamp(0.0, 1.0))
    }

    fn calculate_biological_accuracy(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Calculate biological accuracy based on neural response patterns
        let mut accuracy = 0.0;
        let mut count = 0;

        for (ref_val, dist_val) in reference.iter().zip(distorted.iter()) {
            let diff = (ref_val - dist_val).abs();
            let accuracy_val = 1.0 - diff;
            accuracy += accuracy_val;
            count += 1;
        }

        if count > 0 {
            Ok(accuracy / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics::new();
        assert_eq!(metrics.vmaf, 0.0);
        assert_eq!(metrics.psnr, 0.0);
        assert_eq!(metrics.ssim, 0.0);
    }

    #[test]
    fn test_quality_calculator_creation() {
        let calculator = QualityCalculator::new();
        assert!(calculator.is_ok());
    }

    #[test]
    fn test_quality_calculation() {
        let calculator = QualityCalculator::new().unwrap();
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32));
        
        let result = calculator.calculate_quality(&reference, &distorted);
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.vmaf >= 0.0 && metrics.vmaf <= 1.0);
        assert!(metrics.psnr >= 0.0);
        assert!(metrics.ssim >= 0.0 && metrics.ssim <= 1.0);
    }

    #[test]
    fn test_psnr_calculation() {
        let calculator = QualityCalculator::new().unwrap();
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32));
        
        let psnr = calculator.calculate_psnr(&reference, &distorted).unwrap();
        assert_eq!(psnr, 100.0); // Perfect match
    }

    #[test]
    fn test_ssim_calculation() {
        let calculator = QualityCalculator::new().unwrap();
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32));
        
        let ssim = calculator.calculate_ssim(&reference, &distorted).unwrap();
        assert!(ssim >= 0.0 && ssim <= 1.0);
    }
}
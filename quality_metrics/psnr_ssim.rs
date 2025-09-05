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

//! PSNR and SSIM Quality Metrics
//! 
//! This module implements PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
//! calculations with biological enhancements for Afiyah's quality assessment system.

use ndarray::{Array2, ArrayView2, s};
use crate::AfiyahError;

/// PSNR calculator with biological enhancements
pub struct PSNRCalculator {
    max_value: f64,
    biological_weighting: bool,
    foveal_prioritization: bool,
}

impl PSNRCalculator {
    /// Creates a new PSNR calculator
    pub fn new(max_value: f64, biological_weighting: bool) -> Self {
        Self {
            max_value,
            biological_weighting,
            foveal_prioritization: true,
        }
    }

    /// Calculates PSNR between reference and distorted images
    pub fn calculate_psnr(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mse = self.calculate_mse(reference, distorted)?;
        
        if mse == 0.0 {
            return Ok(f64::INFINITY);
        }
        
        let psnr = 20.0 * self.max_value.log10() - 10.0 * mse.log10();
        
        if self.biological_weighting {
            Ok(self.apply_biological_weighting(psnr, reference, distorted)?)
        } else {
            Ok(psnr)
        }
    }

    /// Calculates Mean Squared Error
    fn calculate_mse(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        if reference.dim() != distorted.dim() {
            return Err(AfiyahError::Mathematical { 
                message: "Reference and distorted images must have the same dimensions".to_string() 
            });
        }

        let diff = reference - distorted;
        let squared_diff = diff.mapv(|x| x * x);
        let mse = squared_diff.sum() / (reference.len() as f64);
        
        Ok(mse)
    }

    /// Applies biological weighting to PSNR score
    fn apply_biological_weighting(&self, psnr: f64, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        if !self.foveal_prioritization {
            return Ok(psnr);
        }

        let foveal_weight = self.calculate_foveal_weight(reference, distorted)?;
        let biological_factor = self.calculate_biological_factor(reference, distorted)?;
        
        Ok(psnr * foveal_weight * biological_factor)
    }

    /// Calculates foveal weight based on center region importance
    fn calculate_foveal_weight(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = reference.dim();
        let center_y = height / 2;
        let center_x = width / 2;
        let foveal_radius = (height.min(width) / 4).max(1);

        let mut foveal_mse = 0.0;
        let mut foveal_count = 0;
        let mut peripheral_mse = 0.0;
        let mut peripheral_count = 0;

        for i in 0..height {
            for j in 0..width {
                let distance = ((i as i32 - center_y as i32).powi(2) + (j as i32 - center_x as i32).powi(2)).sqrt();
                
                if distance <= foveal_radius as f64 {
                    let diff = reference[[i, j]] - distorted[[i, j]];
                    foveal_mse += diff * diff;
                    foveal_count += 1;
                } else {
                    let diff = reference[[i, j]] - distorted[[i, j]];
                    peripheral_mse += diff * diff;
                    peripheral_count += 1;
                }
            }
        }

        let foveal_avg_mse = if foveal_count > 0 { foveal_mse / foveal_count as f64 } else { 0.0 };
        let peripheral_avg_mse = if peripheral_count > 0 { peripheral_mse / peripheral_count as f64 } else { 0.0 };

        // Weight foveal region more heavily
        let foveal_weight = if peripheral_avg_mse > 0.0 {
            1.0 + (foveal_avg_mse / peripheral_avg_mse) * 0.5
        } else {
            1.0
        };

        Ok(foveal_weight.clamp(0.5, 2.0))
    }

    /// Calculates biological factor based on human visual system characteristics
    fn calculate_biological_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Calculate biological factor based on contrast sensitivity and visual masking
        let contrast_factor = self.calculate_contrast_factor(reference, distorted)?;
        let masking_factor = self.calculate_masking_factor(reference, distorted)?;
        
        Ok(contrast_factor * masking_factor)
    }

    /// Calculates contrast sensitivity factor
    fn calculate_contrast_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let ref_contrast = self.calculate_image_contrast(reference)?;
        let dist_contrast = self.calculate_image_contrast(distorted)?;
        
        let contrast_ratio = if ref_contrast > 0.0 {
            dist_contrast / ref_contrast
        } else {
            1.0
        };
        
        Ok(contrast_ratio.clamp(0.5, 1.5))
    }

    /// Calculates visual masking factor
    fn calculate_masking_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let ref_complexity = self.calculate_image_complexity(reference)?;
        let dist_complexity = self.calculate_image_complexity(distorted)?;
        
        let complexity_ratio = if ref_complexity > 0.0 {
            dist_complexity / ref_complexity
        } else {
            1.0
        };
        
        // Higher complexity provides more masking
        Ok(complexity_ratio.clamp(0.8, 1.2))
    }

    /// Calculates image contrast
    fn calculate_image_contrast(&self, image: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mean = image.sum() / (image.len() as f64);
        let variance = image.mapv(|x| (x - mean).powi(2)).sum() / (image.len() as f64);
        Ok(variance.sqrt())
    }

    /// Calculates image complexity
    fn calculate_image_complexity(&self, image: &Array2<f64>) -> Result<f64, AfiyahError> {
        let gradient = self.calculate_gradient_magnitude(image)?;
        Ok(gradient.sum() / (gradient.len() as f64))
    }

    /// Calculates gradient magnitude
    fn calculate_gradient_magnitude(&self, image: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = image.dim();
        let mut gradient = Array2::zeros((height, width));
        
        for i in 1..height-1 {
            for j in 1..width-1 {
                let gx = image[[i, j+1]] - image[[i, j-1]];
                let gy = image[[i+1, j]] - image[[i-1, j]];
                gradient[[i, j]] = (gx.powi(2) + gy.powi(2)).sqrt();
            }
        }
        
        Ok(gradient)
    }
}

/// SSIM calculator with biological enhancements
pub struct SSIMCalculator {
    k1: f64,
    k2: f64,
    window_size: usize,
    biological_weighting: bool,
    foveal_prioritization: bool,
}

impl SSIMCalculator {
    /// Creates a new SSIM calculator
    pub fn new(biological_weighting: bool) -> Self {
        Self {
            k1: 0.01,
            k2: 0.03,
            window_size: 11,
            biological_weighting,
            foveal_prioritization: true,
        }
    }

    /// Calculates SSIM between reference and distorted images
    pub fn calculate_ssim(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        if reference.dim() != distorted.dim() {
            return Err(AfiyahError::Mathematical { 
                message: "Reference and distorted images must have the same dimensions".to_string() 
            });
        }

        let (height, width) = reference.dim();
        let mut ssim_sum = 0.0;
        let mut window_count = 0;

        for i in 0..=(height - self.window_size) {
            for j in 0..=(width - self.window_size) {
                let ref_window = reference.slice(s![i..i+self.window_size, j..j+self.window_size]);
                let dist_window = distorted.slice(s![i..i+self.window_size, j..j+self.window_size]);
                
                let window_ssim = self.calculate_window_ssim(&ref_window, &dist_window)?;
                ssim_sum += window_ssim;
                window_count += 1;
            }
        }

        let ssim = if window_count > 0 { ssim_sum / window_count as f64 } else { 0.0 };

        if self.biological_weighting {
            Ok(self.apply_biological_weighting(ssim, reference, distorted)?)
        } else {
            Ok(ssim)
        }
    }

    /// Calculates SSIM for a single window
    fn calculate_window_ssim(&self, ref_window: &ArrayView2<f64>, dist_window: &ArrayView2<f64>) -> Result<f64, AfiyahError> {
        let ref_mean = ref_window.sum() / (ref_window.len() as f64);
        let dist_mean = dist_window.sum() / (dist_window.len() as f64);

        let ref_var = ref_window.mapv(|x| (x - ref_mean).powi(2)).sum() / (ref_window.len() as f64);
        let dist_var = dist_window.mapv(|x| (x - dist_mean).powi(2)).sum() / (dist_window.len() as f64);

        let cross_cov = ref_window.iter()
            .zip(dist_window.iter())
            .map(|(r, d)| (r - ref_mean) * (d - dist_mean))
            .sum::<f64>() / (ref_window.len() as f64);

        let c1 = (self.k1 * 1.0).powi(2);
        let c2 = (self.k2 * 1.0).powi(2);

        let luminance = (2.0 * ref_mean * dist_mean + c1) / (ref_mean.powi(2) + dist_mean.powi(2) + c1);
        let contrast = (2.0 * (ref_var * dist_var).sqrt() + c2) / (ref_var + dist_var + c2);
        let structure = (cross_cov + c2 / 2.0) / ((ref_var * dist_var).sqrt() + c2 / 2.0);

        Ok(luminance * contrast * structure)
    }

    /// Applies biological weighting to SSIM score
    fn apply_biological_weighting(&self, ssim: f64, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        if !self.foveal_prioritization {
            return Ok(ssim);
        }

        let foveal_weight = self.calculate_foveal_weight(reference, distorted)?;
        let biological_factor = self.calculate_biological_factor(reference, distorted)?;
        
        Ok(ssim * foveal_weight * biological_factor)
    }

    /// Calculates foveal weight for SSIM
    fn calculate_foveal_weight(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = reference.dim();
        let center_y = height / 2;
        let center_x = width / 2;
        let foveal_radius = (height.min(width) / 4).max(1);

        let mut foveal_ssim = 0.0;
        let mut foveal_count = 0;
        let mut peripheral_ssim = 0.0;
        let mut peripheral_count = 0;

        for i in 0..=(height - self.window_size) {
            for j in 0..=(width - self.window_size) {
                let window_center_y = i + self.window_size / 2;
                let window_center_x = j + self.window_size / 2;
                
                let distance = ((window_center_y as i32 - center_y as i32).powi(2) + 
                               (window_center_x as i32 - center_x as i32).powi(2)).sqrt();
                
                let ref_window = reference.slice(s![i..i+self.window_size, j..j+self.window_size]);
                let dist_window = distorted.slice(s![i..i+self.window_size, j..j+self.window_size]);
                
                let window_ssim = self.calculate_window_ssim(&ref_window, &dist_window)?;
                
                if distance <= foveal_radius as f64 {
                    foveal_ssim += window_ssim;
                    foveal_count += 1;
                } else {
                    peripheral_ssim += window_ssim;
                    peripheral_count += 1;
                }
            }
        }

        let foveal_avg = if foveal_count > 0 { foveal_ssim / foveal_count as f64 } else { 0.0 };
        let peripheral_avg = if peripheral_count > 0 { peripheral_ssim / peripheral_count as f64 } else { 0.0 };

        // Weight foveal region more heavily
        let foveal_weight = if peripheral_avg > 0.0 {
            1.0 + (foveal_avg / peripheral_avg) * 0.3
        } else {
            1.0
        };

        Ok(foveal_weight.clamp(0.7, 1.5))
    }

    /// Calculates biological factor for SSIM
    fn calculate_biological_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let structural_factor = self.calculate_structural_factor(reference, distorted)?;
        let perceptual_factor = self.calculate_perceptual_factor(reference, distorted)?;
        
        Ok(structural_factor * perceptual_factor)
    }

    /// Calculates structural preservation factor
    fn calculate_structural_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let ref_edges = self.calculate_edge_strength(reference)?;
        let dist_edges = self.calculate_edge_strength(distorted)?;
        
        let edge_preservation = if ref_edges > 0.0 {
            dist_edges / ref_edges
        } else {
            1.0
        };
        
        Ok(edge_preservation.clamp(0.5, 1.5))
    }

    /// Calculates perceptual factor
    fn calculate_perceptual_factor(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let ref_texture = self.calculate_texture_complexity(reference)?;
        let dist_texture = self.calculate_texture_complexity(distorted)?;
        
        let texture_preservation = if ref_texture > 0.0 {
            dist_texture / ref_texture
        } else {
            1.0
        };
        
        Ok(texture_preservation.clamp(0.8, 1.2))
    }

    /// Calculates edge strength
    fn calculate_edge_strength(&self, image: &Array2<f64>) -> Result<f64, AfiyahError> {
        let gradient = self.calculate_gradient_magnitude(image)?;
        Ok(gradient.sum() / (gradient.len() as f64))
    }

    /// Calculates texture complexity
    fn calculate_texture_complexity(&self, image: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = image.dim();
        let mut complexity = 0.0;
        let mut count = 0;

        for i in 1..height-1 {
            for j in 1..width-1 {
                let center = image[[i, j]];
                let mut variance = 0.0;
                let mut neighbor_count = 0.0;

                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di != 0 || dj != 0 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            let neighbor = image[[ni, nj]];
                            variance += (center - neighbor).powi(2);
                            neighbor_count += 1.0;
                        }
                    }
                }

                complexity += variance / neighbor_count;
                count += 1;
            }
        }

        Ok(if count > 0 { complexity / count as f64 } else { 0.0 })
    }

    /// Calculates gradient magnitude
    fn calculate_gradient_magnitude(&self, image: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = image.dim();
        let mut gradient = Array2::zeros((height, width));
        
        for i in 1..height-1 {
            for j in 1..width-1 {
                let gx = image[[i, j+1]] - image[[i, j-1]];
                let gy = image[[i+1, j]] - image[[i-1, j]];
                gradient[[i, j]] = (gx.powi(2) + gy.powi(2)).sqrt();
            }
        }
        
        Ok(gradient)
    }
}

/// MS-SSIM calculator for multi-scale structural similarity
pub struct MSSSIMCalculator {
    scales: Vec<usize>,
    ssim_calculator: SSIMCalculator,
}

impl MSSSIMCalculator {
    /// Creates a new MS-SSIM calculator
    pub fn new(scales: Vec<usize>) -> Self {
        Self {
            scales,
            ssim_calculator: SSIMCalculator::new(true),
        }
    }

    /// Calculates MS-SSIM between reference and distorted images
    pub fn calculate_ms_ssim(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut ms_ssim = 1.0;
        let mut ref_current = reference.clone();
        let mut dist_current = distorted.clone();

        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            let scale_ssim = self.ssim_calculator.calculate_ssim(&ref_current, &dist_current)?;
            
            if scale_idx == self.scales.len() - 1 {
                // Last scale uses full SSIM
                ms_ssim *= scale_ssim;
            } else {
                // Other scales use only luminance and contrast
                ms_ssim *= self.calculate_luminance_contrast_ssim(&ref_current, &dist_current)?;
            }

            // Downsample for next scale
            if scale_idx < self.scales.len() - 1 {
                ref_current = self.downsample(&ref_current)?;
                dist_current = self.downsample(&dist_current)?;
            }
        }

        Ok(ms_ssim)
    }

    /// Calculates luminance and contrast SSIM
    fn calculate_luminance_contrast_ssim(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<f64, AfiyahError> {
        let ref_mean = reference.sum() / (reference.len() as f64);
        let dist_mean = distorted.sum() / (distorted.len() as f64);

        let ref_var = reference.mapv(|x| (x - ref_mean).powi(2)).sum() / (reference.len() as f64);
        let dist_var = distorted.mapv(|x| (x - dist_mean).powi(2)).sum() / (distorted.len() as f64);

        let cross_cov = reference.iter()
            .zip(distorted.iter())
            .map(|(r, d)| (r - ref_mean) * (d - dist_mean))
            .sum::<f64>() / (reference.len() as f64);

        let c1 = (0.01 * 1.0).powi(2);
        let c2 = (0.03 * 1.0).powi(2);

        let luminance = (2.0 * ref_mean * dist_mean + c1) / (ref_mean.powi(2) + dist_mean.powi(2) + c1);
        let contrast = (2.0 * (ref_var * dist_var).sqrt() + c2) / (ref_var + dist_var + c2);

        Ok(luminance * contrast)
    }

    /// Downsamples image by factor of 2
    fn downsample(&self, image: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = image.dim();
        let new_height = height / 2;
        let new_width = width / 2;
        
        let mut downsampled = Array2::zeros((new_height, new_width));
        
        for i in 0..new_height {
            for j in 0..new_width {
                let mut sum = 0.0;
                let mut count = 0;
                
                for di in 0..2 {
                    for dj in 0..2 {
                        let ni = i * 2 + di;
                        let nj = j * 2 + dj;
                        if ni < height && nj < width {
                            sum += image[[ni, nj]];
                            count += 1;
                        }
                    }
                }
                
                downsampled[[i, j]] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }
        
        Ok(downsampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_calculator_creation() {
        let calculator = PSNRCalculator::new(1.0, true);
        assert_eq!(calculator.max_value, 1.0);
        assert!(calculator.biological_weighting);
    }

    #[test]
    fn test_psnr_calculation() {
        let calculator = PSNRCalculator::new(1.0, true);
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32)) * 0.9;
        
        let psnr = calculator.calculate_psnr(&reference, &distorted);
        assert!(psnr.is_ok());
        
        let score = psnr.unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_ssim_calculator_creation() {
        let calculator = SSIMCalculator::new(true);
        assert!(calculator.biological_weighting);
    }

    #[test]
    fn test_ssim_calculation() {
        let calculator = SSIMCalculator::new(true);
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32)) * 0.9;
        
        let ssim = calculator.calculate_ssim(&reference, &distorted);
        assert!(ssim.is_ok());
        
        let score = ssim.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_ms_ssim_calculator_creation() {
        let scales = vec![1, 2, 4];
        let calculator = MSSSIMCalculator::new(scales);
        assert_eq!(calculator.scales.len(), 3);
    }

    #[test]
    fn test_ms_ssim_calculation() {
        let scales = vec![1, 2, 4];
        let calculator = MSSSIMCalculator::new(scales);
        let reference = Array2::ones((64, 64));
        let distorted = Array2::ones((64, 64)) * 0.9;
        
        let ms_ssim = calculator.calculate_ms_ssim(&reference, &distorted);
        assert!(ms_ssim.is_ok());
        
        let score = ms_ssim.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
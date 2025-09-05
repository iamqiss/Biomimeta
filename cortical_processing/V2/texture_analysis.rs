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

//! Texture Analysis Implementation for V2 Secondary Visual Cortex

use crate::AfiyahError;
use ndarray::Array2;

/// Texture analyzer for local pattern processing
pub struct TextureAnalyzer {
    pub window_size: usize,
    pub config: TextureAnalysisConfig,
}

/// Texture analysis configuration
#[derive(Debug, Clone)]
pub struct TextureAnalysisConfig {
    pub window_size: usize,
    pub orientation_count: usize,
    pub spatial_frequency_count: usize,
    pub regularity_threshold: f64,
    pub contrast_threshold: f64,
}

/// Texture type classification
#[derive(Debug, Clone, PartialEq)]
pub enum TextureType {
    Regular,
    Irregular,
    Periodic,
    Random,
    Oriented,
}

/// Texture feature descriptor
#[derive(Debug, Clone)]
pub struct TextureFeature {
    pub location: (usize, usize),
    pub texture_type: TextureType,
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub regularity: f64,
    pub contrast: f64,
    pub energy: f64,
}

impl TextureAnalyzer {
    /// Creates a new texture analyzer
    pub fn new(config: &TextureAnalysisConfig) -> Result<Self, AfiyahError> {
        Ok(Self {
            window_size: config.window_size,
            config: config.clone(),
        })
    }

    /// Analyzes texture in input image
    pub fn analyze_texture(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut texture_map = Array2::zeros((height, width));
        
        for h in 0..height {
            for w in 0..width {
                let texture_value = self.analyze_local_texture(input, h, w)?;
                texture_map[[h, w]] = texture_value;
            }
        }
        
        Ok(texture_map)
    }

    /// Extracts texture feature at specific location
    pub fn extract_texture_feature(&self, input: &Array2<f64>, h: usize, w: usize) -> Result<TextureFeature, AfiyahError> {
        let (height, width) = input.dim();
        let half_window = self.window_size / 2;
        
        let mut window_values = Vec::new();
        for dh in -(half_window as i32)..=(half_window as i32) {
            for dw in -(half_window as i32)..=(half_window as i32) {
                let nh = (h as i32 + dh) as usize;
                let nw = (w as i32 + dw) as usize;
                
                if nh < height && nw < width {
                    window_values.push(input[[nh, nw]]);
                }
            }
        }
        
        let regularity = self.calculate_regularity(&window_values)?;
        let contrast = self.calculate_contrast(&window_values)?;
        let energy = self.calculate_energy(&window_values)?;
        let texture_type = self.classify_texture_type(regularity, contrast, energy);
        
        Ok(TextureFeature {
            location: (h, w),
            texture_type,
            orientation: 0.0,
            spatial_frequency: 1.0,
            regularity,
            contrast,
            energy,
        })
    }

    fn analyze_local_texture(&self, input: &Array2<f64>, h: usize, w: usize) -> Result<f64, AfiyahError> {
        let (height, width) = input.dim();
        let half_window = self.window_size / 2;
        
        let mut window_values = Vec::new();
        for dh in -(half_window as i32)..=(half_window as i32) {
            for dw in -(half_window as i32)..=(half_window as i32) {
                let nh = (h as i32 + dh) as usize;
                let nw = (w as i32 + dw) as usize;
                
                if nh < height && nw < width {
                    window_values.push(input[[nh, nw]]);
                }
            }
        }
        
        let regularity = self.calculate_regularity(&window_values)?;
        Ok(regularity)
    }

    fn calculate_regularity(&self, values: &[f64]) -> Result<f64, AfiyahError> {
        if values.len() < 2 {
            return Ok(0.0);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let regularity = 1.0 / (1.0 + variance.sqrt());
        Ok(regularity)
    }

    fn calculate_contrast(&self, values: &[f64]) -> Result<f64, AfiyahError> {
        if values.is_empty() {
            return Ok(0.0);
        }
        
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let contrast = if max_val > min_val {
            (max_val - min_val) / (max_val + min_val)
        } else {
            0.0
        };
        
        Ok(contrast)
    }

    fn calculate_energy(&self, values: &[f64]) -> Result<f64, AfiyahError> {
        let energy = values.iter().map(|&x| x * x).sum::<f64>() / values.len() as f64;
        Ok(energy)
    }

    fn classify_texture_type(&self, regularity: f64, contrast: f64, energy: f64) -> TextureType {
        if regularity > 0.8 {
            TextureType::Regular
        } else if regularity > 0.6 {
            TextureType::Periodic
        } else if contrast > 0.7 {
            TextureType::Oriented
        } else if regularity < 0.3 {
            TextureType::Random
        } else {
            TextureType::Irregular
        }
    }
}
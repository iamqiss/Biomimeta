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

//! Content Analyzer for Real-Time Adaptation
//! 
//! Analyzes visual content characteristics to inform adaptive compression decisions.
//! 
//! Biological Basis:
//! - Marr (1982): Vision - Computational approach to understanding visual processing
//! - Olshausen & Field (1996): Sparse coding in visual cortex
//! - Serre et al. (2007): Hierarchical models of visual cortex
//! - Yamins et al. (2014): Deep neural networks and visual cortex

use ndarray::{Array1, Array2, Array3};
use std::collections::VecDeque;
use std::time::Duration;
use crate::AfiyahError;

/// Content features extracted from visual input
#[derive(Debug, Clone)]
pub struct ContentFeatures {
    pub spatial_complexity: f64,
    pub temporal_variance: f64,
    pub edge_density: f64,
    pub texture_complexity: f64,
    pub color_diversity: f64,
    pub motion_magnitude: f64,
    pub luminance_variance: f64,
    pub frequency_spectrum: Array1<f64>,
    pub orientation_histogram: Array1<f64>,
    pub spatial_frequencies: Array1<f64>,
}

/// Content complexity assessment
#[derive(Debug, Clone)]
pub struct ContentComplexity {
    pub overall_complexity: f64,
    pub spatial_complexity: f64,
    pub temporal_complexity: f64,
    pub perceptual_complexity: f64,
    pub compression_difficulty: f64,
    pub biological_processing_load: f64,
    pub variance: f64,
    pub stability: f64,
}

/// Content analyzer configuration
#[derive(Debug, Clone)]
pub struct ContentAnalyzerConfig {
    pub analysis_window: Duration,
    pub spatial_scale: f64,
    pub temporal_scale: f64,
    pub frequency_bands: usize,
    pub orientation_bands: usize,
    pub complexity_threshold: f64,
    pub stability_threshold: f64,
}

impl Default for ContentAnalyzerConfig {
    fn default() -> Self {
        Self {
            analysis_window: Duration::from_millis(100),
            spatial_scale: 1.0,
            temporal_scale: 1.0,
            frequency_bands: 8,
            orientation_bands: 8,
            complexity_threshold: 0.5,
            stability_threshold: 0.1,
        }
    }
}

/// Content analyzer for real-time adaptation
pub struct ContentAnalyzer {
    config: ContentAnalyzerConfig,
    feature_history: VecDeque<ContentFeatures>,
    complexity_history: VecDeque<ContentComplexity>,
    spatial_filters: Vec<Array2<f64>>,
    temporal_filters: Vec<Array2<f64>>,
    frequency_filters: Vec<Array2<f64>>,
    orientation_filters: Vec<Array2<f64>>,
}

impl ContentAnalyzer {
    /// Creates a new content analyzer
    pub fn new() -> Result<Self, AfiyahError> {
        let config = ContentAnalyzerConfig::default();
        Self::with_config(config)
    }

    /// Creates a new content analyzer with custom configuration
    pub fn with_config(config: ContentAnalyzerConfig) -> Result<Self, AfiyahError> {
        let mut analyzer = Self {
            config: config.clone(),
            feature_history: VecDeque::with_capacity(100),
            complexity_history: VecDeque::with_capacity(100),
            spatial_filters: Vec::new(),
            temporal_filters: Vec::new(),
            frequency_filters: Vec::new(),
            orientation_filters: Vec::new(),
        };

        // Initialize filters
        analyzer.initialize_spatial_filters()?;
        analyzer.initialize_temporal_filters()?;
        analyzer.initialize_frequency_filters()?;
        analyzer.initialize_orientation_filters()?;

        Ok(analyzer)
    }

    /// Analyzes content features from visual input
    pub fn analyze_content(&mut self, frame: &Array2<f64>) -> Result<ContentFeatures, AfiyahError> {
        let (height, width) = frame.dim();

        // Calculate spatial complexity
        let spatial_complexity = self.calculate_spatial_complexity(frame)?;

        // Calculate temporal variance (if we have previous frames)
        let temporal_variance = self.calculate_temporal_variance(frame)?;

        // Calculate edge density
        let edge_density = self.calculate_edge_density(frame)?;

        // Calculate texture complexity
        let texture_complexity = self.calculate_texture_complexity(frame)?;

        // Calculate color diversity (for grayscale, this is luminance diversity)
        let color_diversity = self.calculate_color_diversity(frame)?;

        // Calculate motion magnitude
        let motion_magnitude = self.calculate_motion_magnitude(frame)?;

        // Calculate luminance variance
        let luminance_variance = self.calculate_luminance_variance(frame)?;

        // Calculate frequency spectrum
        let frequency_spectrum = self.calculate_frequency_spectrum(frame)?;

        // Calculate orientation histogram
        let orientation_histogram = self.calculate_orientation_histogram(frame)?;

        // Calculate spatial frequencies
        let spatial_frequencies = self.calculate_spatial_frequencies(frame)?;

        let features = ContentFeatures {
            spatial_complexity,
            temporal_variance,
            edge_density,
            texture_complexity,
            color_diversity,
            motion_magnitude,
            luminance_variance,
            frequency_spectrum,
            orientation_histogram,
            spatial_frequencies,
        };

        // Update feature history
        self.feature_history.push_back(features.clone());
        if self.feature_history.len() > 100 {
            self.feature_history.pop_front();
        }

        Ok(features)
    }

    /// Calculates content complexity from features
    pub fn calculate_complexity(&mut self, features: &ContentFeatures) -> Result<ContentComplexity, AfiyahError> {
        // Calculate overall complexity
        let overall_complexity = self.calculate_overall_complexity(features)?;

        // Calculate spatial complexity
        let spatial_complexity = features.spatial_complexity;

        // Calculate temporal complexity
        let temporal_complexity = self.calculate_temporal_complexity(features)?;

        // Calculate perceptual complexity
        let perceptual_complexity = self.calculate_perceptual_complexity(features)?;

        // Calculate compression difficulty
        let compression_difficulty = self.calculate_compression_difficulty(features)?;

        // Calculate biological processing load
        let biological_processing_load = self.calculate_biological_processing_load(features)?;

        // Calculate variance and stability
        let (variance, stability) = self.calculate_variance_and_stability()?;

        let complexity = ContentComplexity {
            overall_complexity,
            spatial_complexity,
            temporal_complexity,
            perceptual_complexity,
            compression_difficulty,
            biological_processing_load,
            variance,
            stability,
        };

        // Update complexity history
        self.complexity_history.push_back(complexity.clone());
        if self.complexity_history.len() > 100 {
            self.complexity_history.pop_front();
        }

        Ok(complexity)
    }

    /// Updates the analyzer configuration
    pub fn update_config(&mut self, analysis_window: Duration) -> Result<(), AfiyahError> {
        self.config.analysis_window = analysis_window;
        Ok(())
    }

    /// Resets the analyzer
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.feature_history.clear();
        self.complexity_history.clear();
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &ContentAnalyzerConfig {
        &self.config
    }

    fn initialize_spatial_filters(&mut self) -> Result<(), AfiyahError> {
        self.spatial_filters.clear();

        // Create Gabor filters for different orientations and scales
        for scale in [0.5, 1.0, 2.0, 4.0] {
            for orientation in 0..self.config.orientation_bands {
                let angle = (orientation as f64) * std::f64::consts::PI / (self.config.orientation_bands as f64);
                let filter = self.create_gabor_filter(scale, angle)?;
                self.spatial_filters.push(filter);
            }
        }

        Ok(())
    }

    fn initialize_temporal_filters(&mut self) -> Result<(), AfiyahError> {
        self.temporal_filters.clear();

        // Create temporal filters for different time scales
        for window_size in [3, 5, 7, 9] {
            let filter = self.create_temporal_filter(window_size)?;
            self.temporal_filters.push(filter);
        }

        Ok(())
    }

    fn initialize_frequency_filters(&mut self) -> Result<(), AfiyahError> {
        self.frequency_filters.clear();

        // Create frequency filters for different bands
        for band in 0..self.config.frequency_bands {
            let frequency = (band as f64 + 1.0) / (self.config.frequency_bands as f64);
            let filter = self.create_frequency_filter(frequency)?;
            self.frequency_filters.push(filter);
        }

        Ok(())
    }

    fn initialize_orientation_filters(&mut self) -> Result<(), AfiyahError> {
        self.orientation_filters.clear();

        // Create orientation filters
        for orientation in 0..self.config.orientation_bands {
            let angle = (orientation as f64) * std::f64::consts::PI / (self.config.orientation_bands as f64);
            let filter = self.create_orientation_filter(angle)?;
            self.orientation_filters.push(filter);
        }

        Ok(())
    }

    fn create_gabor_filter(&self, scale: f64, orientation: f64) -> Result<Array2<f64>, AfiyahError> {
        let size = (scale * 6.0) as usize;
        let mut filter = Array2::zeros((size, size));
        let center = size as f64 / 2.0;

        let sigma_x = scale * 0.5;
        let sigma_y = scale * 0.25;
        let frequency = 1.0 / scale;

        for i in 0..size {
            for j in 0..size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;

                // Rotate coordinates
                let x_rot = x * orientation.cos() + y * orientation.sin();
                let y_rot = -x * orientation.sin() + y * orientation.cos();

                // Gaussian envelope
                let gaussian = (-(x_rot * x_rot) / (2.0 * sigma_x * sigma_x) - 
                               (y_rot * y_rot) / (2.0 * sigma_y * sigma_y)).exp();

                // Sinusoidal component
                let sinusoid = (2.0 * std::f64::consts::PI * frequency * x_rot).cos();

                filter[[i, j]] = gaussian * sinusoid;
            }
        }

        Ok(filter)
    }

    fn create_temporal_filter(&self, window_size: usize) -> Result<Array2<f64>, AfiyahError> {
        let mut filter = Array2::zeros((window_size, window_size));
        let center = window_size as f64 / 2.0;
        let sigma = window_size as f64 * 0.3;

        for i in 0..window_size {
            for j in 0..window_size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;
                let distance = (x * x + y * y).sqrt();

                filter[[i, j]] = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
            }
        }

        Ok(filter)
    }

    fn create_frequency_filter(&self, frequency: f64) -> Result<Array2<f64>, AfiyahError> {
        let size = 32;
        let mut filter = Array2::zeros((size, size));
        let center = size as f64 / 2.0;

        for i in 0..size {
            for j in 0..size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;
                let distance = (x * x + y * y).sqrt();

                // Band-pass filter
                let low_cutoff = frequency * 0.8;
                let high_cutoff = frequency * 1.2;

                if distance >= low_cutoff && distance <= high_cutoff {
                    filter[[i, j]] = 1.0;
                }
            }
        }

        Ok(filter)
    }

    fn create_orientation_filter(&self, orientation: f64) -> Result<Array2<f64>, AfiyahError> {
        let size = 16;
        let mut filter = Array2::zeros((size, size));
        let center = size as f64 / 2.0;

        for i in 0..size {
            for j in 0..size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;

                // Calculate angle
                let angle = y.atan2(x);
                let angle_diff = (angle - orientation).abs().min(2.0 * std::f64::consts::PI - (angle - orientation).abs());

                // Orientation selectivity
                if angle_diff < std::f64::consts::PI / 8.0 {
                    filter[[i, j]] = 1.0;
                }
            }
        }

        Ok(filter)
    }

    fn calculate_spatial_complexity(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = frame.dim();
        let mut complexity_sum = 0.0;
        let mut count = 0;

        // Calculate local variance
        for i in 1..height-1 {
            for j in 1..width-1 {
                let center = frame[[i, j]];
                let mut variance = 0.0;
                let mut neighbor_count = 0;

                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di == 0 && dj == 0 { continue; }
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        if ni < height && nj < width {
                            let neighbor = frame[[ni, nj]];
                            variance += (center - neighbor).powi(2);
                            neighbor_count += 1;
                        }
                    }
                }

                if neighbor_count > 0 {
                    complexity_sum += variance / neighbor_count as f64;
                    count += 1;
                }
            }
        }

        Ok(if count > 0 { complexity_sum / count as f64 } else { 0.0 })
    }

    fn calculate_temporal_variance(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        if self.feature_history.is_empty() {
            return Ok(0.0);
        }

        let previous_frame = self.feature_history.back().unwrap();
        let mut variance_sum = 0.0;
        let mut count = 0;

        for i in 0..frame.len() {
            let current = frame.as_slice().unwrap()[i];
            let previous = previous_frame.spatial_complexity; // Simplified comparison
            variance_sum += (current - previous).powi(2);
            count += 1;
        }

        Ok(if count > 0 { variance_sum / count as f64 } else { 0.0 })
    }

    fn calculate_edge_density(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = frame.dim();
        let mut edge_count = 0;
        let mut total_pixels = 0;

        // Simple edge detection using gradient magnitude
        for i in 1..height-1 {
            for j in 1..width-1 {
                let gx = frame[[i, j+1]] - frame[[i, j-1]];
                let gy = frame[[i+1, j]] - frame[[i-1, j]];
                let gradient_magnitude = (gx * gx + gy * gy).sqrt();

                if gradient_magnitude > 0.1 { // Threshold for edge detection
                    edge_count += 1;
                }
                total_pixels += 1;
            }
        }

        Ok(if total_pixels > 0 { edge_count as f64 / total_pixels as f64 } else { 0.0 })
    }

    fn calculate_texture_complexity(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut texture_responses = Vec::new();

        // Apply Gabor filters to measure texture complexity
        for filter in &self.spatial_filters {
            let response = self.convolve2d(frame, filter)?;
            let response_magnitude = response.iter().map(|&x| x.abs()).sum::<f64>();
            texture_responses.push(response_magnitude);
        }

        // Calculate variance of responses as texture complexity measure
        if texture_responses.is_empty() {
            return Ok(0.0);
        }

        let mean_response = texture_responses.iter().sum::<f64>() / texture_responses.len() as f64;
        let variance = texture_responses.iter()
            .map(|&x| (x - mean_response).powi(2))
            .sum::<f64>() / texture_responses.len() as f64;

        Ok(variance)
    }

    fn calculate_color_diversity(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        // For grayscale images, calculate luminance diversity
        let mean_luminance = frame.iter().sum::<f64>() / frame.len() as f64;
        let variance = frame.iter()
            .map(|&x| (x - mean_luminance).powi(2))
            .sum::<f64>() / frame.len() as f64;

        Ok(variance.sqrt())
    }

    fn calculate_motion_magnitude(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        if self.feature_history.is_empty() {
            return Ok(0.0);
        }

        let previous_frame = self.feature_history.back().unwrap();
        let mut motion_sum = 0.0;
        let mut count = 0;

        for i in 0..frame.len() {
            let current = frame.as_slice().unwrap()[i];
            let previous = previous_frame.spatial_complexity; // Simplified comparison
            motion_sum += (current - previous).abs();
            count += 1;
        }

        Ok(if count > 0 { motion_sum / count as f64 } else { 0.0 })
    }

    fn calculate_luminance_variance(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mean_luminance = frame.iter().sum::<f64>() / frame.len() as f64;
        let variance = frame.iter()
            .map(|&x| (x - mean_luminance).powi(2))
            .sum::<f64>() / frame.len() as f64;

        Ok(variance)
    }

    fn calculate_frequency_spectrum(&self, frame: &Array2<f64>) -> Result<Array1<f64>, AfiyahError> {
        let mut spectrum = Array1::zeros(self.config.frequency_bands);

        for (band, filter) in self.frequency_filters.iter().enumerate() {
            let response = self.convolve2d(frame, filter)?;
            let response_magnitude = response.iter().map(|&x| x.abs()).sum::<f64>();
            spectrum[band] = response_magnitude;
        }

        Ok(spectrum)
    }

    fn calculate_orientation_histogram(&self, frame: &Array2<f64>) -> Result<Array1<f64>, AfiyahError> {
        let mut histogram = Array1::zeros(self.config.orientation_bands);

        for (orientation, filter) in self.orientation_filters.iter().enumerate() {
            let response = self.convolve2d(frame, filter)?;
            let response_magnitude = response.iter().map(|&x| x.abs()).sum::<f64>();
            histogram[orientation] = response_magnitude;
        }

        Ok(histogram)
    }

    fn calculate_spatial_frequencies(&self, frame: &Array2<f64>) -> Result<Array1<f64>, AfiyahError> {
        // Calculate spatial frequencies using FFT-like approach
        let mut frequencies = Array1::zeros(self.config.frequency_bands);

        for band in 0..self.config.frequency_bands {
            let frequency = (band as f64 + 1.0) / (self.config.frequency_bands as f64);
            let filter = &self.frequency_filters[band];
            let response = self.convolve2d(frame, filter)?;
            let response_magnitude = response.iter().map(|&x| x.abs()).sum::<f64>();
            frequencies[band] = response_magnitude;
        }

        Ok(frequencies)
    }

    fn calculate_overall_complexity(&self, features: &ContentFeatures) -> Result<f64, AfiyahError> {
        let spatial_weight = 0.3;
        let temporal_weight = 0.2;
        let edge_weight = 0.2;
        let texture_weight = 0.15;
        let color_weight = 0.1;
        let motion_weight = 0.05;

        let overall = (
            features.spatial_complexity * spatial_weight +
            features.temporal_variance * temporal_weight +
            features.edge_density * edge_weight +
            features.texture_complexity * texture_weight +
            features.color_diversity * color_weight +
            features.motion_magnitude * motion_weight
        );

        Ok(overall.min(1.0).max(0.0))
    }

    fn calculate_temporal_complexity(&self, features: &ContentFeatures) -> Result<f64, AfiyahError> {
        let temporal_complexity = (
            features.temporal_variance * 0.6 +
            features.motion_magnitude * 0.4
        );

        Ok(temporal_complexity.min(1.0).max(0.0))
    }

    fn calculate_perceptual_complexity(&self, features: &ContentFeatures) -> Result<f64, AfiyahError> {
        let perceptual_complexity = (
            features.spatial_complexity * 0.4 +
            features.edge_density * 0.3 +
            features.texture_complexity * 0.2 +
            features.color_diversity * 0.1
        );

        Ok(perceptual_complexity.min(1.0).max(0.0))
    }

    fn calculate_compression_difficulty(&self, features: &ContentFeatures) -> Result<f64, AfiyahError> {
        let compression_difficulty = (
            features.spatial_complexity * 0.3 +
            features.temporal_variance * 0.25 +
            features.edge_density * 0.2 +
            features.texture_complexity * 0.15 +
            features.motion_magnitude * 0.1
        );

        Ok(compression_difficulty.min(1.0).max(0.0))
    }

    fn calculate_biological_processing_load(&self, features: &ContentFeatures) -> Result<f64, AfiyahError> {
        let biological_load = (
            features.spatial_complexity * 0.35 +
            features.edge_density * 0.25 +
            features.texture_complexity * 0.2 +
            features.temporal_variance * 0.15 +
            features.motion_magnitude * 0.05
        );

        Ok(biological_load.min(1.0).max(0.0))
    }

    fn calculate_variance_and_stability(&self) -> Result<(f64, f64), AfiyahError> {
        if self.complexity_history.len() < 2 {
            return Ok((0.0, 1.0));
        }

        let complexities: Vec<f64> = self.complexity_history.iter()
            .map(|c| c.overall_complexity)
            .collect();

        let mean_complexity = complexities.iter().sum::<f64>() / complexities.len() as f64;
        let variance = complexities.iter()
            .map(|&x| (x - mean_complexity).powi(2))
            .sum::<f64>() / complexities.len() as f64;

        let stability = 1.0 - variance.sqrt();
        Ok((variance, stability.max(0.0).min(1.0)))
    }

    fn convolve2d(&self, input: &Array2<f64>, filter: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (input_h, input_w) = input.dim();
        let (filter_h, filter_w) = filter.dim();
        let (output_h, output_w) = (input_h - filter_h + 1, input_w - filter_w + 1);

        let mut output = Array2::zeros((output_h, output_w));

        for i in 0..output_h {
            for j in 0..output_w {
                let mut sum = 0.0;
                for fi in 0..filter_h {
                    for fj in 0..filter_w {
                        sum += input[[i + fi, j + fj]] * filter[[fi, fj]];
                    }
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_analyzer_creation() {
        let analyzer = ContentAnalyzer::new();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_content_analysis() {
        let mut analyzer = ContentAnalyzer::new().unwrap();
        let frame = ndarray::Array2::ones((32, 32));
        
        let result = analyzer.analyze_content(&frame);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(features.spatial_complexity >= 0.0 && features.spatial_complexity <= 1.0);
    }

    #[test]
    fn test_complexity_calculation() {
        let mut analyzer = ContentAnalyzer::new().unwrap();
        let frame = ndarray::Array2::ones((32, 32));
        let features = analyzer.analyze_content(&frame).unwrap();
        
        let result = analyzer.calculate_complexity(&features);
        assert!(result.is_ok());
        
        let complexity = result.unwrap();
        assert!(complexity.overall_complexity >= 0.0 && complexity.overall_complexity <= 1.0);
    }
}
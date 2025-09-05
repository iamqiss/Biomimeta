//! Audio-Visual Correlation Module

use ndarray::Array2;
use crate::AfiyahError;

/// Correlation map for audio-visual correlation
#[derive(Debug, Clone)]
pub struct CorrelationMap {
    pub correlations: Array2<f64>,
    pub correlation_strength: f64,
    pub temporal_alignment: f64,
}

impl CorrelationMap {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            correlations: Array2::zeros((height, width)),
            correlation_strength: 0.0,
            temporal_alignment: 0.0,
        }
    }

    pub fn get_correlation(&self, i: usize, j: usize) -> Result<f64, AfiyahError> {
        if i < self.correlations.nrows() && j < self.correlations.ncols() {
            Ok(self.correlations[[i, j]])
        } else {
            Err(AfiyahError::InputError { message: "Index out of bounds".to_string() })
        }
    }
}

/// Audio-visual features for correlation analysis
#[derive(Debug, Clone)]
pub struct AudioVisualFeatures {
    pub visual_features: Vec<f64>,
    pub audio_features: Vec<f64>,
    pub temporal_features: Vec<f64>,
    pub correlation_features: Vec<f64>,
}

impl AudioVisualFeatures {
    pub fn new() -> Self {
        Self {
            visual_features: Vec::new(),
            audio_features: Vec::new(),
            temporal_features: Vec::new(),
            correlation_features: Vec::new(),
        }
    }
}

/// Audio-visual correlator implementing biological cross-modal processing
pub struct AudioVisualCorrelator {
    correlation_threshold: f64,
    temporal_window: usize,
    feature_extraction_window: usize,
}

impl AudioVisualCorrelator {
    /// Creates a new audio-visual correlator
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            correlation_threshold: 0.5,
            temporal_window: 5,
            feature_extraction_window: 10,
        })
    }

    /// Correlates audio and visual input
    pub fn correlate(&self, visual_input: &Array2<f64>, audio_input: &[f64]) -> Result<CorrelationMap, AfiyahError> {
        let (height, width) = visual_input.dim();
        let mut correlation_map = CorrelationMap::new(height, width);

        // Extract features from both modalities
        let visual_features = self.extract_visual_features(visual_input)?;
        let audio_features = self.extract_audio_features(audio_input)?;

        // Calculate correlations
        for i in 0..height {
            for j in 0..width {
                let visual_feature = visual_features[i * width + j];
                let audio_feature = audio_features[i * width + j];
                
                let correlation = self.calculate_correlation(visual_feature, audio_feature)?;
                correlation_map.correlations[[i, j]] = correlation;
            }
        }

        // Calculate overall correlation strength
        correlation_map.correlation_strength = self.calculate_correlation_strength(&correlation_map)?;

        // Calculate temporal alignment
        correlation_map.temporal_alignment = self.calculate_temporal_alignment(visual_input, audio_input)?;

        Ok(correlation_map)
    }

    fn extract_visual_features(&self, visual_input: &Array2<f64>) -> Result<Vec<f64>, AfiyahError> {
        let (height, width) = visual_input.dim();
        let mut features = Vec::new();

        for i in 0..height {
            for j in 0..width {
                let mut feature = 0.0;
                let mut count = 0;

                // Extract local visual features
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;

                        if ni < height && nj < width {
                            feature += visual_input[[ni, nj]];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    features.push(feature / count as f64);
                } else {
                    features.push(visual_input[[i, j]]);
                }
            }
        }

        Ok(features)
    }

    fn extract_audio_features(&self, audio_input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut features = Vec::new();
        let window_size = self.feature_extraction_window;

        for i in 0..audio_input.len() {
            let mut feature = 0.0;
            let mut count = 0;

            // Extract local audio features
            for j in i.saturating_sub(window_size / 2)..(i + window_size / 2).min(audio_input.len()) {
                feature += audio_input[j];
                count += 1;
            }

            if count > 0 {
                features.push(feature / count as f64);
            } else {
                features.push(audio_input[i]);
            }
        }

        Ok(features)
    }

    fn calculate_correlation(&self, visual_feature: f64, audio_feature: f64) -> Result<f64, AfiyahError> {
        // Calculate correlation between visual and audio features
        let correlation = (visual_feature * audio_feature).sqrt();
        Ok(correlation.clamp(0.0, 1.0))
    }

    fn calculate_correlation_strength(&self, correlation_map: &CorrelationMap) -> Result<f64, AfiyahError> {
        let mut total_correlation = 0.0;
        let mut count = 0;

        for correlation in correlation_map.correlations.iter() {
            total_correlation += correlation;
            count += 1;
        }

        if count > 0 {
            Ok(total_correlation / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_temporal_alignment(&self, visual_input: &Array2<f64>, audio_input: &[f64]) -> Result<f64, AfiyahError> {
        // Calculate temporal alignment between visual and audio
        let visual_variance = self.calculate_variance(visual_input)?;
        let audio_variance = self.calculate_audio_variance(audio_input)?;

        let alignment = 1.0 - (visual_variance - audio_variance).abs();
        Ok(alignment.clamp(0.0, 1.0))
    }

    fn calculate_variance(&self, visual_input: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut sum = 0.0;
        let mut count = 0;

        for value in visual_input.iter() {
            sum += value;
            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }

        let mean = sum / count as f64;
        let mut variance = 0.0;

        for value in visual_input.iter() {
            variance += (value - mean).powi(2);
        }

        Ok(variance / count as f64)
    }

    fn calculate_audio_variance(&self, audio_input: &[f64]) -> Result<f64, AfiyahError> {
        if audio_input.is_empty() {
            return Ok(0.0);
        }

        let sum: f64 = audio_input.iter().sum();
        let mean = sum / audio_input.len() as f64;

        let variance: f64 = audio_input.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();

        Ok(variance / audio_input.len() as f64)
    }

    /// Updates correlation threshold
    pub fn set_correlation_threshold(&mut self, threshold: f64) {
        self.correlation_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates temporal window
    pub fn set_temporal_window(&mut self, window: usize) {
        self.temporal_window = window.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_map_creation() {
        let map = CorrelationMap::new(32, 32);
        assert_eq!(map.correlations.dim(), (32, 32));
        assert_eq!(map.correlation_strength, 0.0);
    }

    #[test]
    fn test_audio_visual_features_creation() {
        let features = AudioVisualFeatures::new();
        assert!(features.visual_features.is_empty());
        assert!(features.audio_features.is_empty());
    }

    #[test]
    fn test_audio_visual_correlator_creation() {
        let correlator = AudioVisualCorrelator::new();
        assert!(correlator.is_ok());
    }

    #[test]
    fn test_audio_visual_correlation() {
        let correlator = AudioVisualCorrelator::new().unwrap();
        let visual_input = Array2::ones((32, 32));
        let audio_input = vec![0.5; 100];
        
        let result = correlator.correlate(&visual_input, &audio_input);
        assert!(result.is_ok());
        
        let correlation_map = result.unwrap();
        assert_eq!(correlation_map.correlations.dim(), (32, 32));
    }

    #[test]
    fn test_correlation_threshold_update() {
        let mut correlator = AudioVisualCorrelator::new().unwrap();
        correlator.set_correlation_threshold(0.7);
        assert_eq!(correlator.correlation_threshold, 0.7);
    }
}
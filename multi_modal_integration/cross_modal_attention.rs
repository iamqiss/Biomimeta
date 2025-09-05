//! Cross-Modal Attention Module

use ndarray::Array2;
use crate::AfiyahError;

/// Attention weights for cross-modal processing
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub weights: Array2<f64>,
    pub attention_strength: f64,
    pub modal_balance: f64,
}

impl AttentionWeights {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            weights: Array2::zeros((height, width)),
            attention_strength: 0.0,
            modal_balance: 0.5,
        }
    }

    pub fn get_attention(&self, i: usize, j: usize) -> Result<f64, AfiyahError> {
        if i < self.weights.nrows() && j < self.weights.ncols() {
            Ok(self.weights[[i, j]])
        } else {
            Err(AfiyahError::InputError { message: "Index out of bounds".to_string() })
        }
    }
}

/// Modal integration for cross-modal processing
#[derive(Debug, Clone)]
pub struct ModalIntegration {
    pub visual_weight: f64,
    pub audio_weight: f64,
    pub integration_strength: f64,
    pub attention_modulation: f64,
}

impl ModalIntegration {
    pub fn new() -> Self {
        Self {
            visual_weight: 0.7,
            audio_weight: 0.3,
            integration_strength: 0.8,
            attention_modulation: 0.5,
        }
    }
}

/// Cross-modal attention processor implementing biological attention mechanisms
pub struct CrossModalAttention {
    attention_threshold: f64,
    modal_balance: f64,
    attention_decay: f64,
    integration_window: usize,
}

impl CrossModalAttention {
    /// Creates a new cross-modal attention processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            attention_threshold: 0.5,
            modal_balance: 0.5,
            attention_decay: 0.95,
            integration_window: 5,
        })
    }

    /// Computes cross-modal attention weights
    pub fn compute_attention(&self, visual_input: &Array2<f64>, audio_input: &[f64]) -> Result<AttentionWeights, AfiyahError> {
        let (height, width) = visual_input.dim();
        let mut attention_weights = AttentionWeights::new(height, width);

        // Calculate attention weights for each spatial location
        for i in 0..height {
            for j in 0..width {
                let visual_attention = self.calculate_visual_attention(visual_input, i, j)?;
                let audio_attention = self.calculate_audio_attention(audio_input, i, j)?;
                
                // Combine visual and audio attention
                let combined_attention = self.combine_attention(visual_attention, audio_attention)?;
                attention_weights.weights[[i, j]] = combined_attention;
            }
        }

        // Calculate overall attention strength
        attention_weights.attention_strength = self.calculate_attention_strength(&attention_weights)?;

        // Calculate modal balance
        attention_weights.modal_balance = self.calculate_modal_balance(visual_input, audio_input)?;

        Ok(attention_weights)
    }

    fn calculate_visual_attention(&self, visual_input: &Array2<f64>, i: usize, j: usize) -> Result<f64, AfiyahError> {
        let (height, width) = visual_input.dim();
        let mut attention = 0.0;
        let mut count = 0;

        // Calculate local visual attention based on contrast and intensity
        for di in -1..=1 {
            for dj in -1..=1 {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;

                if ni < height && nj < width {
                    let center_value = visual_input[[i, j]];
                    let neighbor_value = visual_input[[ni, nj]];
                    let contrast = (center_value - neighbor_value).abs();
                    attention += contrast;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(attention / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_audio_attention(&self, audio_input: &[f64], i: usize, j: usize) -> Result<f64, AfiyahError> {
        if audio_input.is_empty() {
            return Ok(0.0);
        }

        // Map spatial location to audio index
        let audio_index = (i * j) % audio_input.len();
        let audio_value = audio_input[audio_index];

        // Calculate audio attention based on intensity and temporal variation
        let mut attention = audio_value;
        
        // Add temporal variation if available
        if audio_index > 0 && audio_index < audio_input.len() - 1 {
            let prev_value = audio_input[audio_index - 1];
            let next_value = audio_input[audio_index + 1];
            let temporal_variation = (audio_value - prev_value).abs() + (audio_value - next_value).abs();
            attention += temporal_variation * 0.5;
        }

        Ok(attention.clamp(0.0, 1.0))
    }

    fn combine_attention(&self, visual_attention: f64, audio_attention: f64) -> Result<f64, AfiyahError> {
        // Combine visual and audio attention with modal balance
        let combined = visual_attention * self.modal_balance + audio_attention * (1.0 - self.modal_balance);
        Ok(combined.clamp(0.0, 1.0))
    }

    fn calculate_attention_strength(&self, attention_weights: &AttentionWeights) -> Result<f64, AfiyahError> {
        let mut total_attention = 0.0;
        let mut count = 0;

        for weight in attention_weights.weights.iter() {
            total_attention += weight;
            count += 1;
        }

        if count > 0 {
            Ok(total_attention / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_modal_balance(&self, visual_input: &Array2<f64>, audio_input: &[f64]) -> Result<f64, AfiyahError> {
        // Calculate modal balance based on relative strength of visual and audio inputs
        let visual_strength = self.calculate_visual_strength(visual_input)?;
        let audio_strength = self.calculate_audio_strength(audio_input)?;

        if visual_strength + audio_strength > 0.0 {
            Ok(visual_strength / (visual_strength + audio_strength))
        } else {
            Ok(0.5)
        }
    }

    fn calculate_visual_strength(&self, visual_input: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut total_strength = 0.0;
        let mut count = 0;

        for value in visual_input.iter() {
            total_strength += value.abs();
            count += 1;
        }

        if count > 0 {
            Ok(total_strength / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_audio_strength(&self, audio_input: &[f64]) -> Result<f64, AfiyahError> {
        if audio_input.is_empty() {
            return Ok(0.0);
        }

        let total_strength: f64 = audio_input.iter().map(|&x| x.abs()).sum();
        Ok(total_strength / audio_input.len() as f64)
    }

    /// Updates attention threshold
    pub fn set_attention_threshold(&mut self, threshold: f64) {
        self.attention_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates modal balance
    pub fn set_modal_balance(&mut self, balance: f64) {
        self.modal_balance = balance.clamp(0.0, 1.0);
    }

    /// Updates attention decay
    pub fn set_attention_decay(&mut self, decay: f64) {
        self.attention_decay = decay.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_weights_creation() {
        let weights = AttentionWeights::new(32, 32);
        assert_eq!(weights.weights.dim(), (32, 32));
        assert_eq!(weights.attention_strength, 0.0);
    }

    #[test]
    fn test_modal_integration_creation() {
        let integration = ModalIntegration::new();
        assert_eq!(integration.visual_weight, 0.7);
        assert_eq!(integration.audio_weight, 0.3);
    }

    #[test]
    fn test_cross_modal_attention_creation() {
        let attention = CrossModalAttention::new();
        assert!(attention.is_ok());
    }

    #[test]
    fn test_attention_computation() {
        let attention = CrossModalAttention::new().unwrap();
        let visual_input = Array2::ones((32, 32));
        let audio_input = vec![0.5; 100];
        
        let result = attention.compute_attention(&visual_input, &audio_input);
        assert!(result.is_ok());
        
        let attention_weights = result.unwrap();
        assert_eq!(attention_weights.weights.dim(), (32, 32));
    }

    #[test]
    fn test_attention_threshold_update() {
        let mut attention = CrossModalAttention::new().unwrap();
        attention.set_attention_threshold(0.7);
        assert_eq!(attention.attention_threshold, 0.7);
    }

    #[test]
    fn test_modal_balance_update() {
        let mut attention = CrossModalAttention::new().unwrap();
        attention.set_modal_balance(0.8);
        assert_eq!(attention.modal_balance, 0.8);
    }
}
//! Foveated Encoder Module

use ndarray::Array2;
use crate::AfiyahError;

/// Foveated encoding configuration
#[derive(Debug, Clone)]
pub struct FoveatedConfig {
    pub foveal_radius: f64,
    pub peripheral_compression: f64,
    pub quality_falloff: f64,
    pub encoding_regions: usize,
}

impl Default for FoveatedConfig {
    fn default() -> Self {
        Self {
            foveal_radius: 2.0,
            peripheral_compression: 0.5,
            quality_falloff: 0.8,
            encoding_regions: 3,
        }
    }
}

/// Encoding region for foveated encoding
#[derive(Debug, Clone)]
pub struct EncodingRegion {
    pub center: (f64, f64),
    pub radius: f64,
    pub quality: f64,
    pub compression_ratio: f64,
}

impl EncodingRegion {
    pub fn new(center: (f64, f64), radius: f64, quality: f64, compression_ratio: f64) -> Self {
        Self {
            center,
            radius,
            quality,
            compression_ratio,
        }
    }
}

/// Foveated encoder implementing biological foveated encoding
pub struct FoveatedEncoder {
    config: FoveatedConfig,
    encoding_regions: Vec<EncodingRegion>,
    quality_weights: Array2<f64>,
}

impl FoveatedEncoder {
    /// Creates a new foveated encoder
    pub fn new() -> Result<Self, AfiyahError> {
        let config = FoveatedConfig::default();
        let encoding_regions = Vec::new();
        let quality_weights = Array2::zeros((64, 64));

        Ok(Self {
            config,
            encoding_regions,
            quality_weights,
        })
    }

    /// Initializes the foveated encoder
    pub fn initialize(&mut self) -> Result<(), AfiyahError> {
        self.encoding_regions.clear();
        self.quality_weights = Array2::zeros((64, 64));
        Ok(())
    }

    /// Stops the foveated encoder
    pub fn stop(&mut self) -> Result<(), AfiyahError> {
        self.encoding_regions.clear();
        Ok(())
    }

    /// Encodes a frame with foveated encoding
    pub fn encode_frame(&mut self, frame: &Array2<f64>, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<Vec<u8>, AfiyahError> {
        let (height, width) = frame.dim();
        let mut encoded_data = Vec::new();

        // Create encoding regions based on foveal quality
        self.create_encoding_regions(height, width, quality_params)?;

        // Encode each region with appropriate quality
        for region in &self.encoding_regions {
            let region_data = self.encode_region(frame, region)?;
            encoded_data.extend_from_slice(&region_data);
        }

        // Add metadata
        let metadata = self.create_metadata(quality_params)?;
        encoded_data.extend_from_slice(&metadata);

        Ok(encoded_data)
    }

    fn create_encoding_regions(&mut self, height: usize, width: usize, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<(), AfiyahError> {
        self.encoding_regions.clear();

        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        // Create foveal region (highest quality)
        let foveal_region = EncodingRegion::new(
            (center_x, center_y),
            self.config.foveal_radius,
            quality_params.foveal_quality,
            1.0, // No compression
        );
        self.encoding_regions.push(foveal_region);

        // Create intermediate regions
        for i in 1..self.config.encoding_regions {
            let radius = self.config.foveal_radius * (i as f64 + 1.0);
            let quality = quality_params.foveal_quality * self.config.quality_falloff.powi(i as i32);
            let compression_ratio = 1.0 - (i as f64 * self.config.peripheral_compression / self.config.encoding_regions as f64);

            let region = EncodingRegion::new(
                (center_x, center_y),
                radius,
                quality,
                compression_ratio,
            );
            self.encoding_regions.push(region);
        }

        Ok(())
    }

    fn encode_region(&self, frame: &Array2<f64>, region: &EncodingRegion) -> Result<Vec<u8>, AfiyahError> {
        let (height, width) = frame.dim();
        let mut region_data = Vec::new();

        // Sample region data
        let mut region_pixels = Vec::new();
        for i in 0..height {
            for j in 0..width {
                let x = j as f64;
                let y = i as f64;
                let distance = ((x - region.center.0).powi(2) + (y - region.center.1).powi(2)).sqrt();

                if distance <= region.radius {
                    let pixel_value = frame[[i, j]];
                    region_pixels.push(pixel_value);
                }
            }
        }

        // Apply compression based on region settings
        let compressed_pixels = self.compress_region_data(&region_pixels, region)?;

        // Convert to bytes
        for pixel in compressed_pixels {
            let byte_value = (pixel * 255.0).clamp(0.0, 255.0) as u8;
            region_data.push(byte_value);
        }

        Ok(region_data)
    }

    fn compress_region_data(&self, pixels: &[f64], region: &EncodingRegion) -> Result<Vec<f64>, AfiyahError> {
        let mut compressed = Vec::new();

        if region.compression_ratio >= 1.0 {
            // No compression
            compressed.extend_from_slice(pixels);
        } else {
            // Apply compression
            let compression_factor = 1.0 - region.compression_ratio;
            let step = (1.0 / compression_factor).ceil() as usize;

            for (i, &pixel) in pixels.iter().enumerate() {
                if i % step == 0 {
                    compressed.push(pixel);
                }
            }
        }

        Ok(compressed)
    }

    fn create_metadata(&self, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<Vec<u8>, AfiyahError> {
        let mut metadata = Vec::new();

        // Add quality information
        let quality_bytes = quality_params.overall_quality.to_le_bytes();
        metadata.extend_from_slice(&quality_bytes);

        // Add region count
        metadata.push(self.encoding_regions.len() as u8);

        // Add region information
        for region in &self.encoding_regions {
            let center_x_bytes = region.center.0.to_le_bytes();
            let center_y_bytes = region.center.1.to_le_bytes();
            let radius_bytes = region.radius.to_le_bytes();
            let quality_bytes = region.quality.to_le_bytes();

            metadata.extend_from_slice(&center_x_bytes);
            metadata.extend_from_slice(&center_y_bytes);
            metadata.extend_from_slice(&radius_bytes);
            metadata.extend_from_slice(&quality_bytes);
        }

        Ok(metadata)
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &FoveatedConfig {
        &self.config
    }

    /// Updates configuration
    pub fn update_config(&mut self, config: FoveatedConfig) {
        self.config = config;
    }

    /// Gets encoding regions
    pub fn get_encoding_regions(&self) -> &Vec<EncodingRegion> {
        &self.encoding_regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foveated_config_default() {
        let config = FoveatedConfig::default();
        assert_eq!(config.foveal_radius, 2.0);
        assert_eq!(config.peripheral_compression, 0.5);
        assert_eq!(config.encoding_regions, 3);
    }

    #[test]
    fn test_encoding_region_creation() {
        let region = EncodingRegion::new((10.0, 20.0), 5.0, 0.8, 0.5);
        assert_eq!(region.center, (10.0, 20.0));
        assert_eq!(region.radius, 5.0);
        assert_eq!(region.quality, 0.8);
    }

    #[test]
    fn test_foveated_encoder_creation() {
        let encoder = FoveatedEncoder::new();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_initialization() {
        let mut encoder = FoveatedEncoder::new().unwrap();
        let result = encoder.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_frame_encoding() {
        let mut encoder = FoveatedEncoder::new().unwrap();
        encoder.initialize().unwrap();
        
        let frame = Array2::ones((32, 32));
        let quality_params = crate::streaming_engine::biological_qos::PerceptualQuality::new();
        
        let result = encoder.encode_frame(&frame, &quality_params);
        assert!(result.is_ok());
        
        let encoded_data = result.unwrap();
        assert!(!encoded_data.is_empty());
    }
}
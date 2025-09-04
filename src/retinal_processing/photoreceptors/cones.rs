//! Cone Photoreceptor Implementation
//! 
//! This module implements cone photoreceptors for photopic (daylight) vision and
//! color perception. Cones are responsible for high-acuity vision and color
//! discrimination with three types: S (short), M (medium), and L (long) wavelength cones.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::{SpatialSample, PhotoreceptorType};

/// Cone photoreceptor system for color vision
pub struct ConePhotoreceptors {
    s_cones: SCones,
    m_cones: MCones,
    l_cones: LCones,
    color_opponency: ColorOpponency,
    temporal_response: TemporalResponse,
    adaptation_curve: AdaptationCurve,
}

impl ConePhotoreceptors {
    /// Creates new cone photoreceptors with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            s_cones: SCones::new()?,
            m_cones: MCones::new()?,
            l_cones: LCones::new()?,
            color_opponency: ColorOpponency::new()?,
            temporal_response: TemporalResponse::cone_default(),
            adaptation_curve: AdaptationCurve::cone_default(),
        })
    }

    /// Processes chromatic data through cone photoreceptors
    pub fn process(&self, chromatic_data: &[f64], spatial_samples: &[SpatialSample]) -> Result<Vec<f64>, AfiyahError> {
        if chromatic_data.len() != spatial_samples.len() * 2 {
            return Err(AfiyahError::InputError(
                "Chromatic data length must be 2x spatial samples length".to_string()
            ));
        }

        let mut cone_signals = Vec::with_capacity(spatial_samples.len());
        
        for (i, sample) in spatial_samples.iter().enumerate() {
            // Only process cone samples
            if matches!(sample.photoreceptor_type, PhotoreceptorType::Cone) {
                let red_channel = chromatic_data[i * 2];
                let green_channel = chromatic_data[i * 2 + 1];
                let blue_channel = 1.0 - red_channel - green_channel; // Approximate blue from RGB
                
                let cone_signal = self.process_single_cone(red_channel, green_channel, blue_channel, sample)?;
                cone_signals.push(cone_signal);
            } else {
                // For rod samples, use minimal cone contribution
                cone_signals.push(0.01);
            }
        }
        
        Ok(cone_signals)
    }

    /// Calibrates cone photoreceptors based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.s_cones.calibrate(params)?;
        self.m_cones.calibrate(params)?;
        self.l_cones.calibrate(params)?;
        self.color_opponency.calibrate(params)?;
        Ok(())
    }

    fn process_single_cone(&self, red: f64, green: f64, blue: f64, sample: &SpatialSample) -> Result<f64, AfiyahError> {
        // Process through individual cone types
        let s_response = self.s_cones.process(blue);
        let m_response = self.m_cones.process(green);
        let l_response = self.l_cones.process(red);
        
        // Apply color opponency processing
        let opponency_response = self.color_opponency.process(s_response, m_response, l_response)?;
        
        // Apply temporal response filtering
        let temporal_response = self.temporal_response.filter(opponency_response);
        
        // Apply adaptation curve
        let adapted_response = self.adaptation_curve.apply(temporal_response);
        
        // Scale by spatial density
        let density_factor = sample.density / 200_000.0; // Normalize to foveal density
        let final_response = adapted_response * density_factor;
        
        // Ensure response is within biological limits
        Ok(final_response.min(1.0).max(0.0))
    }
}

/// S-cone (short wavelength) photoreceptors for blue vision
#[derive(Debug, Clone)]
pub struct SCones {
    peak_wavelength: f64,    // Peak sensitivity wavelength (nm)
    sensitivity: f64,        // Sensitivity factor
    density_ratio: f64,      // Relative density compared to L/M cones
}

impl SCones {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            peak_wavelength: 420.0,  // Blue peak
            sensitivity: 0.1,        // Lower sensitivity than L/M cones
            density_ratio: 0.05,     // 5% of total cones
        })
    }

    pub fn process(&self, blue_input: f64) -> f64 {
        blue_input * self.sensitivity * self.density_ratio
    }

    pub fn calibrate(&mut self, _params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // S-cones are less sensitive to calibration changes
        Ok(())
    }
}

/// M-cone (medium wavelength) photoreceptors for green vision
#[derive(Debug, Clone)]
pub struct MCones {
    peak_wavelength: f64,    // Peak sensitivity wavelength (nm)
    sensitivity: f64,        // Sensitivity factor
    density_ratio: f64,      // Relative density compared to L cones
}

impl MCones {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            peak_wavelength: 530.0,  // Green peak
            sensitivity: 0.8,        // High sensitivity
            density_ratio: 0.4,      // 40% of total cones
        })
    }

    pub fn process(&self, green_input: f64) -> f64 {
        green_input * self.sensitivity * self.density_ratio
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.sensitivity = params.cone_sensitivity * 0.8;
        Ok(())
    }
}

/// L-cone (long wavelength) photoreceptors for red vision
#[derive(Debug, Clone)]
pub struct LCones {
    peak_wavelength: f64,    // Peak sensitivity wavelength (nm)
    sensitivity: f64,        // Sensitivity factor
    density_ratio: f64,      // Relative density compared to M cones
}

impl LCones {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            peak_wavelength: 560.0,  // Red peak
            sensitivity: 1.0,        // Highest sensitivity
            density_ratio: 0.55,     // 55% of total cones
        })
    }

    pub fn process(&self, red_input: f64) -> f64 {
        red_input * self.sensitivity * self.density_ratio
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.sensitivity = params.cone_sensitivity;
        Ok(())
    }
}

/// Color opponency processing for cone signals
#[derive(Debug, Clone)]
pub struct ColorOpponency {
    red_green_opponency: f64,
    blue_yellow_opponency: f64,
    luminance_channel: f64,
}

impl ColorOpponency {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            red_green_opponency: 0.5,
            blue_yellow_opponency: 0.3,
            luminance_channel: 0.7,
        })
    }

    pub fn process(&self, s_response: f64, m_response: f64, l_response: f64) -> Result<f64, AfiyahError> {
        // Red-Green opponency: L - M
        let rg_opponency = (l_response - m_response) * self.red_green_opponency;
        
        // Blue-Yellow opponency: S - (L + M)
        let by_opponency = (s_response - (l_response + m_response) / 2.0) * self.blue_yellow_opponency;
        
        // Luminance channel: L + M
        let luminance = (l_response + m_response) * self.luminance_channel;
        
        // Combine opponency channels
        let combined_response = (rg_opponency + by_opponency + luminance) / 3.0;
        
        Ok(combined_response)
    }

    pub fn calibrate(&mut self, _params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Color opponency is relatively stable
        Ok(())
    }
}

/// Temporal response characteristics of cone photoreceptors
#[derive(Debug, Clone)]
pub struct TemporalResponse {
    pub time_constant: f64,      // Response time constant (ms)
    pub integration_time: f64,   // Temporal integration window (ms)
    pub cutoff_frequency: f64,   // High-frequency cutoff (Hz)
}

impl TemporalResponse {
    /// Creates default temporal response for cones
    pub fn cone_default() -> Self {
        Self {
            time_constant: 50.0,     // Cones are faster than rods
            integration_time: 20.0,  // Shorter integration for acuity
            cutoff_frequency: 30.0,  // Higher temporal resolution
        }
    }

    /// Applies temporal filtering to input signal
    pub fn filter(&self, input: f64) -> f64 {
        // Simple low-pass filter approximation
        let alpha = 1.0 / (1.0 + self.time_constant / 1000.0);
        input * alpha
    }
}

/// Adaptation curve for cone photoreceptors
#[derive(Debug, Clone)]
pub struct AdaptationCurve {
    pub light_adapted_threshold: f64,
    pub adaptation_rate: f64,
    pub current_threshold: f64,
}

impl AdaptationCurve {
    /// Creates default adaptation curve for cones
    pub fn cone_default() -> Self {
        Self {
            light_adapted_threshold: 1e-3,  // Less sensitive than rods
            adaptation_rate: 0.2,           // Faster adaptation than rods
            current_threshold: 1e-3,
        }
    }

    /// Applies adaptation to input signal
    pub fn apply(&self, input: f64) -> f64 {
        if input > self.current_threshold {
            // Light adaptation
            (input - self.current_threshold) / (1.0 - self.current_threshold)
        } else {
            // Dark adaptation
            input / self.current_threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_photoreceptors_creation() {
        let cones = ConePhotoreceptors::new();
        assert!(cones.is_ok());
    }

    #[test]
    fn test_cone_processing() {
        let cones = ConePhotoreceptors::new().unwrap();
        let chromatic_data = vec![0.1, 0.5, 0.9, 0.3, 0.7, 0.2]; // 3 pixels worth
        let spatial_samples = vec![
            SpatialSample { x: 0.0, y: 0.0, density: 200_000.0, photoreceptor_type: PhotoreceptorType::Cone },
            SpatialSample { x: 0.5, y: 0.5, density: 200_000.0, photoreceptor_type: PhotoreceptorType::Cone },
            SpatialSample { x: 1.0, y: 1.0, density: 200_000.0, photoreceptor_type: PhotoreceptorType::Cone },
        ];
        
        let result = cones.process(&chromatic_data, &spatial_samples);
        assert!(result.is_ok());
        let signals = result.unwrap();
        assert_eq!(signals.len(), 3);
    }

    #[test]
    fn test_individual_cone_types() {
        let s_cones = SCones::new().unwrap();
        let m_cones = MCones::new().unwrap();
        let l_cones = LCones::new().unwrap();
        
        assert_eq!(s_cones.peak_wavelength, 420.0);
        assert_eq!(m_cones.peak_wavelength, 530.0);
        assert_eq!(l_cones.peak_wavelength, 560.0);
    }

    #[test]
    fn test_color_opponency() {
        let opponency = ColorOpponency::new().unwrap();
        let result = opponency.process(0.1, 0.5, 0.8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_response() {
        let temporal = TemporalResponse::cone_default();
        assert_eq!(temporal.time_constant, 50.0);
        assert_eq!(temporal.cutoff_frequency, 30.0);
    }
}
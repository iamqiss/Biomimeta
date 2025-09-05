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

//! Hardware Acceleration Module
//! 
//! This module implements sophisticated hardware acceleration capabilities
//! for biological visual processing and compression.
//! 
//! Biological Basis:
//! - Parallel processing in biological neural networks
//! - Specialized hardware for visual processing
//! - Neuromorphic computing principles
//! - SIMD optimization for biological algorithms

use ndarray::Array2;
use crate::AfiyahError;

// Re-export all sub-modules
pub mod gpu_acceleration;
pub mod simd_optimization;
pub mod neuromorphic_interfaces;

// Re-export the main types
pub use gpu_acceleration::{GPUAccelerator, CudaKernel, OpenCLKernel};
pub use simd_optimization::{SIMDOptimizer, SIMDConfig, SIMDArchitecture};
pub use neuromorphic_interfaces::{NeuromorphicInterface, NeuromorphicConfig, NeuromorphicHardware};

/// Hardware acceleration manager
pub struct HardwareAccelerator {
    gpu_accelerator: Option<GPUAccelerator>,
    simd_optimizer: Option<SIMDOptimizer>,
    neuromorphic_interface: Option<NeuromorphicInterface>,
    acceleration_config: AccelerationConfig,
}

/// Acceleration configuration
#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    pub gpu_enabled: bool,
    pub simd_enabled: bool,
    pub neuromorphic_enabled: bool,
    pub parallel_threads: usize,
    pub memory_optimization: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            gpu_enabled: true,
            simd_enabled: true,
            neuromorphic_enabled: false,
            parallel_threads: 4,
            memory_optimization: true,
        }
    }
}

impl HardwareAccelerator {
    /// Creates a new hardware accelerator
    pub fn new() -> Result<Self, AfiyahError> {
        let acceleration_config = AccelerationConfig::default();

        Ok(Self {
            gpu_accelerator: None,
            simd_optimizer: None,
            neuromorphic_interface: None,
            acceleration_config,
        })
    }

    /// Enables GPU acceleration
    pub fn enable_gpu(&mut self) -> Result<(), AfiyahError> {
        self.acceleration_config.gpu_enabled = true;
        self.gpu_accelerator = Some(GPUAccelerator::new()?);
        Ok(())
    }

    /// Enables SIMD optimization
    pub fn enable_simd(&mut self, architecture: SIMDArchitecture) -> Result<(), AfiyahError> {
        self.acceleration_config.simd_enabled = true;
        let config = SIMDConfig::new(architecture);
        self.simd_optimizer = Some(SIMDOptimizer::new(config));
        Ok(())
    }

    /// Enables neuromorphic processing
    pub fn enable_neuromorphic(&mut self, hardware: NeuromorphicHardware) -> Result<(), AfiyahError> {
        self.acceleration_config.neuromorphic_enabled = true;
        let config = NeuromorphicConfig::new(hardware);
        self.neuromorphic_interface = Some(NeuromorphicInterface::new(config));
        Ok(())
    }

    /// Accelerates processing with hardware optimization
    pub fn accelerate_processing(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let mut output = input.clone();

        // Apply SIMD optimization if enabled
        if self.acceleration_config.simd_enabled {
            if let Some(ref mut simd_optimizer) = self.simd_optimizer {
                output = simd_optimizer.optimize_operations(&output)?;
            }
        }

        // Apply GPU acceleration if enabled
        if self.acceleration_config.gpu_enabled {
            if let Some(ref mut gpu_accelerator) = self.gpu_accelerator {
                output = gpu_accelerator.accelerate_processing(&output)?;
            }
        }

        // Apply neuromorphic processing if enabled
        if self.acceleration_config.neuromorphic_enabled {
            if let Some(ref mut neuromorphic_interface) = self.neuromorphic_interface {
                output = neuromorphic_interface.process_input(&output)?;
            }
        }

        Ok(output)
    }

    /// Updates acceleration configuration
    pub fn update_config(&mut self, config: AccelerationConfig) {
        self.acceleration_config = config;
    }

    /// Gets current acceleration configuration
    pub fn get_config(&self) -> &AccelerationConfig {
        &self.acceleration_config
    }

    /// Gets GPU accelerator
    pub fn get_gpu_accelerator(&self) -> Option<&GPUAccelerator> {
        self.gpu_accelerator.as_ref()
    }

    /// Gets SIMD optimizer
    pub fn get_simd_optimizer(&self) -> Option<&SIMDOptimizer> {
        self.simd_optimizer.as_ref()
    }

    /// Gets neuromorphic interface
    pub fn get_neuromorphic_interface(&self) -> Option<&NeuromorphicInterface> {
        self.neuromorphic_interface.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceleration_config_default() {
        let config = AccelerationConfig::default();
        assert!(config.gpu_enabled);
        assert!(config.simd_enabled);
        assert!(!config.neuromorphic_enabled);
        assert_eq!(config.parallel_threads, 4);
    }

    #[test]
    fn test_hardware_accelerator_creation() {
        let accelerator = HardwareAccelerator::new();
        assert!(accelerator.is_ok());
    }

    #[test]
    fn test_gpu_enable() {
        let mut accelerator = HardwareAccelerator::new().unwrap();
        let result = accelerator.enable_gpu();
        assert!(result.is_ok());
        assert!(accelerator.get_config().gpu_enabled);
        assert!(accelerator.get_gpu_accelerator().is_some());
    }

    #[test]
    fn test_simd_enable() {
        let mut accelerator = HardwareAccelerator::new().unwrap();
        let result = accelerator.enable_simd(SIMDArchitecture::AVX2);
        assert!(result.is_ok());
        assert!(accelerator.get_config().simd_enabled);
        assert!(accelerator.get_simd_optimizer().is_some());
    }

    #[test]
    fn test_neuromorphic_enable() {
        let mut accelerator = HardwareAccelerator::new().unwrap();
        let result = accelerator.enable_neuromorphic(NeuromorphicHardware::Loihi);
        assert!(result.is_ok());
        assert!(accelerator.get_config().neuromorphic_enabled);
        assert!(accelerator.get_neuromorphic_interface().is_some());
    }

    #[test]
    fn test_hardware_acceleration() {
        let mut accelerator = HardwareAccelerator::new().unwrap();
        accelerator.enable_simd(SIMDArchitecture::AVX2).unwrap();
        let input = Array2::ones((32, 32));
        
        let result = accelerator.accelerate_processing(&input);
        assert!(result.is_ok());
        
        let accelerated_output = result.unwrap();
        assert_eq!(accelerated_output.dim(), (32, 32));
    }

    #[test]
    fn test_configuration_update() {
        let mut accelerator = HardwareAccelerator::new().unwrap();
        let config = AccelerationConfig {
            gpu_enabled: false,
            simd_enabled: true,
            neuromorphic_enabled: true,
            parallel_threads: 8,
            memory_optimization: false,
        };
        
        accelerator.update_config(config);
        assert!(!accelerator.get_config().gpu_enabled);
        assert!(accelerator.get_config().neuromorphic_enabled);
    }
}
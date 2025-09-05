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

//! Configuration Module
//! 
//! This module provides configuration management for biological defaults, device
//! profiles, and experimental flags across the Afiyah biomimetic video compression system.

use crate::AfiyahError;
use std::collections::HashMap;

/// Main configuration manager for the Afiyah system
pub struct ConfigManager {
    pub biological_config: BiologicalConfig,
    pub device_profiles: HashMap<String, DeviceProfile>,
    pub experimental_flags: ExperimentalFlags,
    pub user_preferences: UserPreferences,
}

impl ConfigManager {
    /// Creates a new configuration manager with default settings
    pub fn new() -> Result<Self, AfiyahError> {
        let mut device_profiles = HashMap::new();
        
        // Add default device profiles
        device_profiles.insert("gpu_nvidia".to_string(), DeviceProfile::nvidia_gpu());
        device_profiles.insert("gpu_amd".to_string(), DeviceProfile::amd_gpu());
        device_profiles.insert("cpu_intel".to_string(), DeviceProfile::intel_cpu());
        device_profiles.insert("cpu_arm".to_string(), DeviceProfile::arm_cpu());
        device_profiles.insert("neuromorphic".to_string(), DeviceProfile::neuromorphic());
        
        Ok(Self {
            biological_config: BiologicalConfig::default(),
            device_profiles,
            experimental_flags: ExperimentalFlags::default(),
            user_preferences: UserPreferences::default(),
        })
    }

    /// Loads configuration from file
    pub fn load_from_file(&mut self, filename: &str) -> Result<(), AfiyahError> {
        // In a real implementation, this would load from JSON/YAML/TOML
        // For now, we'll use default values
        self.biological_config = BiologicalConfig::default();
        self.experimental_flags = ExperimentalFlags::default();
        self.user_preferences = UserPreferences::default();
        Ok(())
    }

    /// Saves configuration to file
    pub fn save_to_file(&self, filename: &str) -> Result<(), AfiyahError> {
        // In a real implementation, this would save to JSON/YAML/TOML
        // For now, we'll just return Ok
        Ok(())
    }

    /// Gets device profile by name
    pub fn get_device_profile(&self, name: &str) -> Option<&DeviceProfile> {
        self.device_profiles.get(name)
    }

    /// Updates biological configuration
    pub fn update_biological_config(&mut self, config: BiologicalConfig) {
        self.biological_config = config;
    }

    /// Updates experimental flags
    pub fn update_experimental_flags(&mut self, flags: ExperimentalFlags) {
        self.experimental_flags = flags;
    }
}

/// Biological configuration parameters
#[derive(Debug, Clone)]
pub struct BiologicalConfig {
    // Retinal parameters
    pub rod_sensitivity: f64,
    pub cone_sensitivity: f64,
    pub foveal_density: f64,
    pub peripheral_density: f64,
    pub adaptation_rate: f64,
    
    // Cortical parameters
    pub v1_orientation_selectivity: f64,
    pub v2_texture_sensitivity: f64,
    pub v5_motion_sensitivity: f64,
    pub cortical_magnification: f64,
    
    // Synaptic parameters
    pub hebbian_learning_rate: f64,
    pub homeostatic_plasticity_rate: f64,
    pub neuromodulation_strength: f64,
    pub habituation_rate: f64,
    
    // Perceptual parameters
    pub masking_threshold: f64,
    pub foveal_radius: f64,
    pub attention_weight: f64,
    pub temporal_integration_window: u64,
    
    // Streaming parameters
    pub saccadic_prediction: bool,
    pub foveal_attention: bool,
    pub biological_qos: bool,
    pub adaptive_streaming: bool,
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            // Retinal parameters (based on biological research)
            rod_sensitivity: 0.8,
            cone_sensitivity: 0.6,
            foveal_density: 200_000.0, // cones per mm²
            peripheral_density: 150_000.0, // rods per mm²
            adaptation_rate: 0.1,
            
            // Cortical parameters
            v1_orientation_selectivity: 0.7,
            v2_texture_sensitivity: 0.6,
            v5_motion_sensitivity: 0.8,
            cortical_magnification: 2.5,
            
            // Synaptic parameters
            hebbian_learning_rate: 0.01,
            homeostatic_plasticity_rate: 0.001,
            neuromodulation_strength: 0.5,
            habituation_rate: 0.05,
            
            // Perceptual parameters
            masking_threshold: 0.3,
            foveal_radius: 1.2, // degrees
            attention_weight: 0.8,
            temporal_integration_window: 200, // milliseconds
            
            // Streaming parameters
            saccadic_prediction: true,
            foveal_attention: true,
            biological_qos: true,
            adaptive_streaming: true,
        }
    }
}

/// Device profile for hardware optimization
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: String,
    pub device_type: DeviceType,
    pub compute_capability: f64,
    pub memory_bandwidth: f64,
    pub memory_capacity: usize,
    pub parallel_cores: usize,
    pub simd_width: usize,
    pub optimization_flags: Vec<OptimizationFlag>,
    pub biological_accuracy_weight: f64,
    pub performance_weight: f64,
}

/// Device types
#[derive(Debug, Clone, Copy)]
pub enum DeviceType {
    GPU,
    CPU,
    Neuromorphic,
    FPGA,
    ASIC,
}

/// Optimization flags
#[derive(Debug, Clone, Copy)]
pub enum OptimizationFlag {
    SIMD,
    GPU,
    MultiThreading,
    MemoryOptimization,
    CacheOptimization,
    BranchPrediction,
    Vectorization,
}

impl DeviceProfile {
    /// Creates NVIDIA GPU profile
    pub fn nvidia_gpu() -> Self {
        Self {
            name: "NVIDIA GPU".to_string(),
            device_type: DeviceType::GPU,
            compute_capability: 8.6,
            memory_bandwidth: 900.0, // GB/s
            memory_capacity: 24 * 1024 * 1024 * 1024, // 24 GB
            parallel_cores: 10496,
            simd_width: 32,
            optimization_flags: vec![
                OptimizationFlag::GPU,
                OptimizationFlag::SIMD,
                OptimizationFlag::MultiThreading,
                OptimizationFlag::MemoryOptimization,
            ],
            biological_accuracy_weight: 0.9,
            performance_weight: 0.1,
        }
    }

    /// Creates AMD GPU profile
    pub fn amd_gpu() -> Self {
        Self {
            name: "AMD GPU".to_string(),
            device_type: DeviceType::GPU,
            compute_capability: 2.0,
            memory_bandwidth: 800.0, // GB/s
            memory_capacity: 16 * 1024 * 1024 * 1024, // 16 GB
            parallel_cores: 4096,
            simd_width: 64,
            optimization_flags: vec![
                OptimizationFlag::GPU,
                OptimizationFlag::SIMD,
                OptimizationFlag::MultiThreading,
            ],
            biological_accuracy_weight: 0.85,
            performance_weight: 0.15,
        }
    }

    /// Creates Intel CPU profile
    pub fn intel_cpu() -> Self {
        Self {
            name: "Intel CPU".to_string(),
            device_type: DeviceType::CPU,
            compute_capability: 1.0,
            memory_bandwidth: 100.0, // GB/s
            memory_capacity: 64 * 1024 * 1024 * 1024, // 64 GB
            parallel_cores: 16,
            simd_width: 8,
            optimization_flags: vec![
                OptimizationFlag::SIMD,
                OptimizationFlag::MultiThreading,
                OptimizationFlag::CacheOptimization,
                OptimizationFlag::BranchPrediction,
                OptimizationFlag::Vectorization,
            ],
            biological_accuracy_weight: 0.95,
            performance_weight: 0.05,
        }
    }

    /// Creates ARM CPU profile
    pub fn arm_cpu() -> Self {
        Self {
            name: "ARM CPU".to_string(),
            device_type: DeviceType::CPU,
            compute_capability: 0.8,
            memory_bandwidth: 50.0, // GB/s
            memory_capacity: 8 * 1024 * 1024 * 1024, // 8 GB
            parallel_cores: 8,
            simd_width: 4,
            optimization_flags: vec![
                OptimizationFlag::SIMD,
                OptimizationFlag::MultiThreading,
                OptimizationFlag::MemoryOptimization,
            ],
            biological_accuracy_weight: 0.9,
            performance_weight: 0.1,
        }
    }

    /// Creates neuromorphic profile
    pub fn neuromorphic() -> Self {
        Self {
            name: "Neuromorphic Chip".to_string(),
            device_type: DeviceType::Neuromorphic,
            compute_capability: 0.5,
            memory_bandwidth: 10.0, // GB/s
            memory_capacity: 1 * 1024 * 1024 * 1024, // 1 GB
            parallel_cores: 1000000, // 1M neurons
            simd_width: 1,
            optimization_flags: vec![
                OptimizationFlag::MemoryOptimization,
            ],
            biological_accuracy_weight: 1.0,
            performance_weight: 0.0,
        }
    }
}

/// Experimental flags for cutting-edge features
#[derive(Debug, Clone)]
pub struct ExperimentalFlags {
    pub quantum_processing: bool,
    pub cross_species_models: bool,
    pub synesthetic_processing: bool,
    pub neuromorphic_acceleration: bool,
    pub quantum_visual_processing: bool,
    pub cross_modal_integration: bool,
    pub individual_calibration: bool,
    pub circadian_adaptation: bool,
    pub binocular_disparity: bool,
    pub visual_memory_integration: bool,
}

impl Default for ExperimentalFlags {
    fn default() -> Self {
        Self {
            quantum_processing: false,
            cross_species_models: false,
            synesthetic_processing: false,
            neuromorphic_acceleration: false,
            quantum_visual_processing: false,
            cross_modal_integration: false,
            individual_calibration: false,
            circadian_adaptation: false,
            binocular_disparity: false,
            visual_memory_integration: false,
        }
    }
}

/// User preferences for system behavior
#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub quality_priority: QualityPriority,
    pub performance_priority: PerformancePriority,
    pub biological_accuracy_priority: BiologicalAccuracyPriority,
    pub energy_efficiency_priority: EnergyEfficiencyPriority,
    pub real_time_processing: bool,
    pub debug_mode: bool,
    pub verbose_logging: bool,
    pub visualization_enabled: bool,
}

/// Quality priority levels
#[derive(Debug, Clone, Copy)]
pub enum QualityPriority {
    Low,
    Medium,
    High,
    Maximum,
}

/// Performance priority levels
#[derive(Debug, Clone, Copy)]
pub enum PerformancePriority {
    Low,
    Medium,
    High,
    Maximum,
}

/// Biological accuracy priority levels
#[derive(Debug, Clone, Copy)]
pub enum BiologicalAccuracyPriority {
    Low,
    Medium,
    High,
    Maximum,
}

/// Energy efficiency priority levels
#[derive(Debug, Clone, Copy)]
pub enum EnergyEfficiencyPriority {
    Low,
    Medium,
    High,
    Maximum,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            quality_priority: QualityPriority::High,
            performance_priority: PerformancePriority::Medium,
            biological_accuracy_priority: BiologicalAccuracyPriority::Maximum,
            energy_efficiency_priority: EnergyEfficiencyPriority::Medium,
            real_time_processing: true,
            debug_mode: false,
            verbose_logging: false,
            visualization_enabled: true,
        }
    }
}

/// Configuration validator
pub struct ConfigValidator {
    pub validation_rules: Vec<ConfigValidationRule>,
}

impl ConfigValidator {
    /// Creates a new configuration validator
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
        }
    }

    /// Validates biological configuration
    pub fn validate_biological_config(&self, config: &BiologicalConfig) -> Result<ValidationResult, AfiyahError> {
        let mut errors = Vec::new();
        
        // Validate retinal parameters
        if config.rod_sensitivity < 0.0 || config.rod_sensitivity > 1.0 {
            errors.push("Rod sensitivity must be between 0.0 and 1.0".to_string());
        }
        
        if config.cone_sensitivity < 0.0 || config.cone_sensitivity > 1.0 {
            errors.push("Cone sensitivity must be between 0.0 and 1.0".to_string());
        }
        
        if config.foveal_density <= 0.0 {
            errors.push("Foveal density must be positive".to_string());
        }
        
        if config.peripheral_density <= 0.0 {
            errors.push("Peripheral density must be positive".to_string());
        }
        
        // Validate cortical parameters
        if config.v1_orientation_selectivity < 0.0 || config.v1_orientation_selectivity > 1.0 {
            errors.push("V1 orientation selectivity must be between 0.0 and 1.0".to_string());
        }
        
        if config.cortical_magnification <= 0.0 {
            errors.push("Cortical magnification must be positive".to_string());
        }
        
        // Validate synaptic parameters
        if config.hebbian_learning_rate < 0.0 || config.hebbian_learning_rate > 1.0 {
            errors.push("Hebbian learning rate must be between 0.0 and 1.0".to_string());
        }
        
        // Validate perceptual parameters
        if config.masking_threshold < 0.0 || config.masking_threshold > 1.0 {
            errors.push("Masking threshold must be between 0.0 and 1.0".to_string());
        }
        
        if config.foveal_radius <= 0.0 {
            errors.push("Foveal radius must be positive".to_string());
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        })
    }

    /// Validates device profile
    pub fn validate_device_profile(&self, profile: &DeviceProfile) -> Result<ValidationResult, AfiyahError> {
        let mut errors = Vec::new();
        
        if profile.compute_capability < 0.0 {
            errors.push("Compute capability must be non-negative".to_string());
        }
        
        if profile.memory_bandwidth <= 0.0 {
            errors.push("Memory bandwidth must be positive".to_string());
        }
        
        if profile.memory_capacity == 0 {
            errors.push("Memory capacity must be positive".to_string());
        }
        
        if profile.parallel_cores == 0 {
            errors.push("Parallel cores must be positive".to_string());
        }
        
        if profile.biological_accuracy_weight < 0.0 || profile.biological_accuracy_weight > 1.0 {
            errors.push("Biological accuracy weight must be between 0.0 and 1.0".to_string());
        }
        
        if profile.performance_weight < 0.0 || profile.performance_weight > 1.0 {
            errors.push("Performance weight must be between 0.0 and 1.0".to_string());
        }
        
        if (profile.biological_accuracy_weight + profile.performance_weight - 1.0).abs() > 1e-6 {
            errors.push("Biological accuracy weight and performance weight must sum to 1.0".to_string());
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        })
    }
}

/// Configuration validation rule
#[derive(Debug, Clone)]
pub struct ConfigValidationRule {
    pub name: String,
    pub validation_function: fn(&BiologicalConfig) -> bool,
    pub error_message: String,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_biological_config_defaults() {
        let config = BiologicalConfig::default();
        assert_eq!(config.rod_sensitivity, 0.8);
        assert_eq!(config.cone_sensitivity, 0.6);
        assert_eq!(config.foveal_density, 200_000.0);
    }

    #[test]
    fn test_device_profile_creation() {
        let nvidia_profile = DeviceProfile::nvidia_gpu();
        assert_eq!(nvidia_profile.name, "NVIDIA GPU");
        assert_eq!(nvidia_profile.device_type, DeviceType::GPU);
    }

    #[test]
    fn test_config_validator() {
        let validator = ConfigValidator::new();
        let config = BiologicalConfig::default();
        let result = validator.validate_biological_config(&config);
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid);
    }
}
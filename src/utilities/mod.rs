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

//! Utilities Module
//! 
//! This module provides utility functions for logging, data loading, visualization,
//! and benchmarking across the Afiyah biomimetic video compression system.

pub mod logging;
pub mod data_loader;
pub mod visualization;
pub mod benchmarking;

use crate::AfiyahError;
use std::time::{Duration, Instant};

/// Main utilities manager that orchestrates all utility functions
pub struct UtilitiesManager {
    logger: logging::Logger,
    data_loader: data_loader::DataLoader,
    visualizer: visualization::Visualizer,
    benchmarker: benchmarking::Benchmarker,
}

impl UtilitiesManager {
    /// Creates a new utilities manager
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            logger: logging::Logger::new()?,
            data_loader: data_loader::DataLoader::new()?,
            visualizer: visualization::Visualizer::new()?,
            benchmarker: benchmarking::Benchmarker::new()?,
        })
    }

    /// Logs a message with specified level
    pub fn log(&mut self, level: LogLevel, message: &str) -> Result<(), AfiyahError> {
        self.logger.log(level, message)
    }

    /// Loads visual data from file
    pub fn load_visual_data(&self, filename: &str) -> Result<VisualData, AfiyahError> {
        self.data_loader.load_visual_data(filename)
    }

    /// Visualizes neural pathway data
    pub fn visualize_pathways(&self, pathway_data: &NeuralPathwayData) -> Result<VisualizationOutput, AfiyahError> {
        self.visualizer.visualize_pathways(pathway_data)
    }

    /// Benchmarks compression performance
    pub fn benchmark_compression(&self, input_data: &[u8], iterations: usize) -> Result<BenchmarkResults, AfiyahError> {
        self.benchmarker.benchmark_compression(input_data, iterations)
    }
}

/// Log levels for the system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Visual data structure
#[derive(Debug, Clone)]
pub struct VisualData {
    pub luminance_data: Vec<f64>,
    pub chromatic_data: Vec<ChromaticData>,
    pub spatial_resolution: (u32, u32),
    pub temporal_resolution: f64,
    pub metadata: VisualMetadata,
}

/// Chromatic data for color processing
#[derive(Debug, Clone)]
pub struct ChromaticData {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

/// Visual metadata
#[derive(Debug, Clone)]
pub struct VisualMetadata {
    pub timestamp: u64,
    pub frame_number: u32,
    pub quality_hint: f64,
    pub attention_region: Option<AttentionRegion>,
}

/// Attention region for foveal processing
#[derive(Debug, Clone)]
pub struct AttentionRegion {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
    pub priority: f64,
}

/// Neural pathway data for visualization
#[derive(Debug, Clone)]
pub struct NeuralPathwayData {
    pub retinal_pathways: Vec<RetinalPathway>,
    pub cortical_pathways: Vec<CorticalPathway>,
    pub synaptic_connections: Vec<SynapticConnection>,
    pub activity_levels: Vec<f64>,
}

/// Retinal pathway data
#[derive(Debug, Clone)]
pub struct RetinalPathway {
    pub pathway_type: RetinalPathwayType,
    pub activity: f64,
    pub spatial_frequency: f64,
    pub temporal_frequency: f64,
    pub adaptation_level: f64,
}

/// Retinal pathway types
#[derive(Debug, Clone, Copy)]
pub enum RetinalPathwayType {
    Magnocellular,
    Parvocellular,
    Koniocellular,
}

/// Cortical pathway data
#[derive(Debug, Clone)]
pub struct CorticalPathway {
    pub cortical_area: CorticalArea,
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub activity: f64,
    pub selectivity: f64,
}

/// Cortical areas
#[derive(Debug, Clone, Copy)]
pub enum CorticalArea {
    V1,
    V2,
    V3,
    V4,
    V5,
    MT,
    MST,
}

/// Synaptic connection data
#[derive(Debug, Clone)]
pub struct SynapticConnection {
    pub from_neuron: usize,
    pub to_neuron: usize,
    pub weight: f64,
    pub strength: f64,
    pub plasticity: f64,
}

/// Visualization output
#[derive(Debug, Clone)]
pub struct VisualizationOutput {
    pub pathway_visualization: Vec<u8>,
    pub activity_heatmap: Vec<u8>,
    pub connection_graph: Vec<u8>,
    pub temporal_evolution: Vec<u8>,
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub compression_ratio: f64,
    pub compression_time: Duration,
    pub decompression_time: Duration,
    pub memory_usage: usize,
    pub cpu_usage: f64,
    pub gpu_usage: Option<f64>,
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for benchmarking
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub psnr: f64,
    pub ssim: f64,
    pub vmaf: f64,
    pub biological_accuracy: f64,
    pub perceptual_quality: f64,
}

/// Performance profiler
pub struct PerformanceProfiler {
    pub start_time: Option<Instant>,
    pub measurements: Vec<PerformanceMeasurement>,
}

impl PerformanceProfiler {
    /// Creates a new performance profiler
    pub fn new() -> Self {
        Self {
            start_time: None,
            measurements: Vec::new(),
        }
    }

    /// Starts profiling a section
    pub fn start_section(&mut self, section_name: &str) {
        self.start_time = Some(Instant::now());
        self.measurements.push(PerformanceMeasurement {
            section_name: section_name.to_string(),
            duration: Duration::from_secs(0),
            memory_usage: 0,
            cpu_usage: 0.0,
        });
    }

    /// Ends profiling a section
    pub fn end_section(&mut self) -> Result<(), AfiyahError> {
        if let Some(start) = self.start_time {
            let duration = start.elapsed();
            if let Some(last_measurement) = self.measurements.last_mut() {
                last_measurement.duration = duration;
            }
            self.start_time = None;
        }
        Ok(())
    }

    /// Gets profiling results
    pub fn get_results(&self) -> &[PerformanceMeasurement] {
        &self.measurements
    }
}

/// Performance measurement
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub section_name: String,
    pub duration: Duration,
    pub memory_usage: usize,
    pub cpu_usage: f64,
}

/// Data validator
pub struct DataValidator {
    pub validation_rules: Vec<ValidationRule>,
}

impl DataValidator {
    /// Creates a new data validator
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
        }
    }

    /// Adds a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    /// Validates data against all rules
    pub fn validate(&self, data: &[f64]) -> Result<ValidationResult, AfiyahError> {
        let mut results = Vec::new();
        let mut is_valid = true;

        for rule in &self.validation_rules {
            let result = rule.validate(data);
            results.push(result.clone());
            if !result.is_valid {
                is_valid = false;
            }
        }

        Ok(ValidationResult {
            is_valid,
            results,
        })
    }
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub name: String,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub required_length: Option<usize>,
    pub tolerance: f64,
}

impl ValidationRule {
    /// Creates a new validation rule
    pub fn new(name: String) -> Self {
        Self {
            name,
            min_value: None,
            max_value: None,
            required_length: None,
            tolerance: 1e-6,
        }
    }

    /// Validates data against this rule
    pub fn validate(&self, data: &[f64]) -> RuleValidationResult {
        let mut errors = Vec::new();

        // Check length
        if let Some(required_len) = self.required_length {
            if data.len() != required_len {
                errors.push(format!("Expected length {}, got {}", required_len, data.len()));
            }
        }

        // Check value ranges
        for (i, &value) in data.iter().enumerate() {
            if let Some(min_val) = self.min_value {
                if value < min_val - self.tolerance {
                    errors.push(format!("Value at index {} ({}) is below minimum ({})", i, value, min_val));
                }
            }
            if let Some(max_val) = self.max_value {
                if value > max_val + self.tolerance {
                    errors.push(format!("Value at index {} ({}) is above maximum ({})", i, value, max_val));
                }
            }
        }

        RuleValidationResult {
            rule_name: self.name.clone(),
            is_valid: errors.is_empty(),
            errors,
        }
    }
}

/// Rule validation result
#[derive(Debug, Clone)]
pub struct RuleValidationResult {
    pub rule_name: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub results: Vec<RuleValidationResult>,
}

/// Memory manager
pub struct MemoryManager {
    pub allocated_memory: usize,
    pub max_memory: usize,
    pub memory_usage: Vec<MemoryUsage>,
}

impl MemoryManager {
    /// Creates a new memory manager
    pub fn new(max_memory: usize) -> Self {
        Self {
            allocated_memory: 0,
            max_memory,
            memory_usage: Vec::new(),
        }
    }

    /// Allocates memory
    pub fn allocate(&mut self, size: usize) -> Result<usize, AfiyahError> {
        if self.allocated_memory + size > self.max_memory {
            return Err(AfiyahError::HardwareError("Insufficient memory".to_string()));
        }
        
        self.allocated_memory += size;
        self.memory_usage.push(MemoryUsage {
            size,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
        
        Ok(self.allocated_memory)
    }

    /// Deallocates memory
    pub fn deallocate(&mut self, size: usize) {
        if self.allocated_memory >= size {
            self.allocated_memory -= size;
        }
    }

    /// Gets memory usage statistics
    pub fn get_usage_stats(&self) -> MemoryStats {
        MemoryStats {
            allocated_memory: self.allocated_memory,
            max_memory: self.max_memory,
            utilization_rate: self.allocated_memory as f64 / self.max_memory as f64,
            allocation_count: self.memory_usage.len(),
        }
    }
}

/// Memory usage record
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub size: usize,
    pub timestamp: u64,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_memory: usize,
    pub max_memory: usize,
    pub utilization_rate: f64,
    pub allocation_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utilities_manager_creation() {
        let manager = UtilitiesManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_section("test_section");
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.end_section().unwrap();
        let results = profiler.get_results();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_data_validator() {
        let mut validator = DataValidator::new();
        let rule = ValidationRule::new("test_rule".to_string());
        validator.add_rule(rule);
        let data = vec![1.0, 2.0, 3.0];
        let result = validator.validate(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_manager() {
        let mut manager = MemoryManager::new(1000);
        let result = manager.allocate(100);
        assert!(result.is_ok());
        assert_eq!(manager.allocated_memory, 100);
    }
}
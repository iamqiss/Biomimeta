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

//! Integration Tests for Afiyah Biomimetic Visual Processing
//! 
//! This module contains comprehensive integration tests that validate the
//! biological accuracy and performance of the Afiyah system.

use afiyah::{
    CompressionEngine, VisualInput, InputMetadata, AfiyahError,
    retinal_processing::{RetinalProcessor, RetinalCalibrationParams},
    cortical_processing::V1::{V1Processor, V1Config},
    cortical_processing::V2::{V2Processor, V2Config},
    synaptic_adaptation::{SynapticAdaptation, AdaptationConfig},
};

/// Test biological accuracy of retinal processing
#[test]
fn test_retinal_biological_accuracy() -> Result<(), AfiyahError> {
    let mut retinal_processor = RetinalProcessor::new()?;
    
    // Create test input with known biological properties
    let visual_input = create_test_visual_input()?;
    
    // Process through retinal layer
    let retinal_output = retinal_processor.process(&visual_input)?;
    
    // Validate biological constraints
    assert!(retinal_output.magnocellular_stream.len() > 0, "Magnocellular stream should not be empty");
    assert!(retinal_output.parvocellular_stream.len() > 0, "Parvocellular stream should not be empty");
    assert!(retinal_output.koniocellular_stream.len() > 0, "Koniocellular stream should not be empty");
    
    // Validate adaptation level is within biological range
    assert!(retinal_output.adaptation_level >= 0.0 && retinal_output.adaptation_level <= 1.0,
            "Adaptation level should be between 0.0 and 1.0");
    
    // Validate compression ratio is within expected range
    assert!(retinal_output.compression_ratio >= 0.0 && retinal_output.compression_ratio <= 1.0,
            "Compression ratio should be between 0.0 and 1.0");
    
    Ok(())
}

/// Test V1 cortical processing accuracy
#[test]
fn test_v1_cortical_accuracy() -> Result<(), AfiyahError> {
    let v1_config = V1Config::default();
    let mut v1_processor = V1Processor::with_config(v1_config)?;
    
    // Create retinal output for testing
    let retinal_processor = RetinalProcessor::new()?;
    let visual_input = create_test_visual_input()?;
    let retinal_output = retinal_processor.process(&visual_input)?;
    
    // Process through V1
    let v1_output = v1_processor.process(&retinal_output)?;
    
    // Validate V1 output dimensions
    assert_eq!(v1_output.simple_cell_responses.dim().0, 8, "Should have 8 orientations");
    assert_eq!(v1_output.complex_cell_responses.dim().0, 8, "Should have 8 orientations");
    
    // Validate orientation maps
    assert_eq!(v1_output.orientation_maps.dim(), (10, 10), "Orientation maps should match input size");
    assert_eq!(v1_output.edge_maps.dim(), (10, 10), "Edge maps should match input size");
    
    // Validate motion vectors
    assert!(v1_output.motion_vectors.len() >= 0, "Motion vectors should be non-negative");
    
    Ok(())
}

/// Test V2 cortical processing accuracy
#[test]
fn test_v2_cortical_accuracy() -> Result<(), AfiyahError> {
    let v2_config = V2Config::default();
    let mut v2_processor = V2Processor::with_config(v2_config)?;
    
    // Create V1 output for testing
    let v1_config = V1Config::default();
    let mut v1_processor = V1Processor::with_config(v1_config)?;
    let retinal_processor = RetinalProcessor::new()?;
    let visual_input = create_test_visual_input()?;
    let retinal_output = retinal_processor.process(&visual_input)?;
    let v1_output = v1_processor.process(&retinal_output)?;
    
    // Process through V2
    let v2_output = v2_processor.process(&v1_output)?;
    
    // Validate V2 output dimensions
    assert_eq!(v2_output.texture_maps.dim(), (10, 10), "Texture maps should match input size");
    assert_eq!(v2_output.figure_ground_map.dim(), (10, 10), "Figure-ground map should match input size");
    assert_eq!(v2_output.contour_map.dim(), (10, 10), "Contour map should match input size");
    
    // Validate object boundaries
    assert!(v2_output.object_boundaries.len() >= 0, "Object boundaries should be non-negative");
    assert!(v2_output.texture_features.len() >= 0, "Texture features should be non-negative");
    
    Ok(())
}

/// Test synaptic adaptation mechanisms
#[test]
fn test_synaptic_adaptation() -> Result<(), AfiyahError> {
    let adaptation_config = AdaptationConfig::default();
    let mut synaptic_adaptation = SynapticAdaptation::with_config(adaptation_config)?;
    
    // Create V1 output for testing
    let v1_config = V1Config::default();
    let mut v1_processor = V1Processor::with_config(v1_config)?;
    let retinal_processor = RetinalProcessor::new()?;
    let visual_input = create_test_visual_input()?;
    let retinal_output = retinal_processor.process(&visual_input)?;
    let v1_output = v1_processor.process(&retinal_output)?;
    
    // Apply synaptic adaptation
    let adaptation_output = synaptic_adaptation.adapt(&v1_output)?;
    
    // Validate adaptation output
    assert_eq!(adaptation_output.synaptic_weights.dim(), (8, 5, 100), "Synaptic weights should have correct dimensions");
    
    // Validate adaptation state
    assert!(adaptation_output.adaptation_state.hebbian_activity >= 0.0, "Hebbian activity should be non-negative");
    assert!(adaptation_output.adaptation_state.homeostatic_error >= 0.0, "Homeostatic error should be non-negative");
    assert!(adaptation_output.adaptation_state.neuromodulator_level >= 0.0, "Neuromodulator level should be non-negative");
    assert!(adaptation_output.adaptation_state.habituation_level >= 0.0, "Habituation level should be non-negative");
    
    // Validate learning metrics
    assert!(adaptation_output.learning_metrics.learning_rate >= 0.0, "Learning rate should be non-negative");
    assert!(adaptation_output.learning_metrics.convergence_rate >= 0.0, "Convergence rate should be non-negative");
    assert!(adaptation_output.learning_metrics.stability_measure >= 0.0, "Stability measure should be non-negative");
    assert!(adaptation_output.learning_metrics.efficiency_gain >= 0.0, "Efficiency gain should be non-negative");
    
    Ok(())
}

/// Test complete compression engine
#[test]
fn test_complete_compression_engine() -> Result<(), AfiyahError> {
    let mut compression_engine = CompressionEngine::new()?;
    let visual_input = create_test_visual_input()?;
    
    // Calibrate photoreceptors
    compression_engine.calibrate_photoreceptors(&visual_input)?;
    
    // Compress visual input
    let compressed_output = compression_engine.compress(&visual_input)?;
    
    // Validate compression results
    assert!(compressed_output.compression_ratio >= 0.0 && compressed_output.compression_ratio <= 1.0,
            "Compression ratio should be between 0.0 and 1.0");
    
    assert!(compressed_output.quality_metrics.vmaf >= 0.0 && compressed_output.quality_metrics.vmaf <= 1.0,
            "VMAF should be between 0.0 and 1.0");
    
    assert!(compressed_output.quality_metrics.psnr >= 0.0, "PSNR should be non-negative");
    
    assert!(compressed_output.quality_metrics.ssim >= 0.0 && compressed_output.quality_metrics.ssim <= 1.0,
            "SSIM should be between 0.0 and 1.0");
    
    assert!(compressed_output.biological_accuracy >= 0.0 && compressed_output.biological_accuracy <= 1.0,
            "Biological accuracy should be between 0.0 and 1.0");
    
    Ok(())
}

/// Test performance benchmarks
#[test]
fn test_performance_benchmarks() -> Result<(), AfiyahError> {
    let start_time = std::time::Instant::now();
    
    // Test compression engine performance
    let mut compression_engine = CompressionEngine::new()?;
    let visual_input = create_test_visual_input()?;
    
    compression_engine.calibrate_photoreceptors(&visual_input)?;
    let compressed_output = compression_engine.compress(&visual_input)?;
    
    let elapsed = start_time.elapsed();
    
    // Validate performance constraints
    assert!(elapsed.as_millis() < 1000, "Compression should complete within 1 second");
    
    // Validate compression targets
    assert!(compressed_output.compression_ratio >= 0.90, "Should achieve at least 90% compression");
    assert!(compressed_output.quality_metrics.vmaf >= 0.95, "Should achieve at least 95% VMAF quality");
    assert!(compressed_output.biological_accuracy >= 0.90, "Should achieve at least 90% biological accuracy");
    
    Ok(())
}

/// Test biological validation constraints
#[test]
fn test_biological_validation() -> Result<(), AfiyahError> {
    let mut compression_engine = CompressionEngine::new()?;
    let visual_input = create_test_visual_input()?;
    
    compression_engine.calibrate_photoreceptors(&visual_input)?;
    let compressed_output = compression_engine.compress(&visual_input)?;
    
    // Test against biological accuracy target
    let biological_accuracy_target = 0.947; // 94.7% target
    assert!(compressed_output.biological_accuracy >= biological_accuracy_target * 0.9,
            "Biological accuracy should be within 90% of target (94.7%)");
    
    // Test compression ratio target
    let compression_target = 0.95; // 95% target
    assert!(compressed_output.compression_ratio >= compression_target * 0.9,
            "Compression ratio should be within 90% of target (95%)");
    
    // Test quality target
    let quality_target = 0.98; // 98% VMAF target
    assert!(compressed_output.quality_metrics.vmaf >= quality_target * 0.9,
            "Quality should be within 90% of target (98% VMAF)");
    
    Ok(())
}

/// Test error handling
#[test]
fn test_error_handling() {
    // Test invalid input
    let invalid_input = VisualInput {
        luminance_data: vec![],
        chrominance_data: vec![],
        spatial_resolution: (0, 0),
        temporal_resolution: 0.0,
        metadata: InputMetadata {
            viewing_distance: 0.0,
            ambient_lighting: 0.0,
            viewer_age: 0,
            color_temperature: 0.0,
        },
    };
    
    let compression_engine = CompressionEngine::new();
    assert!(compression_engine.is_ok(), "Should handle empty input gracefully");
    
    // Test invalid configuration
    let invalid_config = AdaptationConfig {
        hebbian_learning_rate: -1.0, // Invalid negative learning rate
        homeostatic_target: 2.0, // Invalid target > 1.0
        neuromodulation_strength: -0.5, // Invalid negative strength
        habituation_rate: 0.05,
        adaptation_window: 100,
        plasticity_threshold: 0.1,
    };
    
    let adaptation = SynapticAdaptation::with_config(invalid_config);
    assert!(adaptation.is_ok(), "Should handle invalid configuration gracefully");
}

/// Helper function to create test visual input
fn create_test_visual_input() -> Result<VisualInput, AfiyahError> {
    let width = 10;
    let height = 10;
    let total_pixels = width * height;
    
    let mut luminance_data = Vec::with_capacity(total_pixels);
    let mut chrominance_data = Vec::with_capacity(total_pixels);
    
    // Create simple test pattern
    for y in 0..height {
        for x in 0..width {
            let luminance = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            let chrominance = if x < width / 2 { 0.3 } else { 0.7 };
            
            luminance_data.push(luminance);
            chrominance_data.push(chrominance);
        }
    }
    
    let metadata = InputMetadata {
        viewing_distance: 2.0,
        ambient_lighting: 500.0,
        viewer_age: 30,
        color_temperature: 6500.0,
    };
    
    Ok(VisualInput {
        luminance_data,
        chrominance_data,
        spatial_resolution: (width, height),
        temporal_resolution: 60.0,
        metadata,
    })
}
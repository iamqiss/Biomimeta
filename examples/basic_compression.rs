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

//! Basic Compression Example
//! 
//! This example demonstrates the basic usage of the Afiyah biomimetic video
//! compression system, showing how to compress visual input using the complete
//! biological processing pipeline.

use afiyah::{
    CompressionEngine, VisualInput, ChromaticData, VisualMetadata, AttentionRegion,
    AfiyahError, AfiyahResult
};
use std::time::Instant;

fn main() -> AfiyahResult<()> {
    println!("🧠👁️ Afiyah Biomimetic Video Compression System");
    println!("================================================");
    
    // Create compression engine
    println!("Creating compression engine...");
    let mut engine = CompressionEngine::new()?;
    println!("✅ Compression engine created successfully");
    
    // Create sample visual input
    println!("\nCreating sample visual input...");
    let visual_input = create_sample_visual_input()?;
    println!("✅ Sample visual input created ({}x{} pixels)", 
             visual_input.spatial_resolution.0, visual_input.spatial_resolution.1);
    
    // Calibrate photoreceptors
    println!("\nCalibrating photoreceptors...");
    engine.calibrate_photoreceptors(&visual_input)?;
    println!("✅ Photoreceptors calibrated");
    
    // Train cortical filters (simplified training)
    println!("\nTraining cortical filters...");
    let training_data = create_training_data()?;
    engine.train_cortical_filters(&training_data)?;
    println!("✅ Cortical filters trained");
    
    // Configure compression engine
    println!("\nConfiguring compression engine...");
    let engine = engine
        .with_saccadic_prediction(true)
        .with_foveal_attention(true)
        .with_temporal_integration(200);
    println!("✅ Compression engine configured");
    
    // Perform compression
    println!("\nPerforming biomimetic compression...");
    let start_time = Instant::now();
    let compressed_output = engine.compress(&visual_input)?;
    let compression_time = start_time.elapsed();
    
    println!("✅ Compression completed in {:?}", compression_time);
    println!("📊 Compression Results:");
    println!("   - Compression Ratio: {:.2}%", compressed_output.compression_ratio * 100.0);
    println!("   - Perceptual Quality: {:.2}%", compressed_output.perceptual_quality * 100.0);
    println!("   - Biological Accuracy: {:.2}%", compressed_output.biological_accuracy * 100.0);
    
    // Save compressed output
    println!("\nSaving compressed output...");
    compressed_output.save("output.afiyah")?;
    println!("✅ Compressed output saved to 'output.afiyah'");
    
    // Demonstrate streaming
    println!("\nDemonstrating streaming capabilities...");
    demonstrate_streaming(&compressed_output)?;
    
    println!("\n🎉 Example completed successfully!");
    println!("The Afiyah system has successfully compressed visual input using");
    println!("biomimetic processing that mimics the human visual system.");
    
    Ok(())
}

/// Creates a sample visual input for demonstration
fn create_sample_visual_input() -> AfiyahResult<VisualInput> {
    let width = 128;
    let height = 128;
    let total_pixels = (width * height) as usize;
    
    // Create sample luminance data (grayscale)
    let mut luminance_data = Vec::with_capacity(total_pixels);
    for y in 0..height {
        for x in 0..width {
            // Create a simple gradient pattern
            let intensity = ((x as f64 + y as f64) / (width as f64 + height as f64)) * 255.0;
            luminance_data.push(intensity / 255.0); // Normalize to 0-1
        }
    }
    
    // Create sample chromatic data (RGB)
    let mut chromatic_data = Vec::with_capacity(total_pixels);
    for y in 0..height {
        for x in 0..width {
            // Create a color pattern
            let red = (x as f64 / width as f64) * 255.0;
            let green = (y as f64 / height as f64) * 255.0;
            let blue = ((x as f64 + y as f64) / (width as f64 + height as f64)) * 255.0;
            
            chromatic_data.push(ChromaticData {
                red: red / 255.0, // Normalize to 0-1
                green: green / 255.0,
                blue: blue / 255.0,
                alpha: 1.0,
            });
        }
    }
    
    // Create attention region (center of image)
    let attention_region = AttentionRegion {
        center_x: 0.5,
        center_y: 0.5,
        radius: 0.3,
        priority: 0.9,
    };
    
    // Create metadata
    let metadata = VisualMetadata {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        frame_number: 0,
        quality_hint: 1.0,
        attention_region: Some(attention_region),
    };
    
    Ok(VisualInput {
        luminance_data,
        chromatic_data,
        spatial_resolution: (width, height),
        temporal_resolution: 30.0, // 30 FPS
        metadata,
    })
}

/// Creates training data for cortical filter training
fn create_training_data() -> AfiyahResult<Vec<VisualInput>> {
    let mut training_data = Vec::new();
    
    // Create multiple training samples with different patterns
    for i in 0..10 {
        let width = 64;
        let height = 64;
        let total_pixels = (width * height) as usize;
        
        // Create different patterns for training
        let mut luminance_data = Vec::with_capacity(total_pixels);
        let mut chromatic_data = Vec::with_capacity(total_pixels);
        
        for y in 0..height {
            for x in 0..width {
                // Create different patterns based on training index
                let pattern_value = match i % 4 {
                    0 => (x as f64 / width as f64), // Horizontal gradient
                    1 => (y as f64 / height as f64), // Vertical gradient
                    2 => ((x as f64 + y as f64) / (width as f64 + height as f64)), // Diagonal gradient
                    _ => ((x as f64 - width as f64 / 2.0).powi(2) + (y as f64 - height as f64 / 2.0).powi(2)).sqrt() / (width as f64 / 2.0), // Radial pattern
                };
                
                luminance_data.push(pattern_value);
                
                chromatic_data.push(ChromaticData {
                    red: pattern_value,
                    green: pattern_value * 0.8,
                    blue: pattern_value * 0.6,
                    alpha: 1.0,
                });
            }
        }
        
        let attention_region = AttentionRegion {
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.2,
            priority: 0.8,
        };
        
        let metadata = VisualMetadata {
            timestamp: i as u64,
            frame_number: i,
            quality_hint: 1.0,
            attention_region: Some(attention_region),
        };
        
        training_data.push(VisualInput {
            luminance_data,
            chromatic_data,
            spatial_resolution: (width, height),
            temporal_resolution: 30.0,
            metadata,
        });
    }
    
    Ok(training_data)
}

/// Demonstrates streaming capabilities
fn demonstrate_streaming(compressed_output: &afiyah::CompressedOutput) -> AfiyahResult<()> {
    println!("   - Streaming Quality: {:.2}%", compressed_output.perceptual_quality * 100.0);
    println!("   - Data Size: {} bytes", compressed_output.data.len());
    println!("   - Estimated Bandwidth: {:.2} Mbps", 
             (compressed_output.data.len() as f64 * 8.0) / 1_000_000.0);
    
    // Simulate streaming metrics
    let compression_ratio = compressed_output.compression_ratio;
    let quality = compressed_output.perceptual_quality;
    let biological_accuracy = compressed_output.biological_accuracy;
    
    println!("   - Compression Efficiency: {:.2}%", compression_ratio * 100.0);
    println!("   - Biological Fidelity: {:.2}%", biological_accuracy * 100.0);
    
    if quality > 0.9 {
        println!("   ✅ Excellent perceptual quality maintained");
    } else if quality > 0.8 {
        println!("   ✅ Good perceptual quality maintained");
    } else {
        println!("   ⚠️  Perceptual quality could be improved");
    }
    
    if biological_accuracy > 0.9 {
        println!("   ✅ High biological accuracy achieved");
    } else if biological_accuracy > 0.8 {
        println!("   ✅ Good biological accuracy achieved");
    } else {
        println!("   ⚠️  Biological accuracy could be improved");
    }
    
    Ok(())
}

/// Demonstrates advanced features
fn demonstrate_advanced_features() -> AfiyahResult<()> {
    println!("\n🔬 Advanced Features Demonstration");
    println!("==================================");
    
    // This would demonstrate experimental features like:
    // - Quantum visual processing
    // - Cross-species visual models
    // - Synesthetic processing
    // - Neuromorphic acceleration
    
    println!("Advanced features would be demonstrated here in a full implementation.");
    println!("These include quantum processing, cross-species models, and more.");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_visual_input_creation() {
        let input = create_sample_visual_input();
        assert!(input.is_ok());
        let input = input.unwrap();
        assert_eq!(input.spatial_resolution, (128, 128));
        assert_eq!(input.luminance_data.len(), 128 * 128);
    }

    #[test]
    fn test_training_data_creation() {
        let training_data = create_training_data();
        assert!(training_data.is_ok());
        let training_data = training_data.unwrap();
        assert_eq!(training_data.len(), 10);
    }
}
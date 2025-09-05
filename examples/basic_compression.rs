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
//! This example demonstrates basic usage of the Afiyah compression engine
//! with simple visual input and compression parameters.

use afiyah::{CompressionEngine, VisualInput, InputMetadata, AfiyahError};

fn main() -> Result<(), AfiyahError> {
    println!("🧠👁️ Afiyah Basic Compression Demo");
    println!("===================================");
    
    // Create simple visual input
    let visual_input = create_simple_visual_input()?;
    println!("✅ Created visual input: {}x{} pixels", 
             visual_input.spatial_resolution.0, visual_input.spatial_resolution.1);
    
    // Create compression engine
    let mut engine = CompressionEngine::new()?;
    println!("✅ Initialized compression engine");
    
    // Configure engine
    let engine = engine
        .with_saccadic_prediction(true)
        .with_foveal_attention(true)
        .with_temporal_integration(200); // milliseconds
    println!("✅ Configured engine with biological parameters");
    
    // Calibrate photoreceptors
    engine.calibrate_photoreceptors(&visual_input)?;
    println!("✅ Calibrated photoreceptors");
    
    // Compress visual input
    let compressed_output = engine.compress(&visual_input)?;
    println!("✅ Compression complete!");
    
    // Display results
    println!("\n📊 Compression Results");
    println!("=====================");
    println!("Compression ratio: {:.1}%", compressed_output.compression_ratio * 100.0);
    println!("VMAF quality: {:.1}%", compressed_output.quality_metrics.vmaf * 100.0);
    println!("PSNR: {:.1} dB", compressed_output.quality_metrics.psnr);
    println!("SSIM: {:.3}", compressed_output.quality_metrics.ssim);
    println!("Biological accuracy: {:.1}%", compressed_output.biological_accuracy * 100.0);
    
    // Save compressed output
    compressed_output.save("output.afiyah")?;
    println!("✅ Saved compressed output to output.afiyah");
    
    println!("\n🎉 Basic compression demo complete!");
    
    Ok(())
}

/// Creates simple visual input for demonstration
fn create_simple_visual_input() -> Result<VisualInput, AfiyahError> {
    let width = 64;
    let height = 64;
    let total_pixels = width * height;
    
    let mut luminance_data = Vec::with_capacity(total_pixels);
    let mut chrominance_data = Vec::with_capacity(total_pixels);
    
    // Create simple checkerboard pattern
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
        ambient_lighting: 300.0,
        viewer_age: 25,
        color_temperature: 6000.0,
    };
    
    Ok(VisualInput {
        luminance_data,
        chrominance_data,
        spatial_resolution: (width, height),
        temporal_resolution: 30.0,
        metadata,
    })
}
//! Advanced Cortical Areas Example
//! 
//! This example demonstrates the advanced cortical processing areas including
//! V1 orientation filters, V5/MT motion processing, and real-time adaptation.

use afiyah::{
    VisualSystem, CompressionEngine, CompressionConfig,
    cortical_processing::{CorticalProcessor, CorticalProcessingConfig},
    retinal_processing::RetinalOutput,
    VisualInput,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Afiyah Advanced Cortical Areas Demo");
    println!("=====================================");
    
    // Create visual system with advanced cortical processing
    let mut visual_system = VisualSystem::new()?;
    
    // Create compression engine with biological configuration
    let config = CompressionConfig::biological_default();
    let mut compression_engine = CompressionEngine::new(config)?;
    
    // Create cortical processor with advanced features
    let mut cortical_processor = CorticalProcessor::new()?;
    
    // Configure cortical processing
    let cortical_config = CorticalProcessingConfig {
        orientation_enabled: true,
        motion_enabled: true,
        adaptation_enabled: true,
        processing_resolution: (128, 128),
        adaptation_window_ms: 200,
    };
    cortical_processor.set_config(cortical_config);
    
    // Enable real-time adaptation
    cortical_processor.set_adaptation_enabled(true);
    
    println!("✅ Advanced cortical areas initialized");
    println!("   - V1 Orientation Filters: 8 orientations");
    println!("   - V5/MT Motion Processing: 8 directions");
    println!("   - Real-time Adaptation: Enabled");
    
    // Create sample visual input
    let visual_input = create_sample_visual_input();
    
    println!("\n📊 Processing visual input through cortical pipeline...");
    
    // Process through retinal stage
    let retinal_output = visual_system.retinal_processor.process(&visual_input)?;
    println!("✅ Retinal processing complete");
    println!("   - Magnocellular stream: {} samples", retinal_output.magnocellular_stream.len());
    println!("   - Parvocellular stream: {} samples", retinal_output.parvocellular_stream.len());
    println!("   - Koniocellular stream: {} samples", retinal_output.koniocellular_stream.len());
    println!("   - Adaptation level: {:.3}", retinal_output.adaptation_level);
    println!("   - Compression ratio: {:.1}%", retinal_output.compression_ratio * 100.0);
    
    // Process through cortical stage
    let cortical_output = cortical_processor.process(&retinal_output)?;
    println!("✅ Cortical processing complete");
    println!("   - Orientation maps: {}", cortical_output.orientation_maps.len());
    println!("   - Motion vectors: {}", cortical_output.motion_vectors.len());
    println!("   - Depth maps: {}", cortical_output.depth_maps.len());
    println!("   - Cortical compression: {:.1}%", cortical_output.cortical_compression * 100.0);
    
    // Display orientation analysis
    display_orientation_analysis(&cortical_output);
    
    // Display motion analysis
    display_motion_analysis(&cortical_output);
    
    // Display adaptation analysis
    display_adaptation_analysis(&cortical_output);
    
    // Complete visual processing pipeline
    let visual_output = visual_system.process_visual_input(&visual_input)?;
    println!("\n🎯 Complete visual processing pipeline");
    println!("   - Final compression ratio: {:.1}%", visual_output.compression_ratio * 100.0);
    println!("   - Perceptual quality: {:.1}%", visual_output.perceptual_quality * 100.0);
    
    println!("\n✨ Advanced cortical areas demo completed successfully!");
    
    Ok(())
}

/// Creates sample visual input for demonstration
fn create_sample_visual_input() -> VisualInput {
    // Create a 128x128 visual input with varying characteristics
    let width = 128;
    let height = 128;
    let total_pixels = (width * height) as usize;
    
    let mut luminance_data = Vec::with_capacity(total_pixels);
    let mut chromatic_data = Vec::with_capacity(total_pixels * 2);
    let mut temporal_data = Vec::with_capacity(10);
    
    // Generate luminance data with orientation patterns
    for y in 0..height {
        for x in 0..width {
            let x_norm = x as f64 / width as f64;
            let y_norm = y as f64 / height as f64;
            
            // Create diagonal orientation pattern
            let orientation_value = (x_norm + y_norm).sin() * 0.5 + 0.5;
            
            // Add some noise for realism
            let noise = (rand::random::<f64>() - 0.5) * 0.1;
            
            luminance_data.push((orientation_value + noise).max(0.0).min(1.0));
        }
    }
    
    // Generate chromatic data
    for i in 0..total_pixels {
        let red = (i as f64 / total_pixels as f64).sin() * 0.5 + 0.5;
        let green = (i as f64 / total_pixels as f64).cos() * 0.5 + 0.5;
        chromatic_data.push(red);
        chromatic_data.push(green);
    }
    
    // Generate temporal data (simulating motion)
    for i in 0..10 {
        let temporal_value = (i as f64 / 10.0).sin() * 0.5 + 0.5;
        temporal_data.push(temporal_value);
    }
    
    VisualInput {
        luminance_data,
        chromatic_data,
        temporal_data,
        spatial_resolution: (width, height),
        temporal_resolution: 60,
    }
}

/// Displays orientation analysis results
fn display_orientation_analysis(cortical_output: &afiyah::CorticalOutput) {
    println!("\n📐 Orientation Analysis (V1)");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ Orientation Maps Analysis               │");
    println!("   ├─────────────────────────────────────────┤");
    
    for (i, map) in cortical_output.orientation_maps.iter().enumerate() {
        let orientation_deg = (map.orientation * 180.0 / std::f64::consts::PI) as i32;
        let strength_percent = (map.strength * 100.0) as i32;
        
        println!("   │ {:2}°: {:3}% strength", orientation_deg, strength_percent);
    }
    
    println!("   └─────────────────────────────────────────┘");
}

/// Displays motion analysis results
fn display_motion_analysis(cortical_output: &afiyah::CorticalOutput) {
    println!("\n🎬 Motion Analysis (V5/MT)");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ Motion Vectors Analysis                │");
    println!("   ├─────────────────────────────────────────┤");
    
    for (i, vector) in cortical_output.motion_vectors.iter().enumerate() {
        let direction_deg = (vector.direction * 180.0 / std::f64::consts::PI) as i32;
        let magnitude_percent = (vector.magnitude * 100.0) as i32;
        let confidence_percent = (vector.confidence * 100.0) as i32;
        
        println!("   │ {:3}°: {:3}% magnitude, {:3}% confidence", 
                 direction_deg, magnitude_percent, confidence_percent);
    }
    
    println!("   └─────────────────────────────────────────┘");
}

/// Displays adaptation analysis results
fn display_adaptation_analysis(cortical_output: &afiyah::CorticalOutput) {
    println!("\n🔄 Real-time Adaptation Analysis");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ Adaptation Status                       │");
    println!("   ├─────────────────────────────────────────┤");
    println!("   │ Cortical compression: {:5.1}%", 
             cortical_output.cortical_compression * 100.0);
    println!("   │ Processing efficiency: {:5.1}%", 
             (1.0 - cortical_output.cortical_compression) * 100.0);
    println!("   └─────────────────────────────────────────┘");
}

// Simple random number generator for demo purposes
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    static mut SEED: u64 = 0;
    
    pub fn random<T>() -> T 
    where
        T: From<f64>,
    {
        unsafe {
            SEED = SEED.wrapping_add(1);
            let mut hasher = DefaultHasher::new();
            SEED.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Use current time as additional entropy
            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            
            let combined = hash.wrapping_add(time);
            let normalized = (combined as f64) / (u64::MAX as f64);
            
            T::from(normalized)
        }
    }
}

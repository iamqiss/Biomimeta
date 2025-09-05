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

//! Complete Afiyah Codec Demo
//! 
//! This example demonstrates the complete Afiyah biomimetic video compression
//! and streaming engine with all core components integrated. It showcases:
//! - Biological entropy coding with neural prediction
//! - Biomimetic transform coding with cortical frequency analysis
//! - Advanced motion estimation with saccadic prediction
//! - Perceptual quantization with biological contrast sensitivity
//! - Biological bitstream formatting with adaptive bit allocation
//! - Complete encoder/decoder pipeline with biological accuracy validation

use afiyah::{
    CompressionEngine, EngineConfig, VisualInput, InputMetadata,
    BiologicalEntropyCoder, EntropyCodingConfig, Symbol,
    BiologicalTransformCoder, TransformCodingConfig, TransformType,
    BiologicalMotionEstimator, MotionEstimationConfig,
    BiologicalQuantizer, QuantizationConfig, QuantizerType,
    BiologicalBitstreamFormatter, BitstreamConfig,
    AfiyahError,
};
use ndarray::Array2;
use std::time::Instant;

fn main() -> Result<(), AfiyahError> {
    println!("🧠👁️ Afiyah Complete Codec Demo - Revolutionary Biomimetic Video Compression");
    println!("================================================================================\n");

    // Initialize the complete compression engine
    println!("🔧 Initializing Afiyah Compression Engine...");
    let config = EngineConfig::default();
    let mut engine = CompressionEngine::new(config)?;
    println!("✅ Compression engine initialized successfully\n");

    // Create sample video data
    println!("📹 Creating sample video data...");
    let video_data = create_sample_video_data();
    println!("✅ Sample video data created ({}x{} pixels, {} frames)", 
             video_data.spatial_resolution.0, 
             video_data.spatial_resolution.1, 
             video_data.temporal_resolution);
    println!("   - Luminance data: {} samples", video_data.luminance_data.len());
    println!("   - Chrominance data: {} samples", video_data.chrominance_data.len());
    println!("   - Viewing distance: {} meters", video_data.metadata.viewing_distance);
    println!("   - Ambient lighting: {} lux", video_data.metadata.ambient_lighting);
    println!("   - Viewer age: {} years", video_data.metadata.viewer_age);
    println!("   - Color temperature: {} K\n", video_data.metadata.color_temperature);

    // Demonstrate individual core components
    demonstrate_entropy_coding()?;
    demonstrate_transform_coding()?;
    demonstrate_motion_estimation()?;
    demonstrate_quantization()?;
    demonstrate_bitstream_formatting()?;

    // Run complete compression pipeline
    println!("🚀 Running Complete Afiyah Compression Pipeline");
    println!("================================================\n");

    let start_time = Instant::now();

    // Step 1: Calibrate biological components
    println!("🧬 Step 1: Calibrating biological components...");
    engine.calibrate_photoreceptors(&video_data)?;
    println!("   ✅ Photoreceptors calibrated for biological accuracy");

    // Step 2: Train cortical filters
    println!("🧠 Step 2: Training cortical filters...");
    let training_data = vec![video_data.clone()];
    engine.train_cortical_filters(&training_data)?;
    println!("   ✅ Cortical filters trained with biological parameters");

    // Step 3: Configure biological parameters
    println!("⚙️  Step 3: Configuring biological parameters...");
    engine = engine
        .with_saccadic_prediction(true)
        .with_foveal_attention(true)
        .with_temporal_integration(200);
    println!("   ✅ Saccadic prediction: Enabled");
    println!("   ✅ Foveal attention: Enabled");
    println!("   ✅ Temporal integration: 200ms");

    // Step 4: Compress video data
    println!("🗜️  Step 4: Compressing video data...");
    let compression_start = Instant::now();
    let compression_result = engine.compress(&video_data)?;
    let compression_time = compression_start.elapsed();
    
    println!("   ✅ Compression completed in {:.2}ms", compression_time.as_secs_f64() * 1000.0);
    println!("   📊 Compression Results:");
    println!("      - Compressed size: {} bytes", compression_result.compressed_data.len());
    println!("      - Biological accuracy: {:.1}%", compression_result.biological_accuracy * 100.0);
    println!("      - Compression ratio: {:.1}%", compression_result.compression_ratio * 100.0);
    println!("      - Processing time: {:.2}ms", compression_result.processing_time);

    // Step 5: Decompress video data
    println!("🔄 Step 5: Decompressing video data...");
    let decompression_start = Instant::now();
    let decompressed_data = engine.decompress(&compression_result.compressed_data)?;
    let decompression_time = decompression_start.elapsed();
    
    println!("   ✅ Decompression completed in {:.2}ms", decompression_time.as_secs_f64() * 1000.0);
    println!("   📊 Decompressed data:");
    println!("      - Luminance data: {} samples", decompressed_data.luminance_data.len());
    println!("      - Spatial resolution: {}x{}", decompressed_data.spatial_resolution.0, decompressed_data.spatial_resolution.1);
    println!("      - Temporal resolution: {} fps", decompressed_data.temporal_resolution);

    // Step 6: Validate biological accuracy
    println!("🔬 Step 6: Validating biological accuracy...");
    let accuracy = validate_biological_accuracy(&video_data, &decompressed_data)?;
    println!("   ✅ Biological accuracy validation: {:.1}%", accuracy * 100.0);

    let total_time = start_time.elapsed();
    println!("\n🎉 Complete Afiyah Pipeline Completed Successfully!");
    println!("==================================================");
    println!("⏱️  Total processing time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    println!("🧬 Biological accuracy: {:.1}%", accuracy * 100.0);
    println!("🗜️  Compression ratio: {:.1}%", compression_result.compression_ratio * 100.0);
    println!("🚀 Performance: {:.1}x real-time", 1000.0 / total_time.as_secs_f64() / 30.0);

    Ok(())
}

/// Demonstrate biological entropy coding
fn demonstrate_entropy_coding() -> Result<(), AfiyahError> {
    println!("🧮 Demonstrating Biological Entropy Coding");
    println!("===========================================");
    
    let config = EntropyCodingConfig::default();
    let mut entropy_coder = BiologicalEntropyCoder::new(config)?;
    
    // Create sample symbols
    let symbols = vec![
        Symbol::Luminance(0.5),
        Symbol::Chrominance(0.3),
        Symbol::MotionVector(1.0, 2.0),
        Symbol::TransformCoeff(0.8),
        Symbol::PredictionResidual(0.1),
        Symbol::BiologicalFeature(0.9),
    ];
    
    println!("   📝 Input symbols: {} symbols", symbols.len());
    
    // Encode symbols
    let encoded = entropy_coder.encode(&symbols)?;
    println!("   🔒 Encoded data: {} bytes", encoded.len());
    
    // Decode symbols
    let decoded = entropy_coder.decode(&encoded)?;
    println!("   🔓 Decoded symbols: {} symbols", decoded.len());
    
    println!("   ✅ Entropy coding demonstration completed\n");
    Ok(())
}

/// Demonstrate biomimetic transform coding
fn demonstrate_transform_coding() -> Result<(), AfiyahError> {
    println!("🔄 Demonstrating Biomimetic Transform Coding");
    println!("=============================================");
    
    let config = TransformCodingConfig::default();
    let mut transform_coder = BiologicalTransformCoder::new(config)?;
    
    // Create sample image data
    let image_data = Array2::from_shape_fn((64, 64), |(i, j)| {
        ((i as f64 * j as f64) / 4096.0).sin()
    });
    
    println!("   📊 Input image: {}x{} pixels", image_data.nrows(), image_data.ncols());
    
    // Apply transform
    let transform_output = transform_coder.transform(&image_data)?;
    println!("   🔄 Transform applied: {:?}", transform_output.transform_type);
    println!("   📈 Biological accuracy: {:.1}%", transform_output.biological_accuracy * 100.0);
    println!("   🗜️  Compression potential: {:.1}%", transform_output.compression_potential * 100.0);
    
    // Apply inverse transform
    let inverse_transform = transform_coder.inverse_transform(&transform_output)?;
    println!("   🔄 Inverse transform applied: {}x{} pixels", inverse_transform.nrows(), inverse_transform.ncols());
    
    println!("   ✅ Transform coding demonstration completed\n");
    Ok(())
}

/// Demonstrate motion estimation
fn demonstrate_motion_estimation() -> Result<(), AfiyahError> {
    println!("🏃 Demonstrating Motion Estimation");
    println!("==================================");
    
    let config = MotionEstimationConfig::default();
    let mut motion_estimator = BiologicalMotionEstimator::new(config)?;
    
    // Create sample frames
    let frame1 = Array2::from_shape_fn((64, 64), |(i, j)| {
        ((i as f64 + j as f64) / 128.0).sin()
    });
    let frame2 = Array2::from_shape_fn((64, 64), |(i, j)| {
        (((i + 1) as f64 + j as f64) / 128.0).sin()
    });
    
    println!("   📹 Frame 1: {}x{} pixels", frame1.nrows(), frame1.ncols());
    println!("   📹 Frame 2: {}x{} pixels", frame2.nrows(), frame2.ncols());
    
    // Estimate motion
    let motion_result = motion_estimator.estimate_motion(&frame1, &frame2)?;
    println!("   🏃 Motion vectors: {} vectors", motion_result.motion_vectors.len());
    println!("   🧠 Biological accuracy: {:.1}%", motion_result.biological_accuracy * 100.0);
    println!("   🗜️  Compression ratio: {:.1}%", motion_result.compression_ratio * 100.0);
    
    // Compensate motion
    let compensated = motion_estimator.compensate_motion(&frame2, &motion_result.motion_vectors)?;
    println!("   🔄 Motion compensated: {}x{} pixels", compensated.nrows(), compensated.ncols());
    
    println!("   ✅ Motion estimation demonstration completed\n");
    Ok(())
}

/// Demonstrate quantization
fn demonstrate_quantization() -> Result<(), AfiyahError> {
    println!("📏 Demonstrating Perceptual Quantization");
    println!("=========================================");
    
    let config = QuantizationConfig::default();
    let mut quantizer = BiologicalQuantizer::new(config)?;
    
    // Create sample data
    let data = Array2::from_shape_fn((32, 32), |(i, j)| {
        ((i as f64 * j as f64) / 1024.0).cos()
    });
    
    println!("   📊 Input data: {}x{} pixels", data.nrows(), data.ncols());
    
    // Quantize data
    let quantization_result = quantizer.quantize(&data, None)?;
    println!("   📏 Quantization strategy: {:?}", quantization_result.quantization_strategy);
    println!("   🧬 Biological accuracy: {:.1}%", quantization_result.biological_accuracy * 100.0);
    println!("   🗜️  Compression ratio: {:.1}%", quantization_result.compression_ratio * 100.0);
    println!("   📊 Quantization error: {:.4}", quantization_result.quantization_error);
    
    // Dequantize data
    let dequantized = quantizer.dequantize(&quantization_result.quantized_data, quantization_result.quantization_strategy)?;
    println!("   🔄 Dequantized data: {}x{} pixels", dequantized.nrows(), dequantized.ncols());
    
    println!("   ✅ Quantization demonstration completed\n");
    Ok(())
}

/// Demonstrate bitstream formatting
fn demonstrate_bitstream_formatting() -> Result<(), AfiyahError> {
    println!("📦 Demonstrating Biological Bitstream Formatting");
    println!("================================================");
    
    let config = BitstreamConfig::default();
    let mut bitstream_formatter = BiologicalBitstreamFormatter::new(config)?;
    
    // Create sample compression data
    let compression_data = afiyah::CompressionData::new();
    
    println!("   📊 Input data: {} sections", compression_data.sections.len());
    
    // Format bitstream
    let bitstream_output = bitstream_formatter.format_bitstream(&compression_data)?;
    println!("   📦 Bitstream created: {} bytes", bitstream_output.bitstream.len());
    println!("   🧬 Biological accuracy: {:.1}%", bitstream_output.biological_accuracy * 100.0);
    println!("   🗜️  Compression ratio: {:.1}%", bitstream_output.compression_ratio * 100.0);
    println!("   🔒 Error resilience: {:.1}%", bitstream_output.error_resilience_info.detection_capability * 100.0);
    println!("   🚀 Streaming latency: {:.1}ms", bitstream_output.streaming_info.latency);
    
    // Parse bitstream
    let parsed_data = bitstream_formatter.parse_bitstream(&bitstream_output.bitstream)?;
    println!("   🔓 Bitstream parsed: {} bytes", parsed_data.data.len());
    
    println!("   ✅ Bitstream formatting demonstration completed\n");
    Ok(())
}

/// Create sample video data
fn create_sample_video_data() -> VisualInput {
    let width = 64;
    let height = 64;
    let size = width * height;
    
    // Create sample luminance data with biological patterns
    let luminance_data = (0..size)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            
            // Create biological-like patterns
            let pattern1 = (x * 10.0).sin() * (y * 10.0).cos();
            let pattern2 = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt().sin();
            let pattern3 = (x * y * 20.0).sin();
            
            (pattern1 + pattern2 + pattern3) / 3.0
        })
        .collect();
    
    // Create sample chrominance data
    let chrominance_data = (0..size)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            
            // Create color patterns
            let red = (x * 8.0).sin();
            let blue = (y * 8.0).cos();
            
            vec![red, blue]
        })
        .flatten()
        .collect();
    
    VisualInput {
        luminance_data,
        chrominance_data,
        spatial_resolution: (width, height),
        temporal_resolution: 30.0,
        metadata: InputMetadata {
            viewing_distance: 1.0, // 1 meter
            ambient_lighting: 100.0, // 100 lux
            viewer_age: 30, // 30 years old
            color_temperature: 6500.0, // 6500K
        },
    }
}

/// Validate biological accuracy
fn validate_biological_accuracy(original: &VisualInput, decompressed: &VisualInput) -> Result<f64, AfiyahError> {
    // Calculate similarity between original and decompressed data
    let mut total_error = 0.0;
    let mut count = 0;
    
    for (orig, decomp) in original.luminance_data.iter().zip(decompressed.luminance_data.iter()) {
        let error = (orig - decomp).abs();
        total_error += error * error;
        count += 1;
    }
    
    let mse = total_error / count as f64;
    let psnr = 20.0 * (1.0 / mse).log10();
    let accuracy = (psnr / 100.0).min(1.0).max(0.0);
    
    Ok(accuracy)
}
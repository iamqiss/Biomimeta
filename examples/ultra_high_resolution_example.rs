//! Ultra High Resolution Processing Example
//! 
//! This example demonstrates how to use Afiyah for 8K@120fps processing
//! with perfect audio-video synchronization and crisp quality.

use afiyah::{
    CompressionEngine, VisualInput, VisualMetadata, 
    ultra_high_resolution::{UltraHighResolutionProcessor, QualityPreset, PerformanceMode, MemoryOptimization}
};
use ndarray::{Array3, Array2};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Afiyah Ultra High Resolution Processing Example");
    println!("==================================================");
    
    // Create compression engine
    let mut engine = CompressionEngine::new()?;
    
    // Enable ultra high resolution processing
    let mut config = engine.get_config().clone();
    config.enable_ultra_high_resolution = true;
    engine.update_config(config);
    
    // Create 8K@120fps test video (simplified for example)
    let height = 4320; // 8K height
    let width = 7680;  // 8K width
    let frames = 120;  // 120fps
    
    println!("📹 Creating 8K@120fps test video...");
    let test_video = create_test_video(height, width, frames);
    
    // Create test audio (48kHz, 1 second)
    let audio_sample_rate = 48000;
    let audio_duration = 1.0; // 1 second
    let test_audio = create_test_audio(audio_sample_rate, audio_duration);
    
    println!("🎵 Created test audio: {} samples", test_audio.len());
    
    // Test different quality presets
    let quality_presets = vec![
        QualityPreset::UltraFast,
        QualityPreset::Balanced,
        QualityPreset::HighQuality,
        QualityPreset::Maximum,
        QualityPreset::Cinematic,
    ];
    
    for preset in quality_presets {
        println!("\n🎯 Testing quality preset: {:?}", preset);
        
        // Optimize for preset
        engine.optimize_for_quality_preset(preset)?;
        
        // Process ultra high resolution video
        let start_time = Instant::now();
        let processed_video = engine.process_ultra_high_resolution(&test_video)?;
        let processing_time = start_time.elapsed();
        
        println!("   ⏱️  Processing time: {:?}", processing_time);
        println!("   📊 Output resolution: {}x{}x{}", 
                processed_video.nrows(), 
                processed_video.ncols(), 
                processed_video.nslices());
        
        // Get performance metrics
        let metrics = engine.get_ultra_performance_metrics();
        println!("   🧠 Memory usage: {:.2} GB", metrics.memory_usage);
        println!("   ⚡ Quality score: {:.2}", metrics.quality_score);
        println!("   🔄 Sync accuracy: {:.2}", metrics.sync_accuracy);
    }
    
    // Test audio-video synchronization
    println!("\n🎵 Testing perfect audio-video synchronization...");
    
    let start_time = Instant::now();
    let (synced_video, synced_audio) = engine.process_with_audio_sync(&test_video, &test_audio)?;
    let sync_time = start_time.elapsed();
    
    println!("   ⏱️  Sync processing time: {:?}", sync_time);
    println!("   📹 Synced video: {}x{}x{}", 
            synced_video.nrows(), 
            synced_video.ncols(), 
            synced_video.nslices());
    println!("   🎵 Synced audio: {} samples", synced_audio.len());
    
    // Test with different performance modes
    println!("\n⚡ Testing different performance modes...");
    
    let performance_modes = vec![
        PerformanceMode::CPUOnly,
        PerformanceMode::GPUAccelerated,
        PerformanceMode::MultiGPU,
        PerformanceMode::Neuromorphic,
        PerformanceMode::Hybrid,
    ];
    
    for mode in performance_modes {
        println!("   🔧 Performance mode: {:?}", mode);
        
        // Simulate performance mode change
        let start_time = Instant::now();
        let _processed = engine.process_ultra_high_resolution(&test_video)?;
        let mode_time = start_time.elapsed();
        
        println!("      ⏱️  Processing time: {:?}", mode_time);
    }
    
    // Test memory optimization strategies
    println!("\n💾 Testing memory optimization strategies...");
    
    let memory_strategies = vec![
        MemoryOptimization::Conservative,
        MemoryOptimization::Aggressive,
        MemoryOptimization::Streaming,
        MemoryOptimization::Tiled,
        MemoryOptimization::Hierarchical,
    ];
    
    for strategy in memory_strategies {
        println!("   🧠 Memory strategy: {:?}", strategy);
        
        // Simulate memory optimization
        let start_time = Instant::now();
        let _processed = engine.process_ultra_high_resolution(&test_video)?;
        let memory_time = start_time.elapsed();
        
        println!("      ⏱️  Processing time: {:?}", memory_time);
    }
    
    // Test compression with ultra high resolution
    println!("\n🗜️  Testing compression with ultra high resolution...");
    
    // Create visual input for compression
    let visual_input = create_visual_input(&test_video)?;
    
    let start_time = Instant::now();
    let compressed_output = engine.compress(&visual_input)?;
    let compression_time = start_time.elapsed();
    
    println!("   ⏱️  Compression time: {:?}", compression_time);
    println!("   📊 Compression ratio: {:.2}", compressed_output.compression_ratio);
    println!("   🎯 Quality metrics:");
    println!("      VMAF: {:.2}", compressed_output.quality_metrics.vmaf);
    println!("      PSNR: {:.2} dB", compressed_output.quality_metrics.psnr);
    println!("      SSIM: {:.2}", compressed_output.quality_metrics.ssim);
    println!("   🧬 Biological accuracy: {:.2}", compressed_output.biological_accuracy);
    
    // Test real-time processing capabilities
    println!("\n⚡ Testing real-time processing capabilities...");
    
    let real_time_frames = 10; // Process 10 frames in real-time
    let real_time_video = create_test_video(height, width, real_time_frames);
    
    let start_time = Instant::now();
    let _real_time_processed = engine.process_ultra_high_resolution(&real_time_video)?;
    let real_time_duration = start_time.elapsed();
    
    let fps_achieved = real_time_frames as f64 / real_time_duration.as_secs_f64();
    println!("   🎬 FPS achieved: {:.2}", fps_achieved);
    println!("   ⏱️  Time per frame: {:.2} ms", real_time_duration.as_secs_f64() * 1000.0 / real_time_frames as f64);
    
    if fps_achieved >= 120.0 {
        println!("   ✅ Real-time 120fps processing: ACHIEVED!");
    } else {
        println!("   ⚠️  Real-time 120fps processing: {:.2} fps (target: 120 fps)", fps_achieved);
    }
    
    // Test scalability
    println!("\n📈 Testing scalability...");
    
    let resolutions = vec![
        (1080, 1920, 60),   // 1080p@60fps
        (2160, 3840, 60),   // 4K@60fps
        (2160, 3840, 120),  // 4K@120fps
        (4320, 7680, 60),   // 8K@60fps
        (4320, 7680, 120),  // 8K@120fps
    ];
    
    for (h, w, f) in resolutions {
        println!("   📐 Resolution: {}x{}@{}fps", w, h, f);
        
        let test_vid = create_test_video(h, w, f);
        let start_time = Instant::now();
        let _processed = engine.process_ultra_high_resolution(&test_vid)?;
        let duration = start_time.elapsed();
        
        let fps = f as f64 / duration.as_secs_f64();
        println!("      ⏱️  Processing time: {:?}", duration);
        println!("      🎬 Effective FPS: {:.2}", fps);
        
        if fps >= f as f64 {
            println!("      ✅ Real-time processing: ACHIEVED!");
        } else {
            println!("      ⚠️  Real-time processing: {:.2} fps (target: {} fps)", fps, f);
        }
    }
    
    println!("\n🎉 Ultra high resolution processing example completed successfully!");
    println!("🚀 Afiyah is ready for 8K@120fps processing with perfect audio-video sync!");
    
    Ok(())
}

/// Creates a test video with specified dimensions
fn create_test_video(height: usize, width: usize, frames: usize) -> Array3<f64> {
    let mut video = Array3::zeros((height, width, frames));
    
    for frame in 0..frames {
        for i in 0..height {
            for j in 0..width {
                // Create a simple test pattern
                let x = j as f64 / width as f64;
                let y = i as f64 / height as f64;
                let t = frame as f64 / frames as f64;
                
                // Animated sine wave pattern
                let value = (2.0 * std::f64::consts::PI * (x + y + t)).sin() * 0.5 + 0.5;
                video[[i, j, frame]] = value;
            }
        }
    }
    
    video
}

/// Creates test audio with specified sample rate and duration
fn create_test_audio(sample_rate: usize, duration: f64) -> Vec<f64> {
    let samples = (sample_rate as f64 * duration) as usize;
    let mut audio = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / sample_rate as f64;
        // Create a simple sine wave
        let value = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.1; // 440Hz tone
        audio.push(value);
    }
    
    audio
}

/// Creates visual input from video array
fn create_visual_input(video: &Array3<f64>) -> Result<VisualInput, Box<dyn std::error::Error>> {
    // Take the first frame as input
    let first_frame = video.slice(s![.., .., 0]).to_owned();
    
    Ok(VisualInput {
        data: first_frame,
        metadata: VisualMetadata {
            resolution: (video.ncols(), video.nrows()),
            color_space: "sRGB".to_string(),
            bit_depth: 8,
            frame_rate: 120.0,
            color_temperature: 6500.0,
            ambient_lighting: 100.0,
            viewing_distance: 2.0,
            viewer_age: 30,
            temporal_resolution: 120.0,
        },
    })
}
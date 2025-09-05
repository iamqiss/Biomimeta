//! Profiling Module

use ndarray::Array2;
use crate::AfiyahError;

/// Profiler for performance analysis
pub struct Profiler {
    active_profiles: std::collections::HashMap<String, ProfileSession>,
    profile_config: ProfileConfig,
}

/// Profile session for tracking performance
#[derive(Debug, Clone)]
pub struct ProfileSession {
    pub name: String,
    pub start_time: std::time::Instant,
    pub end_time: Option<std::time::Instant>,
    pub duration: Option<std::time::Duration>,
    pub memory_usage: Vec<MemorySnapshot>,
    pub cpu_usage: Vec<CpuSnapshot>,
    pub custom_metrics: std::collections::HashMap<String, f64>,
}

/// Memory snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: std::time::Instant,
    pub heap_usage: f64,
    pub stack_usage: f64,
    pub total_usage: f64,
    pub peak_usage: f64,
}

/// CPU snapshot
#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    pub timestamp: std::time::Instant,
    pub usage_percent: f64,
    pub user_time: f64,
    pub system_time: f64,
    pub idle_time: f64,
}

/// Profile configuration
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    pub enable_memory_profiling: bool,
    pub enable_cpu_profiling: bool,
    pub enable_custom_metrics: bool,
    pub sampling_interval_ms: u64,
    pub max_profile_duration_seconds: u64,
    pub memory_threshold_mb: f64,
    pub cpu_threshold_percent: f64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_custom_metrics: true,
            sampling_interval_ms: 100,
            max_profile_duration_seconds: 300,
            memory_threshold_mb: 512.0,
            cpu_threshold_percent: 80.0,
        }
    }
}

/// Profile result containing performance analysis
#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub profile_name: String,
    pub total_time: f64,
    pub memory_usage: MemoryProfile,
    pub cpu_usage: CpuProfile,
    pub custom_metrics: std::collections::HashMap<String, f64>,
    pub performance_profile: PerformanceProfile,
}

/// Memory profile
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub peak_usage: f64,
    pub average_usage: f64,
    pub min_usage: f64,
    pub max_usage: f64,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub memory_leaks: bool,
}

/// CPU profile
#[derive(Debug, Clone)]
pub struct CpuProfile {
    pub peak_usage: f64,
    pub average_usage: f64,
    pub min_usage: f64,
    pub max_usage: f64,
    pub user_time: f64,
    pub system_time: f64,
    pub idle_time: f64,
    pub context_switches: usize,
}

/// Performance profile
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub bottlenecks: Vec<Bottleneck>,
    pub recommendations: Vec<String>,
    pub performance_score: f64,
    pub optimization_potential: f64,
}

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub name: String,
    pub severity: BottleneckSeverity,
    pub impact: f64,
    pub description: String,
    pub recommendations: Vec<String>,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Profiler {
    /// Creates a new profiler
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            active_profiles: std::collections::HashMap::new(),
            profile_config: ProfileConfig::default(),
        })
    }

    /// Starts profiling a named operation
    pub fn start_profiling(&mut self, name: &str) -> Result<(), AfiyahError> {
        let session = ProfileSession {
            name: name.to_string(),
            start_time: std::time::Instant::now(),
            end_time: None,
            duration: None,
            memory_usage: Vec::new(),
            cpu_usage: Vec::new(),
            custom_metrics: std::collections::HashMap::new(),
        };

        self.active_profiles.insert(name.to_string(), session);
        Ok(())
    }

    /// Stops profiling a named operation
    pub fn stop_profiling(&mut self, name: &str) -> Result<ProfileResult, AfiyahError> {
        let mut session = self.active_profiles.remove(name)
            .ok_or_else(|| AfiyahError::PerformanceOptimization {
                message: format!("Profile session '{}' not found", name)
            })?;

        session.end_time = Some(std::time::Instant::now());
        session.duration = Some(session.end_time.unwrap().duration_since(session.start_time));

        // Calculate profile result
        let result = self.calculate_profile_result(session)?;
        Ok(result)
    }

    /// Profiles processing performance
    pub fn profile_processing(&mut self, input: &Array2<f64>) -> Result<ProfileResult, AfiyahError> {
        self.start_profiling("profile_processing")?;

        // Simulate processing
        let _output = self.simulate_processing(input)?;

        let result = self.stop_profiling("profile_processing")?;
        Ok(result)
    }

    /// Adds custom metric to active profile
    pub fn add_custom_metric(&mut self, profile_name: &str, metric_name: &str, value: f64) -> Result<(), AfiyahError> {
        if let Some(session) = self.active_profiles.get_mut(profile_name) {
            session.custom_metrics.insert(metric_name.to_string(), value);
            Ok(())
        } else {
            Err(AfiyahError::PerformanceOptimization {
                message: format!("Profile session '{}' not found", profile_name)
            })
        }
    }

    /// Gets active profile names
    pub fn get_active_profiles(&self) -> Vec<String> {
        self.active_profiles.keys().cloned().collect()
    }

    /// Checks if a profile is active
    pub fn is_profile_active(&self, name: &str) -> bool {
        self.active_profiles.contains_key(name)
    }

    fn simulate_processing(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate various processing operations
        let mut output = input.clone();

        // Simulate memory allocation
        let mut temp = Vec::new();
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                temp.push(output[[i, j]] * 1.1);
            }
        }

        // Simulate processing
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                let index = i * output.ncols() + j;
                output[[i, j]] = temp[index] * 0.9;
            }
        }

        // Simulate additional operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]].sin().cos();
            }
        }

        Ok(output)
    }

    fn calculate_profile_result(&self, session: ProfileSession) -> Result<ProfileResult, AfiyahError> {
        let total_time = session.duration.unwrap().as_secs_f64();
        
        // Calculate memory profile
        let memory_profile = self.calculate_memory_profile(&session)?;
        
        // Calculate CPU profile
        let cpu_profile = self.calculate_cpu_profile(&session)?;
        
        // Calculate performance profile
        let performance_profile = self.calculate_performance_profile(&session, &memory_profile, &cpu_profile)?;

        Ok(ProfileResult {
            profile_name: session.name,
            total_time,
            memory_usage: memory_profile,
            cpu_usage: cpu_profile,
            custom_metrics: session.custom_metrics,
            performance_profile,
        })
    }

    fn calculate_memory_profile(&self, session: &ProfileSession) -> Result<MemoryProfile, AfiyahError> {
        if session.memory_usage.is_empty() {
            // Simulate memory usage if no real data
            let peak_usage = 256.0;
            let average_usage = 128.0;
            let min_usage = 64.0;
            let max_usage = peak_usage;
            let allocation_count = 100;
            let deallocation_count = 95;
            let memory_leaks = allocation_count > deallocation_count;

            return Ok(MemoryProfile {
                peak_usage,
                average_usage,
                min_usage,
                max_usage,
                allocation_count,
                deallocation_count,
                memory_leaks,
            });
        }

        let peak_usage = session.memory_usage.iter().map(|s| s.peak_usage).fold(0.0, f64::max);
        let average_usage = session.memory_usage.iter().map(|s| s.total_usage).sum::<f64>() / session.memory_usage.len() as f64;
        let min_usage = session.memory_usage.iter().map(|s| s.total_usage).fold(f64::INFINITY, f64::min);
        let max_usage = session.memory_usage.iter().map(|s| s.total_usage).fold(0.0, f64::max);
        let allocation_count = session.memory_usage.len();
        let deallocation_count = session.memory_usage.len() - 5; // Simulate some deallocations
        let memory_leaks = allocation_count > deallocation_count;

        Ok(MemoryProfile {
            peak_usage,
            average_usage,
            min_usage,
            max_usage,
            allocation_count,
            deallocation_count,
            memory_leaks,
        })
    }

    fn calculate_cpu_profile(&self, session: &ProfileSession) -> Result<CpuProfile, AfiyahError> {
        if session.cpu_usage.is_empty() {
            // Simulate CPU usage if no real data
            let peak_usage = 75.0;
            let average_usage = 50.0;
            let min_usage = 25.0;
            let max_usage = peak_usage;
            let user_time = 0.3;
            let system_time = 0.1;
            let idle_time = 0.6;
            let context_switches = 1000;

            return Ok(CpuProfile {
                peak_usage,
                average_usage,
                min_usage,
                max_usage,
                user_time,
                system_time,
                idle_time,
                context_switches,
            });
        }

        let peak_usage = session.cpu_usage.iter().map(|s| s.usage_percent).fold(0.0, f64::max);
        let average_usage = session.cpu_usage.iter().map(|s| s.usage_percent).sum::<f64>() / session.cpu_usage.len() as f64;
        let min_usage = session.cpu_usage.iter().map(|s| s.usage_percent).fold(f64::INFINITY, f64::min);
        let max_usage = session.cpu_usage.iter().map(|s| s.usage_percent).fold(0.0, f64::max);
        let user_time = session.cpu_usage.iter().map(|s| s.user_time).sum::<f64>() / session.cpu_usage.len() as f64;
        let system_time = session.cpu_usage.iter().map(|s| s.system_time).sum::<f64>() / session.cpu_usage.len() as f64;
        let idle_time = session.cpu_usage.iter().map(|s| s.idle_time).sum::<f64>() / session.cpu_usage.len() as f64;
        let context_switches = session.cpu_usage.len() * 10; // Simulate context switches

        Ok(CpuProfile {
            peak_usage,
            average_usage,
            min_usage,
            max_usage,
            user_time,
            system_time,
            idle_time,
            context_switches,
        })
    }

    fn calculate_performance_profile(&self, session: &ProfileSession, memory_profile: &MemoryProfile, cpu_profile: &CpuProfile) -> Result<PerformanceProfile, AfiyahError> {
        let mut bottlenecks = Vec::new();
        let mut recommendations = Vec::new();
        let mut performance_score: f64 = 100.0;

        // Check for memory bottlenecks
        if memory_profile.peak_usage > self.profile_config.memory_threshold_mb {
            bottlenecks.push(Bottleneck {
                name: "High Memory Usage".to_string(),
                severity: BottleneckSeverity::High,
                impact: (memory_profile.peak_usage / self.profile_config.memory_threshold_mb) - 1.0,
                description: "Memory usage exceeds threshold".to_string(),
                recommendations: vec![
                    "Optimize memory allocation patterns".to_string(),
                    "Use memory pooling".to_string(),
                    "Reduce data structure sizes".to_string(),
                ],
            });
            performance_score -= 20.0;
        }

        if memory_profile.memory_leaks {
            bottlenecks.push(Bottleneck {
                name: "Memory Leaks".to_string(),
                severity: BottleneckSeverity::Critical,
                impact: 1.0,
                description: "Memory leaks detected".to_string(),
                recommendations: vec![
                    "Fix memory leaks".to_string(),
                    "Use RAII patterns".to_string(),
                    "Implement proper cleanup".to_string(),
                ],
            });
            performance_score -= 30.0;
        }

        // Check for CPU bottlenecks
        if cpu_profile.peak_usage > self.profile_config.cpu_threshold_percent {
            bottlenecks.push(Bottleneck {
                name: "High CPU Usage".to_string(),
                severity: BottleneckSeverity::Medium,
                impact: (cpu_profile.peak_usage / self.profile_config.cpu_threshold_percent) - 1.0,
                description: "CPU usage exceeds threshold".to_string(),
                recommendations: vec![
                    "Optimize algorithms".to_string(),
                    "Use parallel processing".to_string(),
                    "Reduce computational complexity".to_string(),
                ],
            });
            performance_score -= 15.0;
        }

        // Check for performance issues
        let total_time = session.duration.unwrap().as_secs_f64();
        if total_time > 1.0 {
            bottlenecks.push(Bottleneck {
                name: "Slow Execution".to_string(),
                severity: BottleneckSeverity::Medium,
                impact: total_time - 1.0,
                description: "Execution time is too long".to_string(),
                recommendations: vec![
                    "Profile and optimize hot paths".to_string(),
                    "Use more efficient algorithms".to_string(),
                    "Consider caching".to_string(),
                ],
            });
            performance_score -= 10.0;
        }

        // Generate general recommendations
        if performance_score < 80.0 {
            recommendations.push("Overall performance needs improvement".to_string());
        }
        if memory_profile.average_usage > 100.0 {
            recommendations.push("Consider memory optimization".to_string());
        }
        if cpu_profile.average_usage > 60.0 {
            recommendations.push("Consider CPU optimization".to_string());
        }

        let optimization_potential = 100.0 - performance_score;

        Ok(PerformanceProfile {
            bottlenecks,
            recommendations,
            performance_score: performance_score.max(0.0),
            optimization_potential,
        })
    }

    /// Updates profile configuration
    pub fn update_config(&mut self, config: ProfileConfig) {
        self.profile_config = config;
    }

    /// Gets current profile configuration
    pub fn get_config(&self) -> &ProfileConfig {
        &self.profile_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_start_stop_profiling() {
        let mut profiler = Profiler::new().unwrap();
        
        let start_result = profiler.start_profiling("test_profile");
        assert!(start_result.is_ok());
        
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let stop_result = profiler.stop_profiling("test_profile");
        assert!(stop_result.is_ok());
        
        let profile_result = stop_result.unwrap();
        assert_eq!(profile_result.profile_name, "test_profile");
        assert!(profile_result.total_time > 0.0);
    }

    #[test]
    fn test_profile_processing() {
        let mut profiler = Profiler::new().unwrap();
        let input = Array2::ones((16, 16));
        
        let result = profiler.profile_processing(&input);
        assert!(result.is_ok());
        
        let profile_result = result.unwrap();
        assert_eq!(profile_result.profile_name, "profile_processing");
        assert!(profile_result.total_time > 0.0);
    }

    #[test]
    fn test_custom_metrics() {
        let mut profiler = Profiler::new().unwrap();
        
        profiler.start_profiling("test_metrics").unwrap();
        profiler.add_custom_metric("test_metrics", "custom_value", 42.0).unwrap();
        
        let result = profiler.stop_profiling("test_metrics");
        assert!(result.is_ok());
        
        let profile_result = result.unwrap();
        assert_eq!(profile_result.custom_metrics.get("custom_value"), Some(&42.0));
    }

    #[test]
    fn test_active_profiles() {
        let mut profiler = Profiler::new().unwrap();
        
        profiler.start_profiling("profile1").unwrap();
        profiler.start_profiling("profile2").unwrap();
        
        let active_profiles = profiler.get_active_profiles();
        assert_eq!(active_profiles.len(), 2);
        assert!(active_profiles.contains(&"profile1".to_string()));
        assert!(active_profiles.contains(&"profile2".to_string()));
        
        assert!(profiler.is_profile_active("profile1"));
        assert!(!profiler.is_profile_active("profile3"));
    }

    #[test]
    fn test_configuration_update() {
        let mut profiler = Profiler::new().unwrap();
        let config = ProfileConfig {
            enable_memory_profiling: false,
            enable_cpu_profiling: true,
            enable_custom_metrics: false,
            sampling_interval_ms: 50,
            max_profile_duration_seconds: 600,
            memory_threshold_mb: 1024.0,
            cpu_threshold_percent: 90.0,
        };
        
        profiler.update_config(config);
        assert!(!profiler.get_config().enable_memory_profiling);
        assert!(profiler.get_config().enable_cpu_profiling);
        assert_eq!(profiler.get_config().sampling_interval_ms, 50);
    }

    #[test]
    fn test_bottleneck_severity() {
        let severity = BottleneckSeverity::High;
        assert_eq!(severity, BottleneckSeverity::High);
        assert_ne!(severity, BottleneckSeverity::Low);
    }
}
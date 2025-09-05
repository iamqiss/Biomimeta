//! Benchmarking Module

use ndarray::Array2;
use crate::AfiyahError;

/// Benchmark suite for comprehensive performance testing
pub struct BenchmarkSuite {
    benchmarks: Vec<Benchmark>,
    benchmark_config: BenchmarkConfig,
}

/// Individual benchmark
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub name: String,
    pub description: String,
    pub benchmark_type: BenchmarkType,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub timeout_seconds: f64,
}

/// Types of benchmarks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchmarkType {
    Processing,
    Memory,
    Latency,
    Throughput,
    Scalability,
    Stress,
    Regression,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub enable_processing_benchmarks: bool,
    pub enable_memory_benchmarks: bool,
    pub enable_latency_benchmarks: bool,
    pub enable_throughput_benchmarks: bool,
    pub enable_scalability_benchmarks: bool,
    pub enable_stress_benchmarks: bool,
    pub enable_regression_benchmarks: bool,
    pub default_iterations: usize,
    pub default_warmup_iterations: usize,
    pub default_timeout_seconds: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_processing_benchmarks: true,
            enable_memory_benchmarks: true,
            enable_latency_benchmarks: true,
            enable_throughput_benchmarks: true,
            enable_scalability_benchmarks: true,
            enable_stress_benchmarks: false,
            enable_regression_benchmarks: true,
            default_iterations: 100,
            default_warmup_iterations: 10,
            default_timeout_seconds: 60.0,
        }
    }
}

/// Benchmark result containing performance metrics
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub total_time: f64,
    pub average_time: f64,
    pub min_time: f64,
    pub max_time: f64,
    pub median_time: f64,
    pub std_deviation: f64,
    pub iterations: usize,
    pub success_rate: f64,
    pub metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub latency: f64,
    pub throughput: f64,
    pub cache_hit_rate: f64,
    pub branch_prediction_accuracy: f64,
    pub instruction_retirement_rate: f64,
    pub power_consumption: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            latency: 0.0,
            throughput: 0.0,
            cache_hit_rate: 0.0,
            branch_prediction_accuracy: 0.0,
            instruction_retirement_rate: 0.0,
            power_consumption: 0.0,
        }
    }
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BenchmarkSuite {
    /// Creates a new benchmark suite
    pub fn new() -> Result<Self, AfiyahError> {
        let benchmarks = Self::initialize_benchmarks()?;
        let benchmark_config = BenchmarkConfig::default();

        Ok(Self {
            benchmarks,
            benchmark_config,
        })
    }

    /// Runs comprehensive benchmarks
    pub fn run_comprehensive_benchmarks(&mut self, input: &Array2<f64>) -> Result<BenchmarkResult, AfiyahError> {
        let mut results = Vec::new();
        let mut total_time = 0.0;
        let mut total_iterations = 0;

        for benchmark in &self.benchmarks {
            if self.is_benchmark_enabled(benchmark.benchmark_type) {
                let result = self.run_single_benchmark(benchmark, input)?;
                results.push(result.clone());
                total_time += result.total_time;
                total_iterations += result.iterations;
            }
        }

        // Calculate aggregate metrics
        let average_time = if total_iterations > 0 { total_time / total_iterations as f64 } else { 0.0 };
        let min_time = results.iter().map(|r| r.min_time).fold(f64::INFINITY, f64::min);
        let max_time = results.iter().map(|r| r.max_time).fold(0.0, f64::max);
        let median_time = self.calculate_median(&results.iter().map(|r| r.median_time).collect::<Vec<_>>());
        let std_deviation = self.calculate_std_deviation(&results.iter().map(|r| r.average_time).collect::<Vec<_>>());
        let success_rate = results.iter().map(|r| r.success_rate).sum::<f64>() / results.len() as f64;

        // Calculate aggregate performance metrics
        let metrics = self.calculate_aggregate_metrics(&results)?;

        Ok(BenchmarkResult {
            benchmark_name: "Comprehensive".to_string(),
            total_time,
            average_time,
            min_time,
            max_time,
            median_time,
            std_deviation,
            iterations: total_iterations,
            success_rate,
            metrics,
        })
    }

    /// Runs a single benchmark
    pub fn run_single_benchmark(&self, benchmark: &Benchmark, input: &Array2<f64>) -> Result<BenchmarkResult, AfiyahError> {
        let mut times = Vec::new();
        let mut metrics = PerformanceMetrics::new();
        let mut successful_iterations = 0;

        // Warmup iterations
        for _ in 0..benchmark.warmup_iterations {
            let _ = self.execute_benchmark(benchmark, input)?;
        }

        // Main benchmark iterations
        for i in 0..benchmark.iterations {
            let start_time = std::time::Instant::now();
            let result = self.execute_benchmark(benchmark, input)?;
            let elapsed = start_time.elapsed().as_secs_f64();

            if result.is_ok() {
                times.push(elapsed);
                successful_iterations += 1;
            }

            // Update metrics
            self.update_metrics(&mut metrics, &result?);
        }

        if times.is_empty() {
            return Err(AfiyahError::PerformanceOptimization {
                message: "No successful benchmark iterations".to_string()
            });
        }

        // Calculate statistics
        let total_time = times.iter().sum::<f64>();
        let average_time = total_time / times.len() as f64;
        let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = times.iter().fold(0.0, |a, &b| a.max(b));
        let median_time = self.calculate_median(&times);
        let std_deviation = self.calculate_std_deviation(&times);
        let success_rate = successful_iterations as f64 / benchmark.iterations as f64;

        Ok(BenchmarkResult {
            benchmark_name: benchmark.name.clone(),
            total_time,
            average_time,
            min_time,
            max_time,
            median_time,
            std_deviation,
            iterations: successful_iterations,
            success_rate,
            metrics,
        })
    }

    fn initialize_benchmarks() -> Result<Vec<Benchmark>, AfiyahError> {
        let mut benchmarks = Vec::new();

        // Processing benchmarks
        benchmarks.push(Benchmark {
            name: "Array Processing".to_string(),
            description: "Benchmark array processing operations".to_string(),
            benchmark_type: BenchmarkType::Processing,
            iterations: 100,
            warmup_iterations: 10,
            timeout_seconds: 30.0,
        });

        benchmarks.push(Benchmark {
            name: "Mathematical Operations".to_string(),
            description: "Benchmark mathematical operations".to_string(),
            benchmark_type: BenchmarkType::Processing,
            iterations: 100,
            warmup_iterations: 10,
            timeout_seconds: 30.0,
        });

        // Memory benchmarks
        benchmarks.push(Benchmark {
            name: "Memory Allocation".to_string(),
            description: "Benchmark memory allocation patterns".to_string(),
            benchmark_type: BenchmarkType::Memory,
            iterations: 50,
            warmup_iterations: 5,
            timeout_seconds: 20.0,
        });

        benchmarks.push(Benchmark {
            name: "Memory Access".to_string(),
            description: "Benchmark memory access patterns".to_string(),
            benchmark_type: BenchmarkType::Memory,
            iterations: 100,
            warmup_iterations: 10,
            timeout_seconds: 30.0,
        });

        // Latency benchmarks
        benchmarks.push(Benchmark {
            name: "Single Operation Latency".to_string(),
            description: "Benchmark single operation latency".to_string(),
            benchmark_type: BenchmarkType::Latency,
            iterations: 1000,
            warmup_iterations: 100,
            timeout_seconds: 10.0,
        });

        // Throughput benchmarks
        benchmarks.push(Benchmark {
            name: "Batch Processing".to_string(),
            description: "Benchmark batch processing throughput".to_string(),
            benchmark_type: BenchmarkType::Throughput,
            iterations: 20,
            warmup_iterations: 2,
            timeout_seconds: 60.0,
        });

        // Scalability benchmarks
        benchmarks.push(Benchmark {
            name: "Scalability Test".to_string(),
            description: "Benchmark scalability with different input sizes".to_string(),
            benchmark_type: BenchmarkType::Scalability,
            iterations: 10,
            warmup_iterations: 1,
            timeout_seconds: 120.0,
        });

        // Stress benchmarks
        benchmarks.push(Benchmark {
            name: "Stress Test".to_string(),
            description: "Benchmark under stress conditions".to_string(),
            benchmark_type: BenchmarkType::Stress,
            iterations: 5,
            warmup_iterations: 1,
            timeout_seconds: 300.0,
        });

        // Regression benchmarks
        benchmarks.push(Benchmark {
            name: "Regression Test".to_string(),
            description: "Benchmark for regression testing".to_string(),
            benchmark_type: BenchmarkType::Regression,
            iterations: 50,
            warmup_iterations: 5,
            timeout_seconds: 30.0,
        });

        Ok(benchmarks)
    }

    fn is_benchmark_enabled(&self, benchmark_type: BenchmarkType) -> bool {
        match benchmark_type {
            BenchmarkType::Processing => self.benchmark_config.enable_processing_benchmarks,
            BenchmarkType::Memory => self.benchmark_config.enable_memory_benchmarks,
            BenchmarkType::Latency => self.benchmark_config.enable_latency_benchmarks,
            BenchmarkType::Throughput => self.benchmark_config.enable_throughput_benchmarks,
            BenchmarkType::Scalability => self.benchmark_config.enable_scalability_benchmarks,
            BenchmarkType::Stress => self.benchmark_config.enable_stress_benchmarks,
            BenchmarkType::Regression => self.benchmark_config.enable_regression_benchmarks,
        }
    }

    fn execute_benchmark(&self, benchmark: &Benchmark, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        match benchmark.benchmark_type {
            BenchmarkType::Processing => self.execute_processing_benchmark(input),
            BenchmarkType::Memory => self.execute_memory_benchmark(input),
            BenchmarkType::Latency => self.execute_latency_benchmark(input),
            BenchmarkType::Throughput => self.execute_throughput_benchmark(input),
            BenchmarkType::Scalability => self.execute_scalability_benchmark(input),
            BenchmarkType::Stress => self.execute_stress_benchmark(input),
            BenchmarkType::Regression => self.execute_regression_benchmark(input),
        }
    }

    fn execute_processing_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate processing benchmark
        let mut output = input.clone();
        
        // Apply various processing operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.1 + 0.01;
            }
        }

        Ok(output)
    }

    fn execute_memory_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate memory benchmark
        let mut output = input.clone();
        
        // Simulate memory-intensive operations
        let mut temp = Vec::new();
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                temp.push(output[[i, j]] * 1.05);
            }
        }
        
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                let index = i * output.ncols() + j;
                output[[i, j]] = temp[index];
            }
        }

        Ok(output)
    }

    fn execute_latency_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate latency benchmark
        let mut output = input.clone();
        
        // Simulate low-latency operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.02;
            }
        }

        Ok(output)
    }

    fn execute_throughput_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate throughput benchmark
        let mut output = input.clone();
        
        // Simulate high-throughput operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.03;
            }
        }

        Ok(output)
    }

    fn execute_scalability_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate scalability benchmark
        let mut output = input.clone();
        
        // Simulate scalable operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.04;
            }
        }

        Ok(output)
    }

    fn execute_stress_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate stress benchmark
        let mut output = input.clone();
        
        // Simulate stress operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.05;
            }
        }

        Ok(output)
    }

    fn execute_regression_benchmark(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Simulate regression benchmark
        let mut output = input.clone();
        
        // Simulate regression operations
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] * 1.01;
            }
        }

        Ok(output)
    }

    fn update_metrics(&self, metrics: &mut PerformanceMetrics, _result: &Array2<f64>) {
        // Simulate metrics update
        metrics.cpu_usage = 0.6;
        metrics.memory_usage = 256.0;
        metrics.latency = 10.0;
        metrics.throughput = 1000.0;
        metrics.cache_hit_rate = 0.95;
        metrics.branch_prediction_accuracy = 0.90;
        metrics.instruction_retirement_rate = 0.85;
        metrics.power_consumption = 50.0;
    }

    fn calculate_median(&self, values: &[f64]) -> f64 {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted_values.len();
        if len % 2 == 0 {
            (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
        } else {
            sorted_values[len / 2]
        }
    }

    fn calculate_std_deviation(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn calculate_aggregate_metrics(&self, results: &[BenchmarkResult]) -> Result<PerformanceMetrics, AfiyahError> {
        let mut metrics = PerformanceMetrics::new();

        if results.is_empty() {
            return Ok(metrics);
        }

        // Calculate aggregate metrics
        metrics.cpu_usage = results.iter().map(|r| r.metrics.cpu_usage).sum::<f64>() / results.len() as f64;
        metrics.memory_usage = results.iter().map(|r| r.metrics.memory_usage).sum::<f64>() / results.len() as f64;
        metrics.latency = results.iter().map(|r| r.metrics.latency).sum::<f64>() / results.len() as f64;
        metrics.throughput = results.iter().map(|r| r.metrics.throughput).sum::<f64>() / results.len() as f64;
        metrics.cache_hit_rate = results.iter().map(|r| r.metrics.cache_hit_rate).sum::<f64>() / results.len() as f64;
        metrics.branch_prediction_accuracy = results.iter().map(|r| r.metrics.branch_prediction_accuracy).sum::<f64>() / results.len() as f64;
        metrics.instruction_retirement_rate = results.iter().map(|r| r.metrics.instruction_retirement_rate).sum::<f64>() / results.len() as f64;
        metrics.power_consumption = results.iter().map(|r| r.metrics.power_consumption).sum::<f64>() / results.len() as f64;

        Ok(metrics)
    }

    /// Updates benchmark configuration
    pub fn update_config(&mut self, config: BenchmarkConfig) {
        self.benchmark_config = config;
    }

    /// Gets current benchmark configuration
    pub fn get_config(&self) -> &BenchmarkConfig {
        &self.benchmark_config
    }

    /// Gets all benchmarks
    pub fn get_benchmarks(&self) -> &Vec<Benchmark> {
        &self.benchmarks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert!(suite.is_ok());
    }

    #[test]
    fn test_comprehensive_benchmarks() {
        let mut suite = BenchmarkSuite::new().unwrap();
        let input = Array2::ones((16, 16));
        
        let result = suite.run_comprehensive_benchmarks(&input);
        assert!(result.is_ok());
        
        let benchmark_result = result.unwrap();
        assert!(benchmark_result.total_time > 0.0);
        assert!(benchmark_result.iterations > 0);
    }

    #[test]
    fn test_single_benchmark() {
        let suite = BenchmarkSuite::new().unwrap();
        let input = Array2::ones((16, 16));
        let benchmark = Benchmark {
            name: "Test Benchmark".to_string(),
            description: "Test benchmark".to_string(),
            benchmark_type: BenchmarkType::Processing,
            iterations: 10,
            warmup_iterations: 2,
            timeout_seconds: 10.0,
        };
        
        let result = suite.run_single_benchmark(&benchmark, &input);
        assert!(result.is_ok());
        
        let benchmark_result = result.unwrap();
        assert_eq!(benchmark_result.benchmark_name, "Test Benchmark");
        assert!(benchmark_result.iterations > 0);
    }

    #[test]
    fn test_benchmark_types() {
        let suite = BenchmarkSuite::new().unwrap();
        let input = Array2::ones((8, 8));
        
        let benchmark_types = vec![
            BenchmarkType::Processing,
            BenchmarkType::Memory,
            BenchmarkType::Latency,
            BenchmarkType::Throughput,
            BenchmarkType::Scalability,
            BenchmarkType::Stress,
            BenchmarkType::Regression,
        ];

        for benchmark_type in benchmark_types {
            let benchmark = Benchmark {
                name: format!("Test {:?}", benchmark_type),
                description: "Test benchmark".to_string(),
                benchmark_type,
                iterations: 5,
                warmup_iterations: 1,
                timeout_seconds: 5.0,
            };
            
            let result = suite.run_single_benchmark(&benchmark, &input);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_configuration_update() {
        let mut suite = BenchmarkSuite::new().unwrap();
        let config = BenchmarkConfig {
            enable_processing_benchmarks: false,
            enable_memory_benchmarks: true,
            enable_latency_benchmarks: false,
            enable_throughput_benchmarks: true,
            enable_scalability_benchmarks: false,
            enable_stress_benchmarks: true,
            enable_regression_benchmarks: false,
            default_iterations: 200,
            default_warmup_iterations: 20,
            default_timeout_seconds: 120.0,
        };
        
        suite.update_config(config);
        assert!(!suite.get_config().enable_processing_benchmarks);
        assert!(suite.get_config().enable_memory_benchmarks);
        assert_eq!(suite.get_config().default_iterations, 200);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0.0);
        assert_eq!(metrics.latency, 0.0);
        assert_eq!(metrics.throughput, 0.0);
    }
}
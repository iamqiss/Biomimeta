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

//! Performance Monitor for Real-Time Adaptation
//! 
//! Monitors system performance and health to inform adaptive decisions.
//! 
//! Biological Basis:
//! - Kandel et al. (2013): Principles of Neural Science - Homeostatic mechanisms
//! - Turrigiano (2008): The self-tuning neuron: Homeostatic plasticity
//! - Buzsáki (2006): Rhythms of the Brain - Neural oscillations and performance
//! - Bullmore & Sporns (2009): Complex brain networks - Network efficiency

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use crate::AfiyahError;

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub efficiency: f64,
    pub latency: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub variance: f64,
    pub throughput: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            efficiency: 0.8,
            latency: 50.0,
            memory_usage: 0.5,
            cpu_usage: 0.5,
            variance: 0.1,
            throughput: 100.0,
        }
    }
}

/// System health levels
#[derive(Debug, Clone, PartialEq)]
pub enum SystemHealth {
    Optimal,
    Good,
    Degraded,
    Critical,
}

/// Performance monitor configuration
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    pub monitoring_interval: Duration,
    pub history_size: usize,
    pub efficiency_threshold: f64,
    pub latency_threshold: f64,
    pub memory_threshold: f64,
    pub cpu_threshold: f64,
    pub variance_threshold: f64,
    pub health_check_interval: Duration,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(50),
            history_size: 1000,
            efficiency_threshold: 0.7,
            latency_threshold: 100.0,
            memory_threshold: 0.8,
            cpu_threshold: 0.8,
            variance_threshold: 0.3,
            health_check_interval: Duration::from_secs(1),
        }
    }
}

/// Performance monitor
pub struct PerformanceMonitor {
    config: PerformanceMonitorConfig,
    metrics_history: VecDeque<PerformanceMetrics>,
    health_history: VecDeque<SystemHealth>,
    last_monitoring: Instant,
    last_health_check: Instant,
    current_health: SystemHealth,
    performance_trends: PerformanceTrends,
    alert_thresholds: AlertThresholds,
}

/// Performance trends analysis
#[derive(Debug, Clone)]
struct PerformanceTrends {
    efficiency_trend: f64,
    latency_trend: f64,
    memory_trend: f64,
    cpu_trend: f64,
    variance_trend: f64,
    throughput_trend: f64,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            efficiency_trend: 0.0,
            latency_trend: 0.0,
            memory_trend: 0.0,
            cpu_trend: 0.0,
            variance_trend: 0.0,
            throughput_trend: 0.0,
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone)]
struct AlertThresholds {
    efficiency_critical: f64,
    latency_critical: f64,
    memory_critical: f64,
    cpu_critical: f64,
    variance_critical: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            efficiency_critical: 0.5,
            latency_critical: 200.0,
            memory_critical: 0.9,
            cpu_critical: 0.9,
            variance_critical: 0.5,
        }
    }
}

impl PerformanceMonitor {
    /// Creates a new performance monitor
    pub fn new() -> Result<Self, AfiyahError> {
        let config = PerformanceMonitorConfig::default();
        Self::with_config(config)
    }

    /// Creates a new performance monitor with custom configuration
    pub fn with_config(config: PerformanceMonitorConfig) -> Result<Self, AfiyahError> {
        Ok(Self {
            config: config.clone(),
            metrics_history: VecDeque::with_capacity(config.history_size),
            health_history: VecDeque::with_capacity(100),
            last_monitoring: Instant::now(),
            last_health_check: Instant::now(),
            current_health: SystemHealth::Good,
            performance_trends: PerformanceTrends::default(),
            alert_thresholds: AlertThresholds::default(),
        })
    }

    /// Collects current performance metrics
    pub fn collect_metrics(&mut self) -> Result<PerformanceMetrics, AfiyahError> {
        let now = Instant::now();
        
        // Check if enough time has passed since last monitoring
        if now.duration_since(self.last_monitoring) < self.config.monitoring_interval {
            return Ok(self.get_latest_metrics());
        }

        // Collect system metrics
        let metrics = self.measure_system_metrics()?;

        // Update metrics history
        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > self.config.history_size {
            self.metrics_history.pop_front();
        }

        // Update performance trends
        self.update_performance_trends()?;

        // Update health if needed
        if now.duration_since(self.last_health_check) >= self.config.health_check_interval {
            self.update_system_health()?;
            self.last_health_check = now;
        }

        self.last_monitoring = now;
        Ok(metrics)
    }

    /// Assesses system health based on current metrics
    pub fn assess_health(&self, metrics: &PerformanceMetrics) -> Result<SystemHealth, AfiyahError> {
        let health_score = self.calculate_health_score(metrics)?;

        let health = if health_score >= 0.9 {
            SystemHealth::Optimal
        } else if health_score >= 0.7 {
            SystemHealth::Good
        } else if health_score >= 0.5 {
            SystemHealth::Degraded
        } else {
            SystemHealth::Critical
        };

        Ok(health)
    }

    /// Gets current system health
    pub fn get_current_health(&self) -> &SystemHealth {
        &self.current_health
    }

    /// Gets performance trends
    pub fn get_performance_trends(&self) -> &PerformanceTrends {
        &self.performance_trends
    }

    /// Gets latest metrics
    pub fn get_latest_metrics(&self) -> PerformanceMetrics {
        self.metrics_history.back().cloned().unwrap_or_default()
    }

    /// Gets metrics history
    pub fn get_metrics_history(&self) -> &VecDeque<PerformanceMetrics> {
        &self.metrics_history
    }

    /// Updates the monitor configuration
    pub fn update_config(&mut self, monitoring_interval: Duration) -> Result<(), AfiyahError> {
        self.config.monitoring_interval = monitoring_interval;
        Ok(())
    }

    /// Resets the monitor
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.metrics_history.clear();
        self.health_history.clear();
        self.current_health = SystemHealth::Good;
        self.performance_trends = PerformanceTrends::default();
        self.last_monitoring = Instant::now();
        self.last_health_check = Instant::now();
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &PerformanceMonitorConfig {
        &self.config
    }

    fn measure_system_metrics(&self) -> Result<PerformanceMetrics, AfiyahError> {
        // In a real implementation, this would collect actual system metrics
        // For now, we'll simulate metrics based on current system state
        
        let efficiency = self.simulate_efficiency()?;
        let latency = self.simulate_latency()?;
        let memory_usage = self.simulate_memory_usage()?;
        let cpu_usage = self.simulate_cpu_usage()?;
        let variance = self.calculate_variance()?;
        let throughput = self.simulate_throughput()?;

        Ok(PerformanceMetrics {
            efficiency,
            latency,
            memory_usage,
            cpu_usage,
            variance,
            throughput,
        })
    }

    fn simulate_efficiency(&self) -> Result<f64, AfiyahError> {
        // Simulate efficiency based on current system state
        let base_efficiency = 0.8;
        let random_factor = (rand::random::<f64>() - 0.5) * 0.2;
        let trend_factor = self.performance_trends.efficiency_trend * 0.1;
        
        let efficiency = base_efficiency + random_factor + trend_factor;
        Ok(efficiency.max(0.0).min(1.0))
    }

    fn simulate_latency(&self) -> Result<f64, AfiyahError> {
        // Simulate latency based on current system state
        let base_latency = 50.0;
        let random_factor = (rand::random::<f64>() - 0.5) * 20.0;
        let trend_factor = self.performance_trends.latency_trend * 10.0;
        
        let latency = base_latency + random_factor + trend_factor;
        Ok(latency.max(0.0))
    }

    fn simulate_memory_usage(&self) -> Result<f64, AfiyahError> {
        // Simulate memory usage based on current system state
        let base_memory = 0.5;
        let random_factor = (rand::random::<f64>() - 0.5) * 0.2;
        let trend_factor = self.performance_trends.memory_trend * 0.1;
        
        let memory_usage = base_memory + random_factor + trend_factor;
        Ok(memory_usage.max(0.0).min(1.0))
    }

    fn simulate_cpu_usage(&self) -> Result<f64, AfiyahError> {
        // Simulate CPU usage based on current system state
        let base_cpu = 0.5;
        let random_factor = (rand::random::<f64>() - 0.5) * 0.2;
        let trend_factor = self.performance_trends.cpu_trend * 0.1;
        
        let cpu_usage = base_cpu + random_factor + trend_factor;
        Ok(cpu_usage.max(0.0).min(1.0))
    }

    fn simulate_throughput(&self) -> Result<f64, AfiyahError> {
        // Simulate throughput based on current system state
        let base_throughput = 100.0;
        let random_factor = (rand::random::<f64>() - 0.5) * 20.0;
        let trend_factor = self.performance_trends.throughput_trend * 10.0;
        
        let throughput = base_throughput + random_factor + trend_factor;
        Ok(throughput.max(0.0))
    }

    fn calculate_variance(&self) -> Result<f64, AfiyahError> {
        if self.metrics_history.len() < 2 {
            return Ok(0.1);
        }

        let recent_metrics: Vec<&PerformanceMetrics> = self.metrics_history.iter().rev().take(10).collect();
        
        // Calculate variance of efficiency
        let efficiency_values: Vec<f64> = recent_metrics.iter().map(|m| m.efficiency).collect();
        let mean_efficiency = efficiency_values.iter().sum::<f64>() / efficiency_values.len() as f64;
        let efficiency_variance = efficiency_values.iter()
            .map(|&x| (x - mean_efficiency).powi(2))
            .sum::<f64>() / efficiency_values.len() as f64;

        Ok(efficiency_variance.sqrt())
    }

    fn update_performance_trends(&mut self) -> Result<(), AfiyahError> {
        if self.metrics_history.len() < 2 {
            return Ok(());
        }

        let recent_metrics: Vec<&PerformanceMetrics> = self.metrics_history.iter().rev().take(20).collect();
        
        // Calculate trends using linear regression
        self.performance_trends.efficiency_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.efficiency).collect())?;
        self.performance_trends.latency_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.latency).collect())?;
        self.performance_trends.memory_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.memory_usage).collect())?;
        self.performance_trends.cpu_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.cpu_usage).collect())?;
        self.performance_trends.variance_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.variance).collect())?;
        self.performance_trends.throughput_trend = self.calculate_trend(
            &recent_metrics.iter().map(|m| m.throughput).collect())?;

        Ok(())
    }

    fn calculate_trend(&self, values: &[f64]) -> Result<f64, AfiyahError> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Ok(slope)
    }

    fn update_system_health(&mut self) -> Result<(), AfiyahError> {
        if let Some(latest_metrics) = self.metrics_history.back() {
            let new_health = self.assess_health(latest_metrics)?;
            
            if new_health != self.current_health {
                self.current_health = new_health;
                self.health_history.push_back(self.current_health.clone());
                
                if self.health_history.len() > 100 {
                    self.health_history.pop_front();
                }
            }
        }

        Ok(())
    }

    fn calculate_health_score(&self, metrics: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // Calculate health score based on multiple factors
        let efficiency_score = if metrics.efficiency >= self.config.efficiency_threshold {
            1.0
        } else {
            metrics.efficiency / self.config.efficiency_threshold
        };

        let latency_score = if metrics.latency <= self.config.latency_threshold {
            1.0
        } else {
            self.config.latency_threshold / metrics.latency
        };

        let memory_score = if metrics.memory_usage <= self.config.memory_threshold {
            1.0
        } else {
            self.config.memory_threshold / metrics.memory_usage
        };

        let cpu_score = if metrics.cpu_usage <= self.config.cpu_threshold {
            1.0
        } else {
            self.config.cpu_threshold / metrics.cpu_usage
        };

        let variance_score = if metrics.variance <= self.config.variance_threshold {
            1.0
        } else {
            self.config.variance_threshold / metrics.variance
        };

        // Weighted combination
        let health_score = (
            efficiency_score * 0.3 +
            latency_score * 0.25 +
            memory_score * 0.2 +
            cpu_score * 0.15 +
            variance_score * 0.1
        );

        Ok(health_score.min(1.0).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_metrics_collection() {
        let mut monitor = PerformanceMonitor::new().unwrap();
        let result = monitor.collect_metrics();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.efficiency >= 0.0 && metrics.efficiency <= 1.0);
        assert!(metrics.latency >= 0.0);
        assert!(metrics.memory_usage >= 0.0 && metrics.memory_usage <= 1.0);
    }

    #[test]
    fn test_health_assessment() {
        let monitor = PerformanceMonitor::new().unwrap();
        let metrics = PerformanceMetrics::default();
        
        let result = monitor.assess_health(&metrics);
        assert!(result.is_ok());
        
        let health = result.unwrap();
        assert!(matches!(health, SystemHealth::Good | SystemHealth::Optimal));
    }

    #[test]
    fn test_trend_calculation() {
        let monitor = PerformanceMonitor::new().unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = monitor.calculate_trend(&values);
        assert!(result.is_ok());
        
        let trend = result.unwrap();
        assert!(trend > 0.0); // Should be positive trend
    }
}
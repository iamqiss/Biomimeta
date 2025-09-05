//! Frame Scheduler Module

use crate::AfiyahError;

/// Frame scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_queue_size: usize,
    pub priority_threshold: f64,
    pub scheduling_algorithm: SchedulingAlgorithm,
    pub biological_optimization: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 100,
            priority_threshold: 0.5,
            scheduling_algorithm: SchedulingAlgorithm::Biological,
            biological_optimization: true,
        }
    }
}

/// Scheduling algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingAlgorithm {
    FIFO,
    Priority,
    Biological,
    Adaptive,
}

/// Frame priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum FramePriority {
    Critical,
    High,
    Medium,
    Low,
}

impl FramePriority {
    pub fn from_quality(quality: f64) -> Self {
        if quality >= 0.9 {
            FramePriority::Critical
        } else if quality >= 0.7 {
            FramePriority::High
        } else if quality >= 0.5 {
            FramePriority::Medium
        } else {
            FramePriority::Low
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            FramePriority::Critical => 1.0,
            FramePriority::High => 0.8,
            FramePriority::Medium => 0.6,
            FramePriority::Low => 0.4,
        }
    }
}

/// Frame scheduler implementing biological frame scheduling
pub struct FrameScheduler {
    config: SchedulerConfig,
    frame_queue: Vec<ScheduledFrame>,
    scheduling_history: Vec<f64>,
}

/// Scheduled frame with priority and metadata
#[derive(Debug, Clone)]
struct ScheduledFrame {
    data: Vec<u8>,
    priority: FramePriority,
    quality: f64,
    timestamp: u64,
    biological_weight: f64,
}

impl FrameScheduler {
    /// Creates a new frame scheduler
    pub fn new() -> Result<Self, AfiyahError> {
        let config = SchedulerConfig::default();
        let frame_queue = Vec::new();
        let scheduling_history = Vec::new();

        Ok(Self {
            config,
            frame_queue,
            scheduling_history,
        })
    }

    /// Initializes the frame scheduler
    pub fn initialize(&mut self) -> Result<(), AfiyahError> {
        self.frame_queue.clear();
        self.scheduling_history.clear();
        Ok(())
    }

    /// Stops the frame scheduler
    pub fn stop(&mut self) -> Result<(), AfiyahError> {
        self.frame_queue.clear();
        self.scheduling_history.clear();
        Ok(())
    }

    /// Schedules a frame for transmission
    pub fn schedule_frame(&mut self, frame_data: Vec<u8>, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<Vec<u8>, AfiyahError> {
        // Determine frame priority
        let priority = FramePriority::from_quality(quality_params.overall_quality);
        
        // Calculate biological weight
        let biological_weight = self.calculate_biological_weight(quality_params)?;

        // Create scheduled frame
        let scheduled_frame = ScheduledFrame {
            data: frame_data,
            priority,
            quality: quality_params.overall_quality,
            timestamp: self.get_current_timestamp(),
            biological_weight,
        };

        // Add to queue
        self.frame_queue.push(scheduled_frame);

        // Apply scheduling algorithm
        let scheduled_data = self.apply_scheduling_algorithm()?;

        // Update scheduling history
        self.scheduling_history.push(quality_params.overall_quality);
        if self.scheduling_history.len() > 100 {
            self.scheduling_history.remove(0);
        }

        Ok(scheduled_data)
    }

    fn calculate_biological_weight(&self, quality_params: &crate::streaming_engine::biological_qos::PerceptualQuality) -> Result<f64, AfiyahError> {
        if !self.config.biological_optimization {
            return Ok(1.0);
        }

        // Calculate biological weight based on perceptual quality
        let foveal_weight = quality_params.foveal_quality * 0.4;
        let peripheral_weight = quality_params.peripheral_quality * 0.2;
        let motion_weight = quality_params.motion_quality * 0.2;
        let color_weight = quality_params.color_quality * 0.1;
        let temporal_weight = quality_params.temporal_quality * 0.1;

        let biological_weight = foveal_weight + peripheral_weight + motion_weight + color_weight + temporal_weight;
        Ok(biological_weight.clamp(0.0, 1.0))
    }

    fn apply_scheduling_algorithm(&mut self) -> Result<Vec<u8>, AfiyahError> {
        if self.frame_queue.is_empty() {
            return Ok(Vec::new());
        }

        match self.config.scheduling_algorithm {
            SchedulingAlgorithm::FIFO => self.schedule_fifo(),
            SchedulingAlgorithm::Priority => self.schedule_priority(),
            SchedulingAlgorithm::Biological => self.schedule_biological(),
            SchedulingAlgorithm::Adaptive => self.schedule_adaptive(),
        }
    }

    fn schedule_fifo(&mut self) -> Result<Vec<u8>, AfiyahError> {
        // First in, first out scheduling
        if let Some(frame) = self.frame_queue.pop() {
            Ok(frame.data)
        } else {
            Ok(Vec::new())
        }
    }

    fn schedule_priority(&mut self) -> Result<Vec<u8>, AfiyahError> {
        // Priority-based scheduling
        self.frame_queue.sort_by(|a, b| {
            let priority_a = a.priority.weight();
            let priority_b = b.priority.weight();
            priority_b.partial_cmp(&priority_a).unwrap()
        });

        if let Some(frame) = self.frame_queue.pop() {
            Ok(frame.data)
        } else {
            Ok(Vec::new())
        }
    }

    fn schedule_biological(&mut self) -> Result<Vec<u8>, AfiyahError> {
        // Biological scheduling based on perceptual importance
        self.frame_queue.sort_by(|a, b| {
            let weight_a = a.biological_weight * a.priority.weight();
            let weight_b = b.biological_weight * b.priority.weight();
            weight_b.partial_cmp(&weight_a).unwrap()
        });

        if let Some(frame) = self.frame_queue.pop() {
            Ok(frame.data)
        } else {
            Ok(Vec::new())
        }
    }

    fn schedule_adaptive(&mut self) -> Result<Vec<u8>, AfiyahError> {
        // Adaptive scheduling based on quality history
        let avg_quality = if self.scheduling_history.is_empty() {
            0.5
        } else {
            self.scheduling_history.iter().sum::<f64>() / self.scheduling_history.len() as f64
        };

        // Adjust scheduling based on quality trends
        if avg_quality < 0.6 {
            // Low quality - prioritize high-quality frames
            self.frame_queue.sort_by(|a, b| {
                let weight_a = a.quality * a.biological_weight;
                let weight_b = b.quality * b.biological_weight;
                weight_b.partial_cmp(&weight_a).unwrap()
            });
        } else {
            // High quality - use biological scheduling
            self.schedule_biological()
        }
    }

    fn get_current_timestamp(&self) -> u64 {
        // In a real implementation, this would return the current timestamp
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Updates configuration
    pub fn update_config(&mut self, config: SchedulerConfig) {
        self.config = config;
    }

    /// Gets queue size
    pub fn get_queue_size(&self) -> usize {
        self.frame_queue.len()
    }

    /// Gets scheduling history
    pub fn get_scheduling_history(&self) -> &Vec<f64> {
        &self.scheduling_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.max_queue_size, 100);
        assert_eq!(config.scheduling_algorithm, SchedulingAlgorithm::Biological);
    }

    #[test]
    fn test_frame_priority_from_quality() {
        assert_eq!(FramePriority::from_quality(0.95), FramePriority::Critical);
        assert_eq!(FramePriority::from_quality(0.8), FramePriority::High);
        assert_eq!(FramePriority::from_quality(0.6), FramePriority::Medium);
        assert_eq!(FramePriority::from_quality(0.3), FramePriority::Low);
    }

    #[test]
    fn test_frame_priority_weight() {
        assert_eq!(FramePriority::Critical.weight(), 1.0);
        assert_eq!(FramePriority::High.weight(), 0.8);
        assert_eq!(FramePriority::Medium.weight(), 0.6);
        assert_eq!(FramePriority::Low.weight(), 0.4);
    }

    #[test]
    fn test_frame_scheduler_creation() {
        let scheduler = FrameScheduler::new();
        assert!(scheduler.is_ok());
    }

    #[test]
    fn test_scheduler_initialization() {
        let mut scheduler = FrameScheduler::new().unwrap();
        let result = scheduler.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_frame_scheduling() {
        let mut scheduler = FrameScheduler::new().unwrap();
        scheduler.initialize().unwrap();
        
        let frame_data = vec![1, 2, 3, 4, 5];
        let quality_params = crate::streaming_engine::biological_qos::PerceptualQuality::new();
        
        let result = scheduler.schedule_frame(frame_data, &quality_params);
        assert!(result.is_ok());
    }
}
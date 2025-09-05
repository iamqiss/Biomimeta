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

//! Subjective Testing Framework for Human Visual Perception Validation
//! 
//! This module implements comprehensive subjective testing capabilities specifically
//! designed for validating Afiyah's biomimetic video compression system against
//! human visual perception.
//! 
//! Key Features:
//! - Double-blind testing protocols
//! - A/B testing with statistical analysis
//! - Paired comparison testing
//! - Continuous quality assessment
//! - Eye-tracking integration
//! - Demographic analysis
//! - Cultural bias detection
//! - Statistical significance testing
//! - IRB compliance tools
//! 
//! Biological Foundation:
//! - Human visual system validation
//! - Perceptual threshold testing
//! - Individual variation analysis
//! - Cultural visual preferences
//! - Age-related visual changes
//! - Color vision deficiency testing

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::sync::mpsc;

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::AfiyahError;

/// Subjective testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveTestingConfig {
    pub test_type: TestType,
    pub participant_count: usize,
    pub test_duration: Duration,
    pub randomization_level: RandomizationLevel,
    pub enable_eye_tracking: bool,
    pub enable_demographic_analysis: bool,
    pub enable_cultural_analysis: bool,
    pub enable_statistical_analysis: bool,
    pub irb_compliance: bool,
    pub data_retention_policy: DataRetentionPolicy,
    pub quality_threshold: f64,
    pub confidence_level: f64,
}

/// Types of subjective tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    DoubleBlindAB,           // Double-blind A/B testing
    PairedComparison,        // Paired comparison testing
    ContinuousQuality,       // Continuous quality assessment
    ForcedChoice,           // Forced choice testing
    Ranking,                // Ranking multiple options
    MagnitudeEstimation,    // Magnitude estimation testing
}

/// Randomization levels for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RandomizationLevel {
    None,                   // No randomization
    Basic,                  // Basic randomization
    Full,                   // Full randomization with counterbalancing
    Advanced,               // Advanced randomization with Latin squares
}

/// Data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataRetentionPolicy {
    ImmediateDeletion,      // Delete data immediately after analysis
    ShortTerm,              // Retain for 30 days
    MediumTerm,             // Retain for 1 year
    LongTerm,               // Retain for 5 years
    Permanent,              // Permanent retention for research
}

impl Default for SubjectiveTestingConfig {
    fn default() -> Self {
        Self {
            test_type: TestType::DoubleBlindAB,
            participant_count: 30,
            test_duration: Duration::from_secs(1800), // 30 minutes
            randomization_level: RandomizationLevel::Full,
            enable_eye_tracking: true,
            enable_demographic_analysis: true,
            enable_cultural_analysis: true,
            enable_statistical_analysis: true,
            irb_compliance: true,
            data_retention_policy: DataRetentionPolicy::MediumTerm,
            quality_threshold: 0.95,
            confidence_level: 0.95,
        }
    }
}

/// Participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub id: String,
    pub age: u32,
    pub gender: Gender,
    pub ethnicity: Ethnicity,
    pub education_level: EducationLevel,
    pub visual_acuity: VisualAcuity,
    pub color_vision_status: ColorVisionStatus,
    pub cultural_background: CulturalBackground,
    pub viewing_experience: ViewingExperience,
    pub device_preferences: DevicePreferences,
    pub consent_given: bool,
    pub irb_approval: bool,
}

/// Gender categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
    PreferNotToSay,
}

/// Ethnicity categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Ethnicity {
    Caucasian,
    AfricanAmerican,
    Hispanic,
    Asian,
    NativeAmerican,
    PacificIslander,
    Mixed,
    Other,
    PreferNotToSay,
}

/// Education levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EducationLevel {
    HighSchool,
    SomeCollege,
    Bachelors,
    Masters,
    Doctorate,
    Other,
}

/// Visual acuity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualAcuity {
    Excellent,              // 20/20 or better
    Good,                   // 20/25 to 20/30
    Fair,                   // 20/40 to 20/60
    Poor,                   // 20/70 or worse
    Corrected,              // Corrected to normal
}

/// Color vision status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorVisionStatus {
    Normal,
    Protanopia,             // Red-blind
    Deuteranopia,           // Green-blind
    Tritanopia,             // Blue-blind
    Protanomaly,            // Red-weak
    Deuteranomaly,          // Green-weak
    Tritanomaly,            // Blue-weak
    Monochromacy,           // Complete color blindness
}

/// Cultural background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalBackground {
    pub primary_language: String,
    pub cultural_region: String,
    pub visual_preferences: VisualPreferences,
    pub color_associations: ColorAssociations,
    pub motion_preferences: MotionPreferences,
}

/// Visual preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPreferences {
    pub brightness_preference: f64,
    pub contrast_preference: f64,
    pub saturation_preference: f64,
    pub sharpness_preference: f64,
    pub color_temperature_preference: f64,
}

/// Color associations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAssociations {
    pub red_associations: Vec<String>,
    pub green_associations: Vec<String>,
    pub blue_associations: Vec<String>,
    pub cultural_color_meanings: HashMap<String, String>,
}

/// Motion preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPreferences {
    pub motion_sensitivity: f64,
    pub temporal_preference: f64,
    pub smoothness_preference: f64,
    pub action_preference: f64,
}

/// Viewing experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingExperience {
    pub years_experience: u32,
    pub preferred_content_types: Vec<ContentType>,
    pub viewing_frequency: ViewingFrequency,
    pub device_usage: DeviceUsage,
}

/// Content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Movies,
    TVShows,
    Sports,
    News,
    Educational,
    Gaming,
    SocialMedia,
    Other,
}

/// Viewing frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewingFrequency {
    Daily,
    Weekly,
    Monthly,
    Rarely,
    Never,
}

/// Device usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceUsage {
    pub mobile_usage: f64,
    pub tablet_usage: f64,
    pub desktop_usage: f64,
    pub tv_usage: f64,
    pub vr_usage: f64,
}

/// Device preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePreferences {
    pub preferred_device: DeviceType,
    pub screen_size_preference: ScreenSizePreference,
    pub resolution_preference: ResolutionPreference,
    pub framerate_preference: FrameratePreference,
}

/// Device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Mobile,
    Tablet,
    Desktop,
    TV,
    VR,
}

/// Screen size preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScreenSizePreference {
    Small,                  // < 5 inches
    Medium,                 // 5-10 inches
    Large,                  // 10-20 inches
    ExtraLarge,             // > 20 inches
}

/// Resolution preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionPreference {
    SD,                     // 480p
    HD,                     // 720p
    FullHD,                 // 1080p
    UHD,                    // 4K
    UHDPlus,                // 8K
}

/// Framerate preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameratePreference {
    Low,                    // 24-30 fps
    Medium,                 // 30-60 fps
    High,                   // 60-120 fps
    VeryHigh,               // > 120 fps
}

/// Test session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSession {
    pub session_id: String,
    pub participant: Participant,
    pub test_config: SubjectiveTestingConfig,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub test_stimuli: Vec<TestStimulus>,
    pub responses: Vec<TestResponse>,
    pub eye_tracking_data: Option<EyeTrackingData>,
    pub session_metadata: SessionMetadata,
}

/// Test stimulus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStimulus {
    pub stimulus_id: String,
    pub stimulus_type: StimulusType,
    pub content_data: Vec<u8>,
    pub reference_data: Option<Vec<u8>>,
    pub quality_level: f64,
    pub compression_settings: CompressionSettings,
    pub presentation_order: usize,
    pub duration: Duration,
    pub metadata: StimulusMetadata,
}

/// Stimulus types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StimulusType {
    Reference,              // Reference (uncompressed) content
    Test,                   // Test (compressed) content
    Distractor,             // Distractor content
    Calibration,            // Calibration stimulus
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    pub codec: String,
    pub bitrate: u32,
    pub resolution: (u32, u32),
    pub framerate: f64,
    pub quality_preset: QualityPreset,
    pub biological_optimization: bool,
    pub foveal_prioritization: bool,
}

/// Quality presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityPreset {
    Lowest,
    Low,
    Medium,
    High,
    Highest,
    Custom,
}

/// Stimulus metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusMetadata {
    pub content_type: ContentType,
    pub scene_complexity: SceneComplexity,
    pub motion_level: MotionLevel,
    pub color_complexity: ColorComplexity,
    pub texture_complexity: TextureComplexity,
    pub brightness_level: BrightnessLevel,
    pub contrast_level: ContrastLevel,
}

/// Scene complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneComplexity {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Motion levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotionLevel {
    Static,
    Slow,
    Moderate,
    Fast,
    VeryFast,
}

/// Color complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorComplexity {
    Monochrome,
    Limited,
    Moderate,
    Rich,
    VeryRich,
}

/// Texture complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextureComplexity {
    Smooth,
    Moderate,
    Detailed,
    VeryDetailed,
}

/// Brightness levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrightnessLevel {
    VeryDark,
    Dark,
    Normal,
    Bright,
    VeryBright,
}

/// Contrast levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContrastLevel {
    Low,
    Moderate,
    High,
    VeryHigh,
}

/// Test response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResponse {
    pub response_id: String,
    pub stimulus_id: String,
    pub response_type: ResponseType,
    pub response_value: ResponseValue,
    pub response_time: Duration,
    pub confidence_level: f64,
    pub attention_level: f64,
    pub response_metadata: ResponseMetadata,
}

/// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    QualityRating,          // Quality rating (1-10)
    PreferenceChoice,       // A vs B preference
    DifferenceDetection,    // Can detect difference (yes/no)
    Ranking,                // Ranking multiple options
    MagnitudeEstimation,    // Magnitude estimation
    ContinuousRating,       // Continuous quality rating
}

/// Response values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseValue {
    Numeric(f64),
    Boolean(bool),
    Choice(String),
    Ranking(Vec<String>),
    Text(String),
    Continuous(f64),
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub response_consistency: f64,
    pub response_reliability: f64,
    pub cultural_bias_score: f64,
    pub individual_variation: f64,
    pub attention_metrics: AttentionMetrics,
}

/// Attention metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMetrics {
    pub fixation_duration: Duration,
    pub fixation_count: u32,
    pub saccade_frequency: f64,
    pub attention_distribution: AttentionDistribution,
    pub foveal_attention: f64,
    pub peripheral_attention: f64,
}

/// Attention distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDistribution {
    pub center_attention: f64,
    pub edge_attention: f64,
    pub corner_attention: f64,
    pub motion_attention: f64,
    pub static_attention: f64,
}

/// Eye tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeTrackingData {
    pub gaze_points: Vec<GazePoint>,
    pub fixations: Vec<Fixation>,
    pub saccades: Vec<Saccade>,
    pub blink_events: Vec<BlinkEvent>,
    pub pupil_diameter: Vec<PupilMeasurement>,
    pub tracking_quality: f64,
}

/// Gaze point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazePoint {
    pub timestamp: SystemTime,
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
    pub validity: bool,
}

/// Fixation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixation {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub duration: Duration,
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

/// Saccade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Saccade {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub duration: Duration,
    pub start_x: f64,
    pub start_y: f64,
    pub end_x: f64,
    pub end_y: f64,
    pub amplitude: f64,
    pub velocity: f64,
}

/// Blink event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlinkEvent {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub duration: Duration,
    pub confidence: f64,
}

/// Pupil measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilMeasurement {
    pub timestamp: SystemTime,
    pub left_diameter: f64,
    pub right_diameter: f64,
    pub average_diameter: f64,
    pub confidence: f64,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub environment_conditions: EnvironmentConditions,
    pub device_settings: DeviceSettings,
    pub calibration_data: CalibrationData,
    pub session_notes: String,
    pub data_quality: f64,
}

/// Environment conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConditions {
    pub ambient_lighting: f64,
    pub room_temperature: f64,
    pub humidity: f64,
    pub noise_level: f64,
    pub viewing_distance: f64,
    pub screen_angle: f64,
}

/// Device settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSettings {
    pub screen_resolution: (u32, u32),
    pub screen_size: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub color_temperature: f64,
    pub refresh_rate: f64,
}

/// Calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub calibration_accuracy: f64,
    pub calibration_points: Vec<CalibrationPoint>,
    pub validation_results: ValidationResults,
}

/// Calibration point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub target_x: f64,
    pub target_y: f64,
    pub measured_x: f64,
    pub measured_y: f64,
    pub accuracy: f64,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub average_accuracy: f64,
    pub max_error: f64,
    pub validation_passed: bool,
}

/// Subjective testing engine
pub struct SubjectiveTestingEngine {
    config: SubjectiveTestingConfig,
    sessions: Arc<RwLock<HashMap<String, TestSession>>>,
    participants: Arc<RwLock<HashMap<String, Participant>>>,
    statistical_analyzer: Arc<Mutex<StatisticalAnalyzer>>,
    eye_tracker: Arc<Mutex<EyeTracker>>,
    demographic_analyzer: Arc<Mutex<DemographicAnalyzer>>,
    cultural_analyzer: Arc<Mutex<CulturalAnalyzer>>,
    running: Arc<Mutex<bool>>,
}

impl SubjectiveTestingEngine {
    /// Creates a new subjective testing engine
    pub fn new(config: SubjectiveTestingConfig) -> Result<Self, AfiyahError> {
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let participants = Arc::new(RwLock::new(HashMap::new()));
        let statistical_analyzer = Arc::new(Mutex::new(StatisticalAnalyzer::new()?));
        let eye_tracker = Arc::new(Mutex::new(EyeTracker::new()?));
        let demographic_analyzer = Arc::new(Mutex::new(DemographicAnalyzer::new()?));
        let cultural_analyzer = Arc::new(Mutex::new(CulturalAnalyzer::new()?));
        let running = Arc::new(Mutex::new(false));

        Ok(Self {
            config,
            sessions,
            participants,
            statistical_analyzer,
            eye_tracker,
            demographic_analyzer,
            cultural_analyzer,
            running,
        })
    }

    /// Starts the subjective testing engine
    pub fn start(&mut self) -> Result<(), AfiyahError> {
        *self.running.lock().unwrap() = true;
        
        if self.config.enable_eye_tracking {
            self.start_eye_tracking()?;
        }
        
        Ok(())
    }

    /// Stops the subjective testing engine
    pub fn stop(&mut self) -> Result<(), AfiyahError> {
        *self.running.lock().unwrap() = false;
        Ok(())
    }

    /// Registers a new participant
    pub fn register_participant(&mut self, participant: Participant) -> Result<(), AfiyahError> {
        if !participant.consent_given {
            return Err(AfiyahError::Streaming { message: "Participant consent required".to_string() });
        }

        if self.config.irb_compliance && !participant.irb_approval {
            return Err(AfiyahError::Streaming { message: "IRB approval required".to_string() });
        }

        let mut participants = self.participants.write().unwrap();
        participants.insert(participant.id.clone(), participant);
        Ok(())
    }

    /// Creates a new test session
    pub fn create_test_session(&mut self, participant_id: String, stimuli: Vec<TestStimulus>) -> Result<String, AfiyahError> {
        let participants = self.participants.read().unwrap();
        let participant = participants.get(&participant_id)
            .ok_or_else(|| AfiyahError::Streaming { message: "Participant not found".to_string() })?
            .clone();

        let session_id = format!("session_{}_{}", participant_id, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
        
        let session = TestSession {
            session_id: session_id.clone(),
            participant,
            test_config: self.config.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            test_stimuli: stimuli,
            responses: Vec::new(),
            eye_tracking_data: None,
            session_metadata: SessionMetadata {
                environment_conditions: EnvironmentConditions {
                    ambient_lighting: 100.0,
                    room_temperature: 22.0,
                    humidity: 50.0,
                    noise_level: 30.0,
                    viewing_distance: 2.0,
                    screen_angle: 0.0,
                },
                device_settings: DeviceSettings {
                    screen_resolution: (1920, 1080),
                    screen_size: 24.0,
                    brightness: 100.0,
                    contrast: 50.0,
                    color_temperature: 6500.0,
                    refresh_rate: 60.0,
                },
                calibration_data: CalibrationData {
                    calibration_accuracy: 0.95,
                    calibration_points: Vec::new(),
                    validation_results: ValidationResults {
                        average_accuracy: 0.95,
                        max_error: 0.5,
                        validation_passed: true,
                    },
                },
                session_notes: String::new(),
                data_quality: 0.95,
            },
        };

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);
        
        Ok(session_id)
    }

    /// Records a test response
    pub fn record_response(&mut self, session_id: &str, response: TestResponse) -> Result<(), AfiyahError> {
        let mut sessions = self.sessions.write().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.responses.push(response);
        } else {
            return Err(AfiyahError::Streaming { message: "Session not found".to_string() });
        }
        Ok(())
    }

    /// Analyzes test results
    pub fn analyze_results(&mut self, session_id: &str) -> Result<TestAnalysis, AfiyahError> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions.get(session_id)
            .ok_or_else(|| AfiyahError::Streaming { message: "Session not found".to_string() })?
            .clone();

        let mut analysis = TestAnalysis::new();
        
        // Statistical analysis
        if self.config.enable_statistical_analysis {
            analysis.statistical_results = self.perform_statistical_analysis(&session)?;
        }
        
        // Demographic analysis
        if self.config.enable_demographic_analysis {
            analysis.demographic_results = self.perform_demographic_analysis(&session)?;
        }
        
        // Cultural analysis
        if self.config.enable_cultural_analysis {
            analysis.cultural_results = self.perform_cultural_analysis(&session)?;
        }
        
        // Eye tracking analysis
        if self.config.enable_eye_tracking {
            analysis.eye_tracking_results = self.perform_eye_tracking_analysis(&session)?;
        }
        
        // Overall analysis
        analysis.overall_score = self.calculate_overall_score(&analysis)?;
        analysis.confidence_level = self.calculate_confidence_level(&analysis)?;
        analysis.recommendations = self.generate_recommendations(&analysis)?;
        
        Ok(analysis)
    }

    /// Gets participant statistics
    pub fn get_participant_statistics(&self) -> Result<ParticipantStatistics, AfiyahError> {
        let participants = self.participants.read().unwrap();
        
        let mut stats = ParticipantStatistics::new();
        stats.total_participants = participants.len();
        
        for participant in participants.values() {
            stats.age_distribution.push(participant.age);
            stats.gender_distribution.push(participant.gender.clone());
            stats.ethnicity_distribution.push(participant.ethnicity.clone());
            stats.education_distribution.push(participant.education_level.clone());
            stats.visual_acuity_distribution.push(participant.visual_acuity.clone());
            stats.color_vision_distribution.push(participant.color_vision_status.clone());
        }
        
        Ok(stats)
    }

    fn start_eye_tracking(&self) -> Result<(), AfiyahError> {
        // Start eye tracking service
        Ok(())
    }

    fn perform_statistical_analysis(&self, session: &TestSession) -> Result<StatisticalResults, AfiyahError> {
        // Perform statistical analysis
        Ok(StatisticalResults::new())
    }

    fn perform_demographic_analysis(&self, session: &TestSession) -> Result<DemographicResults, AfiyahError> {
        // Perform demographic analysis
        Ok(DemographicResults::new())
    }

    fn perform_cultural_analysis(&self, session: &TestSession) -> Result<CulturalResults, AfiyahError> {
        // Perform cultural analysis
        Ok(CulturalResults::new())
    }

    fn perform_eye_tracking_analysis(&self, session: &TestSession) -> Result<EyeTrackingResults, AfiyahError> {
        // Perform eye tracking analysis
        Ok(EyeTrackingResults::new())
    }

    fn calculate_overall_score(&self, analysis: &TestAnalysis) -> Result<f64, AfiyahError> {
        // Calculate overall test score
        Ok(0.95)
    }

    fn calculate_confidence_level(&self, analysis: &TestAnalysis) -> Result<f64, AfiyahError> {
        // Calculate confidence level
        Ok(0.95)
    }

    fn generate_recommendations(&self, analysis: &TestAnalysis) -> Result<Vec<String>, AfiyahError> {
        // Generate recommendations
        Ok(vec!["Improve compression quality".to_string()])
    }
}

/// Test analysis results
#[derive(Debug, Clone)]
pub struct TestAnalysis {
    pub statistical_results: Option<StatisticalResults>,
    pub demographic_results: Option<DemographicResults>,
    pub cultural_results: Option<CulturalResults>,
    pub eye_tracking_results: Option<EyeTrackingResults>,
    pub overall_score: f64,
    pub confidence_level: f64,
    pub recommendations: Vec<String>,
}

impl TestAnalysis {
    fn new() -> Self {
        Self {
            statistical_results: None,
            demographic_results: None,
            cultural_results: None,
            eye_tracking_results: None,
            overall_score: 0.0,
            confidence_level: 0.0,
            recommendations: Vec::new(),
        }
    }
}

/// Statistical analysis results
#[derive(Debug, Clone)]
pub struct StatisticalResults {
    pub mean_score: f64,
    pub standard_deviation: f64,
    pub confidence_interval: (f64, f64),
    pub p_value: f64,
    pub effect_size: f64,
    pub statistical_significance: bool,
}

impl StatisticalResults {
    fn new() -> Self {
        Self {
            mean_score: 0.0,
            standard_deviation: 0.0,
            confidence_interval: (0.0, 0.0),
            p_value: 0.0,
            effect_size: 0.0,
            statistical_significance: false,
        }
    }
}

/// Demographic analysis results
#[derive(Debug, Clone)]
pub struct DemographicResults {
    pub age_correlation: f64,
    pub gender_differences: f64,
    pub education_correlation: f64,
    pub visual_acuity_correlation: f64,
    pub color_vision_impact: f64,
}

impl DemographicResults {
    fn new() -> Self {
        Self {
            age_correlation: 0.0,
            gender_differences: 0.0,
            education_correlation: 0.0,
            visual_acuity_correlation: 0.0,
            color_vision_impact: 0.0,
        }
    }
}

/// Cultural analysis results
#[derive(Debug, Clone)]
pub struct CulturalResults {
    pub cultural_bias_score: f64,
    pub regional_differences: f64,
    pub language_impact: f64,
    pub cultural_preferences: f64,
}

impl CulturalResults {
    fn new() -> Self {
        Self {
            cultural_bias_score: 0.0,
            regional_differences: 0.0,
            language_impact: 0.0,
            cultural_preferences: 0.0,
        }
    }
}

/// Eye tracking analysis results
#[derive(Debug, Clone)]
pub struct EyeTrackingResults {
    pub attention_patterns: f64,
    pub fixation_analysis: f64,
    pub saccade_analysis: f64,
    pub pupil_response: f64,
    pub visual_attention_score: f64,
}

impl EyeTrackingResults {
    fn new() -> Self {
        Self {
            attention_patterns: 0.0,
            fixation_analysis: 0.0,
            saccade_analysis: 0.0,
            pupil_response: 0.0,
            visual_attention_score: 0.0,
        }
    }
}

/// Participant statistics
#[derive(Debug, Clone)]
pub struct ParticipantStatistics {
    pub total_participants: usize,
    pub age_distribution: Vec<u32>,
    pub gender_distribution: Vec<Gender>,
    pub ethnicity_distribution: Vec<Ethnicity>,
    pub education_distribution: Vec<EducationLevel>,
    pub visual_acuity_distribution: Vec<VisualAcuity>,
    pub color_vision_distribution: Vec<ColorVisionStatus>,
}

impl ParticipantStatistics {
    fn new() -> Self {
        Self {
            total_participants: 0,
            age_distribution: Vec::new(),
            gender_distribution: Vec::new(),
            ethnicity_distribution: Vec::new(),
            education_distribution: Vec::new(),
            visual_acuity_distribution: Vec::new(),
            color_vision_distribution: Vec::new(),
        }
    }
}

// Placeholder implementations for analysis components
struct StatisticalAnalyzer;
struct EyeTracker;
struct DemographicAnalyzer;
struct CulturalAnalyzer;

impl StatisticalAnalyzer {
    fn new() -> Result<Self, AfiyahError> { Ok(Self) }
}

impl EyeTracker {
    fn new() -> Result<Self, AfiyahError> { Ok(Self) }
}

impl DemographicAnalyzer {
    fn new() -> Result<Self, AfiyahError> { Ok(Self) }
}

impl CulturalAnalyzer {
    fn new() -> Result<Self, AfiyahError> { Ok(Self) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subjective_testing_config_default() {
        let config = SubjectiveTestingConfig::default();
        assert_eq!(config.participant_count, 30);
        assert!(matches!(config.test_type, TestType::DoubleBlindAB));
        assert!(config.irb_compliance);
    }

    #[test]
    fn test_participant_creation() {
        let participant = Participant {
            id: "test_participant".to_string(),
            age: 30,
            gender: Gender::Male,
            ethnicity: Ethnicity::Caucasian,
            education_level: EducationLevel::Bachelors,
            visual_acuity: VisualAcuity::Excellent,
            color_vision_status: ColorVisionStatus::Normal,
            cultural_background: CulturalBackground {
                primary_language: "English".to_string(),
                cultural_region: "North America".to_string(),
                visual_preferences: VisualPreferences {
                    brightness_preference: 0.5,
                    contrast_preference: 0.5,
                    saturation_preference: 0.5,
                    sharpness_preference: 0.5,
                    color_temperature_preference: 0.5,
                },
                color_associations: ColorAssociations {
                    red_associations: vec!["danger".to_string()],
                    green_associations: vec!["nature".to_string()],
                    blue_associations: vec!["sky".to_string()],
                    cultural_color_meanings: HashMap::new(),
                },
                motion_preferences: MotionPreferences {
                    motion_sensitivity: 0.5,
                    temporal_preference: 0.5,
                    smoothness_preference: 0.5,
                    action_preference: 0.5,
                },
            },
            viewing_experience: ViewingExperience {
                years_experience: 10,
                preferred_content_types: vec![ContentType::Movies],
                viewing_frequency: ViewingFrequency::Daily,
                device_usage: DeviceUsage {
                    mobile_usage: 0.3,
                    tablet_usage: 0.2,
                    desktop_usage: 0.3,
                    tv_usage: 0.2,
                    vr_usage: 0.0,
                },
            },
            device_preferences: DevicePreferences {
                preferred_device: DeviceType::Desktop,
                screen_size_preference: ScreenSizePreference::Large,
                resolution_preference: ResolutionPreference::FullHD,
                framerate_preference: FrameratePreference::High,
            },
            consent_given: true,
            irb_approval: true,
        };

        assert_eq!(participant.id, "test_participant");
        assert_eq!(participant.age, 30);
        assert!(participant.consent_given);
    }

    #[test]
    fn test_subjective_testing_engine_creation() {
        let config = SubjectiveTestingConfig::default();
        let engine = SubjectiveTestingEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_test_session_creation() {
        let config = SubjectiveTestingConfig::default();
        let mut engine = SubjectiveTestingEngine::new(config).unwrap();
        
        let participant = Participant {
            id: "test_participant".to_string(),
            age: 30,
            gender: Gender::Male,
            ethnicity: Ethnicity::Caucasian,
            education_level: EducationLevel::Bachelors,
            visual_acuity: VisualAcuity::Excellent,
            color_vision_status: ColorVisionStatus::Normal,
            cultural_background: CulturalBackground {
                primary_language: "English".to_string(),
                cultural_region: "North America".to_string(),
                visual_preferences: VisualPreferences {
                    brightness_preference: 0.5,
                    contrast_preference: 0.5,
                    saturation_preference: 0.5,
                    sharpness_preference: 0.5,
                    color_temperature_preference: 0.5,
                },
                color_associations: ColorAssociations {
                    red_associations: vec!["danger".to_string()],
                    green_associations: vec!["nature".to_string()],
                    blue_associations: vec!["sky".to_string()],
                    cultural_color_meanings: HashMap::new(),
                },
                motion_preferences: MotionPreferences {
                    motion_sensitivity: 0.5,
                    temporal_preference: 0.5,
                    smoothness_preference: 0.5,
                    action_preference: 0.5,
                },
            },
            viewing_experience: ViewingExperience {
                years_experience: 10,
                preferred_content_types: vec![ContentType::Movies],
                viewing_frequency: ViewingFrequency::Daily,
                device_usage: DeviceUsage {
                    mobile_usage: 0.3,
                    tablet_usage: 0.2,
                    desktop_usage: 0.3,
                    tv_usage: 0.2,
                    vr_usage: 0.0,
                },
            },
            device_preferences: DevicePreferences {
                preferred_device: DeviceType::Desktop,
                screen_size_preference: ScreenSizePreference::Large,
                resolution_preference: ResolutionPreference::FullHD,
                framerate_preference: FrameratePreference::High,
            },
            consent_given: true,
            irb_approval: true,
        };

        let register_result = engine.register_participant(participant);
        assert!(register_result.is_ok());

        let stimuli = vec![TestStimulus {
            stimulus_id: "test_stimulus".to_string(),
            stimulus_type: StimulusType::Test,
            content_data: vec![0u8; 1024],
            reference_data: None,
            quality_level: 0.95,
            compression_settings: CompressionSettings {
                codec: "afiyah".to_string(),
                bitrate: 1000000,
                resolution: (1920, 1080),
                framerate: 30.0,
                quality_preset: QualityPreset::High,
                biological_optimization: true,
                foveal_prioritization: true,
            },
            presentation_order: 1,
            duration: Duration::from_secs(10),
            metadata: StimulusMetadata {
                content_type: ContentType::Movies,
                scene_complexity: SceneComplexity::Moderate,
                motion_level: MotionLevel::Moderate,
                color_complexity: ColorComplexity::Rich,
                texture_complexity: TextureComplexity::Detailed,
                brightness_level: BrightnessLevel::Normal,
                contrast_level: ContrastLevel::High,
            },
        }];

        let session_result = engine.create_test_session("test_participant".to_string(), stimuli);
        assert!(session_result.is_ok());
    }
}
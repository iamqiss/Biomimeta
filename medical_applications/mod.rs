//! Medical Applications Module

use ndarray::Array2;
use crate::AfiyahError;

pub mod diagnostic_tools;
pub mod retinal_disease_modeling;
pub mod clinical_validation;

pub use diagnostic_tools::{DiagnosticTool, DiagnosticResult, DiseaseType};
pub use retinal_disease_modeling::{RetinalDiseaseModel, DiseaseProgression, TreatmentResponse};
pub use clinical_validation::{ClinicalValidator, ValidationResult, ClinicalMetrics};

/// Medical applications processor for diagnostic and therapeutic applications
pub struct MedicalProcessor {
    diagnostic_tool: DiagnosticTool,
    disease_model: RetinalDiseaseModel,
    clinical_validator: ClinicalValidator,
    medical_config: MedicalConfig,
}

/// Medical configuration
#[derive(Debug, Clone)]
pub struct MedicalConfig {
    pub enable_diagnostic_mode: bool,
    pub enable_disease_modeling: bool,
    pub enable_clinical_validation: bool,
    pub diagnostic_sensitivity: f64,
    pub diagnostic_specificity: f64,
    pub validation_threshold: f64,
}

impl Default for MedicalConfig {
    fn default() -> Self {
        Self {
            enable_diagnostic_mode: true,
            enable_disease_modeling: true,
            enable_clinical_validation: true,
            diagnostic_sensitivity: 0.95,
            diagnostic_specificity: 0.90,
            validation_threshold: 0.85,
        }
    }
}

impl MedicalProcessor {
    /// Creates a new medical processor
    pub fn new() -> Result<Self, AfiyahError> {
        let diagnostic_tool = DiagnosticTool::new()?;
        let disease_model = RetinalDiseaseModel::new()?;
        let clinical_validator = ClinicalValidator::new()?;
        let medical_config = MedicalConfig::default();

        Ok(Self {
            diagnostic_tool,
            disease_model,
            clinical_validator,
            medical_config,
        })
    }

    /// Processes medical imaging data for diagnostic purposes
    pub fn process_diagnostic(&mut self, input: &Array2<f64>) -> Result<DiagnosticResult, AfiyahError> {
        if !self.medical_config.enable_diagnostic_mode {
            return Err(AfiyahError::MedicalApplication { 
                message: "Diagnostic mode is disabled".to_string() 
            });
        }

        // Analyze retinal structure
        let retinal_analysis = self.analyze_retinal_structure(input)?;
        
        // Detect abnormalities
        let abnormalities = self.detect_abnormalities(&retinal_analysis)?;
        
        // Generate diagnostic result
        let diagnostic_result = DiagnosticResult {
            disease_type: self.classify_disease(&abnormalities)?,
            confidence: self.calculate_confidence(&abnormalities)?,
            severity: self.assess_severity(&abnormalities)?,
            recommendations: self.generate_recommendations(&abnormalities)?,
            metadata: self.extract_medical_metadata(input)?,
        };

        Ok(diagnostic_result)
    }

    /// Models disease progression
    pub fn model_disease_progression(&mut self, input: &Array2<f64>, time_steps: usize) -> Result<DiseaseProgression, AfiyahError> {
        if !self.medical_config.enable_disease_modeling {
            return Err(AfiyahError::MedicalApplication { 
                message: "Disease modeling is disabled".to_string() 
            });
        }

        let mut progression = DiseaseProgression::new();
        let mut current_state = input.clone();

        for step in 0..time_steps {
            // Model disease progression
            let next_state = self.disease_model.simulate_progression(&current_state, step as f64)?;
            
            // Calculate progression metrics
            let metrics = self.calculate_progression_metrics(&current_state, &next_state)?;
            
            // Add to progression
            progression.add_timepoint(step, next_state.clone(), metrics);
            current_state = next_state;
        }

        Ok(progression)
    }

    /// Validates clinical accuracy
    pub fn validate_clinical_accuracy(&mut self, input: &Array2<f64>, ground_truth: &Array2<f64>) -> Result<ValidationResult, AfiyahError> {
        if !self.medical_config.enable_clinical_validation {
            return Err(AfiyahError::MedicalApplication { 
                message: "Clinical validation is disabled".to_string() 
            });
        }

        // Calculate clinical metrics
        let metrics = self.clinical_validator.calculate_metrics(input, ground_truth)?;
        
        // Validate against thresholds
        let validation_result = ValidationResult {
            is_valid: metrics.accuracy >= self.medical_config.validation_threshold,
            metrics,
            recommendations: self.generate_validation_recommendations(&metrics)?,
        };

        Ok(validation_result)
    }

    fn analyze_retinal_structure(&self, input: &Array2<f64>) -> Result<RetinalAnalysis, AfiyahError> {
        // Analyze retinal layers
        let layers = self.analyze_retinal_layers(input)?;
        
        // Analyze blood vessels
        let vessels = self.analyze_blood_vessels(input)?;
        
        // Analyze optic nerve
        let optic_nerve = self.analyze_optic_nerve(input)?;

        Ok(RetinalAnalysis {
            layers,
            vessels,
            optic_nerve,
        })
    }

    fn analyze_retinal_layers(&self, input: &Array2<f64>) -> Result<Vec<RetinalLayer>, AfiyahError> {
        // Simulate retinal layer analysis
        let mut layers = Vec::new();
        
        // Ganglion cell layer
        layers.push(RetinalLayer {
            name: "Ganglion Cell Layer".to_string(),
            thickness: 0.1,
            density: 0.8,
            health_score: 0.9,
        });
        
        // Inner plexiform layer
        layers.push(RetinalLayer {
            name: "Inner Plexiform Layer".to_string(),
            thickness: 0.15,
            density: 0.7,
            health_score: 0.85,
        });
        
        // Inner nuclear layer
        layers.push(RetinalLayer {
            name: "Inner Nuclear Layer".to_string(),
            thickness: 0.2,
            density: 0.75,
            health_score: 0.88,
        });
        
        // Outer plexiform layer
        layers.push(RetinalLayer {
            name: "Outer Plexiform Layer".to_string(),
            thickness: 0.12,
            density: 0.65,
            health_score: 0.82,
        });
        
        // Outer nuclear layer
        layers.push(RetinalLayer {
            name: "Outer Nuclear Layer".to_string(),
            thickness: 0.18,
            density: 0.6,
            health_score: 0.8,
        });
        
        // Photoreceptor layer
        layers.push(RetinalLayer {
            name: "Photoreceptor Layer".to_string(),
            thickness: 0.25,
            density: 0.9,
            health_score: 0.95,
        });

        Ok(layers)
    }

    fn analyze_blood_vessels(&self, input: &Array2<f64>) -> Result<BloodVesselAnalysis, AfiyahError> {
        // Simulate blood vessel analysis
        Ok(BloodVesselAnalysis {
            vessel_density: 0.7,
            vessel_diameter: 0.15,
            tortuosity: 0.3,
            branching_pattern: 0.8,
            health_score: 0.85,
        })
    }

    fn analyze_optic_nerve(&self, input: &Array2<f64>) -> Result<OpticNerveAnalysis, AfiyahError> {
        // Simulate optic nerve analysis
        Ok(OpticNerveAnalysis {
            cup_to_disc_ratio: 0.3,
            rim_area: 0.8,
            nerve_fiber_layer_thickness: 0.12,
            health_score: 0.9,
        })
    }

    fn detect_abnormalities(&self, analysis: &RetinalAnalysis) -> Result<Vec<Abnormality>, AfiyahError> {
        let mut abnormalities = Vec::new();

        // Check for diabetic retinopathy
        if analysis.vessels.tortuosity > 0.5 {
            abnormalities.push(Abnormality {
                type_: "Diabetic Retinopathy".to_string(),
                severity: 0.7,
                location: "Blood Vessels".to_string(),
                confidence: 0.8,
            });
        }

        // Check for glaucoma
        if analysis.optic_nerve.cup_to_disc_ratio > 0.6 {
            abnormalities.push(Abnormality {
                type_: "Glaucoma".to_string(),
                severity: 0.6,
                location: "Optic Nerve".to_string(),
                confidence: 0.75,
            });
        }

        // Check for macular degeneration
        let avg_photoreceptor_health = analysis.layers
            .iter()
            .find(|l| l.name == "Photoreceptor Layer")
            .map(|l| l.health_score)
            .unwrap_or(1.0);
        
        if avg_photoreceptor_health < 0.7 {
            abnormalities.push(Abnormality {
                type_: "Macular Degeneration".to_string(),
                severity: 0.8,
                location: "Macula".to_string(),
                confidence: 0.85,
            });
        }

        Ok(abnormalities)
    }

    fn classify_disease(&self, abnormalities: &[Abnormality]) -> Result<DiseaseType, AfiyahError> {
        if abnormalities.is_empty() {
            return Ok(DiseaseType::Normal);
        }

        // Find the most severe abnormality
        let most_severe = abnormalities
            .iter()
            .max_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap())
            .unwrap();

        match most_severe.type_.as_str() {
            "Diabetic Retinopathy" => Ok(DiseaseType::DiabeticRetinopathy),
            "Glaucoma" => Ok(DiseaseType::Glaucoma),
            "Macular Degeneration" => Ok(DiseaseType::MacularDegeneration),
            _ => Ok(DiseaseType::Other),
        }
    }

    fn calculate_confidence(&self, abnormalities: &[Abnormality]) -> Result<f64, AfiyahError> {
        if abnormalities.is_empty() {
            return Ok(1.0);
        }

        let avg_confidence = abnormalities
            .iter()
            .map(|a| a.confidence)
            .sum::<f64>() / abnormalities.len() as f64;

        Ok(avg_confidence)
    }

    fn assess_severity(&self, abnormalities: &[Abnormality]) -> Result<f64, AfiyahError> {
        if abnormalities.is_empty() {
            return Ok(0.0);
        }

        let max_severity = abnormalities
            .iter()
            .map(|a| a.severity)
            .fold(0.0, f64::max);

        Ok(max_severity)
    }

    fn generate_recommendations(&self, abnormalities: &[Abnormality]) -> Result<Vec<String>, AfiyahError> {
        let mut recommendations = Vec::new();

        for abnormality in abnormalities {
            match abnormality.type_.as_str() {
                "Diabetic Retinopathy" => {
                    recommendations.push("Monitor blood sugar levels".to_string());
                    recommendations.push("Regular eye examinations".to_string());
                    recommendations.push("Consider laser treatment".to_string());
                },
                "Glaucoma" => {
                    recommendations.push("Regular intraocular pressure monitoring".to_string());
                    recommendations.push("Consider medication or surgery".to_string());
                    recommendations.push("Avoid activities that increase pressure".to_string());
                },
                "Macular Degeneration" => {
                    recommendations.push("Nutritional supplements (AREDS formula)".to_string());
                    recommendations.push("Regular monitoring".to_string());
                    recommendations.push("Consider anti-VEGF treatment".to_string());
                },
                _ => {
                    recommendations.push("Consult with ophthalmologist".to_string());
                }
            }
        }

        Ok(recommendations)
    }

    fn extract_medical_metadata(&self, input: &Array2<f64>) -> Result<MedicalMetadata, AfiyahError> {
        Ok(MedicalMetadata {
            image_resolution: input.dim(),
            acquisition_date: chrono::Utc::now(),
            patient_id: "ANONYMOUS".to_string(),
            study_id: "STUDY_001".to_string(),
            equipment: "Afiyah Medical Scanner".to_string(),
        })
    }

    fn calculate_progression_metrics(&self, current: &Array2<f64>, next: &Array2<f64>) -> Result<ProgressionMetrics, AfiyahError> {
        // Calculate progression metrics
        let change_rate = self.calculate_change_rate(current, next)?;
        let severity_change = self.calculate_severity_change(current, next)?;
        let area_affected = self.calculate_area_affected(next)?;

        Ok(ProgressionMetrics {
            change_rate,
            severity_change,
            area_affected,
            timepoint: chrono::Utc::now(),
        })
    }

    fn calculate_change_rate(&self, current: &Array2<f64>, next: &Array2<f64>) -> Result<f64, AfiyahError> {
        let diff = next - current;
        let change_rate = diff.mapv(|x| x.abs()).sum() / current.len() as f64;
        Ok(change_rate)
    }

    fn calculate_severity_change(&self, current: &Array2<f64>, next: &Array2<f64>) -> Result<f64, AfiyahError> {
        let current_severity = current.mean().unwrap_or(0.0);
        let next_severity = next.mean().unwrap_or(0.0);
        Ok(next_severity - current_severity)
    }

    fn calculate_area_affected(&self, input: &Array2<f64>) -> Result<f64, AfiyahError> {
        let threshold = 0.5;
        let affected_pixels = input.iter().filter(|&&x| x > threshold).count();
        let total_pixels = input.len();
        Ok(affected_pixels as f64 / total_pixels as f64)
    }

    fn generate_validation_recommendations(&self, metrics: &ClinicalMetrics) -> Result<Vec<String>, AfiyahError> {
        let mut recommendations = Vec::new();

        if metrics.accuracy < 0.9 {
            recommendations.push("Improve diagnostic accuracy".to_string());
        }

        if metrics.sensitivity < 0.95 {
            recommendations.push("Increase diagnostic sensitivity".to_string());
        }

        if metrics.specificity < 0.90 {
            recommendations.push("Improve diagnostic specificity".to_string());
        }

        if metrics.precision < 0.85 {
            recommendations.push("Enhance diagnostic precision".to_string());
        }

        Ok(recommendations)
    }

    /// Updates medical configuration
    pub fn update_config(&mut self, config: MedicalConfig) {
        self.medical_config = config;
    }

    /// Gets current medical configuration
    pub fn get_config(&self) -> &MedicalConfig {
        &self.medical_config
    }
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct RetinalAnalysis {
    pub layers: Vec<RetinalLayer>,
    pub vessels: BloodVesselAnalysis,
    pub optic_nerve: OpticNerveAnalysis,
}

#[derive(Debug, Clone)]
pub struct RetinalLayer {
    pub name: String,
    pub thickness: f64,
    pub density: f64,
    pub health_score: f64,
}

#[derive(Debug, Clone)]
pub struct BloodVesselAnalysis {
    pub vessel_density: f64,
    pub vessel_diameter: f64,
    pub tortuosity: f64,
    pub branching_pattern: f64,
    pub health_score: f64,
}

#[derive(Debug, Clone)]
pub struct OpticNerveAnalysis {
    pub cup_to_disc_ratio: f64,
    pub rim_area: f64,
    pub nerve_fiber_layer_thickness: f64,
    pub health_score: f64,
}

#[derive(Debug, Clone)]
pub struct Abnormality {
    pub type_: String,
    pub severity: f64,
    pub location: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MedicalMetadata {
    pub image_resolution: (usize, usize),
    pub acquisition_date: chrono::DateTime<chrono::Utc>,
    pub patient_id: String,
    pub study_id: String,
    pub equipment: String,
}

#[derive(Debug, Clone)]
pub struct ProgressionMetrics {
    pub change_rate: f64,
    pub severity_change: f64,
    pub area_affected: f64,
    pub timepoint: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medical_processor_creation() {
        let processor = MedicalProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_diagnostic_processing() {
        let mut processor = MedicalProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.process_diagnostic(&input);
        assert!(result.is_ok());
        
        let diagnostic_result = result.unwrap();
        assert!(diagnostic_result.confidence >= 0.0);
        assert!(diagnostic_result.confidence <= 1.0);
    }

    #[test]
    fn test_disease_progression_modeling() {
        let mut processor = MedicalProcessor::new().unwrap();
        let input = Array2::ones((16, 16));
        
        let result = processor.model_disease_progression(&input, 5);
        assert!(result.is_ok());
        
        let progression = result.unwrap();
        assert_eq!(progression.timepoints.len(), 5);
    }

    #[test]
    fn test_clinical_validation() {
        let mut processor = MedicalProcessor::new().unwrap();
        let input = Array2::ones((16, 16));
        let ground_truth = Array2::ones((16, 16));
        
        let result = processor.validate_clinical_accuracy(&input, &ground_truth);
        assert!(result.is_ok());
        
        let validation_result = result.unwrap();
        assert!(validation_result.is_valid);
    }

    #[test]
    fn test_configuration_update() {
        let mut processor = MedicalProcessor::new().unwrap();
        let config = MedicalConfig {
            enable_diagnostic_mode: false,
            enable_disease_modeling: true,
            enable_clinical_validation: true,
            diagnostic_sensitivity: 0.98,
            diagnostic_specificity: 0.95,
            validation_threshold: 0.90,
        };
        
        processor.update_config(config);
        assert!(!processor.get_config().enable_diagnostic_mode);
        assert_eq!(processor.get_config().diagnostic_sensitivity, 0.98);
    }
}
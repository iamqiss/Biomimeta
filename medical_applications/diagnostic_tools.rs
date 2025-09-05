//! Diagnostic Tools Module

use ndarray::Array2;
use crate::AfiyahError;

/// Disease types that can be diagnosed
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiseaseType {
    Normal,
    DiabeticRetinopathy,
    Glaucoma,
    MacularDegeneration,
    RetinalDetachment,
    RetinopathyOfPrematurity,
    Other,
}

/// Diagnostic result containing disease classification and confidence
#[derive(Debug, Clone)]
pub struct DiagnosticResult {
    pub disease_type: DiseaseType,
    pub confidence: f64,
    pub severity: f64,
    pub recommendations: Vec<String>,
    pub metadata: MedicalMetadata,
}

/// Medical metadata for diagnostic results
#[derive(Debug, Clone)]
pub struct MedicalMetadata {
    pub image_resolution: (usize, usize),
    pub acquisition_date: chrono::DateTime<chrono::Utc>,
    pub patient_id: String,
    pub study_id: String,
    pub equipment: String,
}

/// Diagnostic tool for analyzing retinal images
pub struct DiagnosticTool {
    disease_classifiers: Vec<DiseaseClassifier>,
    feature_extractors: Vec<FeatureExtractor>,
    diagnostic_config: DiagnosticConfig,
}

/// Disease classifier for specific conditions
#[derive(Debug, Clone)]
pub struct DiseaseClassifier {
    pub disease_type: DiseaseType,
    pub sensitivity: f64,
    pub specificity: f64,
    pub threshold: f64,
    pub features: Vec<String>,
}

/// Feature extractor for medical imaging
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    pub name: String,
    pub feature_type: FeatureType,
    pub parameters: Vec<f64>,
}

/// Types of features that can be extracted
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    Morphological,
    Texture,
    Vascular,
    Structural,
    Functional,
}

/// Diagnostic configuration
#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    pub enable_ai_classification: bool,
    pub enable_rule_based_classification: bool,
    pub confidence_threshold: f64,
    pub severity_threshold: f64,
    pub feature_weights: Vec<f64>,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            enable_ai_classification: true,
            enable_rule_based_classification: true,
            confidence_threshold: 0.8,
            severity_threshold: 0.5,
            feature_weights: vec![0.3, 0.3, 0.2, 0.2], // Equal weights for 4 features
        }
    }
}

impl DiagnosticTool {
    /// Creates a new diagnostic tool
    pub fn new() -> Result<Self, AfiyahError> {
        let disease_classifiers = Self::initialize_classifiers()?;
        let feature_extractors = Self::initialize_feature_extractors()?;
        let diagnostic_config = DiagnosticConfig::default();

        Ok(Self {
            disease_classifiers,
            feature_extractors,
            diagnostic_config,
        })
    }

    /// Analyzes retinal image for disease detection
    pub fn analyze_retinal_image(&mut self, input: &Array2<f64>) -> Result<DiagnosticResult, AfiyahError> {
        // Extract features
        let features = self.extract_features(input)?;
        
        // Classify diseases
        let classifications = self.classify_diseases(&features)?;
        
        // Determine primary diagnosis
        let primary_diagnosis = self.determine_primary_diagnosis(&classifications)?;
        
        // Calculate confidence and severity
        let confidence = self.calculate_confidence(&classifications, &primary_diagnosis)?;
        let severity = self.calculate_severity(&features, &primary_diagnosis)?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&primary_diagnosis, severity)?;
        
        // Extract metadata
        let metadata = self.extract_metadata(input)?;

        Ok(DiagnosticResult {
            disease_type: primary_diagnosis,
            confidence,
            severity,
            recommendations,
            metadata,
        })
    }

    fn initialize_classifiers() -> Result<Vec<DiseaseClassifier>, AfiyahError> {
        let mut classifiers = Vec::new();

        // Diabetic Retinopathy classifier
        classifiers.push(DiseaseClassifier {
            disease_type: DiseaseType::DiabeticRetinopathy,
            sensitivity: 0.95,
            specificity: 0.90,
            threshold: 0.7,
            features: vec!["vessel_tortuosity".to_string(), "microaneurysms".to_string(), "hemorrhages".to_string()],
        });

        // Glaucoma classifier
        classifiers.push(DiseaseClassifier {
            disease_type: DiseaseType::Glaucoma,
            sensitivity: 0.92,
            specificity: 0.88,
            threshold: 0.6,
            features: vec!["cup_to_disc_ratio".to_string(), "rim_area".to_string(), "nerve_fiber_layer".to_string()],
        });

        // Macular Degeneration classifier
        classifiers.push(DiseaseClassifier {
            disease_type: DiseaseType::MacularDegeneration,
            sensitivity: 0.90,
            specificity: 0.85,
            threshold: 0.65,
            features: vec!["drusen".to_string(), "geographic_atrophy".to_string(), "choroidal_neovascularization".to_string()],
        });

        // Retinal Detachment classifier
        classifiers.push(DiseaseClassifier {
            disease_type: DiseaseType::RetinalDetachment,
            sensitivity: 0.98,
            specificity: 0.95,
            threshold: 0.8,
            features: vec!["retinal_elevation".to_string(), "subretinal_fluid".to_string(), "retinal_breaks".to_string()],
        });

        // Retinopathy of Prematurity classifier
        classifiers.push(DiseaseClassifier {
            disease_type: DiseaseType::RetinopathyOfPrematurity,
            sensitivity: 0.88,
            specificity: 0.82,
            threshold: 0.7,
            features: vec!["vascular_tortuosity".to_string(), "plus_disease".to_string(), "ridge_formation".to_string()],
        });

        Ok(classifiers)
    }

    fn initialize_feature_extractors() -> Result<Vec<FeatureExtractor>, AfiyahError> {
        let mut extractors = Vec::new();

        // Morphological features
        extractors.push(FeatureExtractor {
            name: "vessel_tortuosity".to_string(),
            feature_type: FeatureType::Morphological,
            parameters: vec![0.5, 1.0, 2.0],
        });

        extractors.push(FeatureExtractor {
            name: "cup_to_disc_ratio".to_string(),
            feature_type: FeatureType::Morphological,
            parameters: vec![0.3, 0.6, 0.9],
        });

        // Texture features
        extractors.push(FeatureExtractor {
            name: "drusen_density".to_string(),
            feature_type: FeatureType::Texture,
            parameters: vec![0.1, 0.5, 1.0],
        });

        extractors.push(FeatureExtractor {
            name: "microaneurysms".to_string(),
            feature_type: FeatureType::Texture,
            parameters: vec![0.05, 0.1, 0.2],
        });

        // Vascular features
        extractors.push(FeatureExtractor {
            name: "vessel_diameter".to_string(),
            feature_type: FeatureType::Vascular,
            parameters: vec![0.1, 0.2, 0.3],
        });

        extractors.push(FeatureExtractor {
            name: "branching_pattern".to_string(),
            feature_type: FeatureType::Vascular,
            parameters: vec![0.5, 0.7, 0.9],
        });

        // Structural features
        extractors.push(FeatureExtractor {
            name: "retinal_thickness".to_string(),
            feature_type: FeatureType::Structural,
            parameters: vec![0.2, 0.3, 0.4],
        });

        extractors.push(FeatureExtractor {
            name: "nerve_fiber_layer".to_string(),
            feature_type: FeatureType::Structural,
            parameters: vec![0.1, 0.15, 0.2],
        });

        Ok(extractors)
    }

    fn extract_features(&self, input: &Array2<f64>) -> Result<Vec<Feature>, AfiyahError> {
        let mut features = Vec::new();

        for extractor in &self.feature_extractors {
            let feature_value = self.extract_single_feature(input, extractor)?;
            features.push(Feature {
                name: extractor.name.clone(),
                value: feature_value,
                confidence: 0.8, // Placeholder confidence
            });
        }

        Ok(features)
    }

    fn extract_single_feature(&self, input: &Array2<f64>, extractor: &FeatureExtractor) -> Result<f64, AfiyahError> {
        match extractor.feature_type {
            FeatureType::Morphological => {
                self.extract_morphological_feature(input, &extractor.name, &extractor.parameters)
            },
            FeatureType::Texture => {
                self.extract_texture_feature(input, &extractor.name, &extractor.parameters)
            },
            FeatureType::Vascular => {
                self.extract_vascular_feature(input, &extractor.name, &extractor.parameters)
            },
            FeatureType::Structural => {
                self.extract_structural_feature(input, &extractor.name, &extractor.parameters)
            },
            FeatureType::Functional => {
                self.extract_functional_feature(input, &extractor.name, &extractor.parameters)
            },
        }
    }

    fn extract_morphological_feature(&self, input: &Array2<f64>, name: &str, params: &[f64]) -> Result<f64, AfiyahError> {
        match name {
            "vessel_tortuosity" => {
                // Simulate vessel tortuosity calculation
                let mean_value = input.mean().unwrap_or(0.0);
                Ok(mean_value * params[0])
            },
            "cup_to_disc_ratio" => {
                // Simulate cup-to-disc ratio calculation
                let max_value = input.max().unwrap_or(0.0);
                Ok(max_value * params[1])
            },
            _ => Ok(0.0),
        }
    }

    fn extract_texture_feature(&self, input: &Array2<f64>, name: &str, params: &[f64]) -> Result<f64, AfiyahError> {
        match name {
            "drusen_density" => {
                // Simulate drusen density calculation
                let variance = input.var(0.0);
                Ok(variance * params[0])
            },
            "microaneurysms" => {
                // Simulate microaneurysm detection
                let threshold = params[1];
                let count = input.iter().filter(|&&x| x > threshold).count();
                Ok(count as f64 / input.len() as f64)
            },
            _ => Ok(0.0),
        }
    }

    fn extract_vascular_feature(&self, input: &Array2<f64>, name: &str, params: &[f64]) -> Result<f64, AfiyahError> {
        match name {
            "vessel_diameter" => {
                // Simulate vessel diameter calculation
                let mean_value = input.mean().unwrap_or(0.0);
                Ok(mean_value * params[0])
            },
            "branching_pattern" => {
                // Simulate branching pattern analysis
                let std_dev = input.std(0.0);
                Ok(std_dev * params[1])
            },
            _ => Ok(0.0),
        }
    }

    fn extract_structural_feature(&self, input: &Array2<f64>, name: &str, params: &[f64]) -> Result<f64, AfiyahError> {
        match name {
            "retinal_thickness" => {
                // Simulate retinal thickness calculation
                let mean_value = input.mean().unwrap_or(0.0);
                Ok(mean_value * params[0])
            },
            "nerve_fiber_layer" => {
                // Simulate nerve fiber layer analysis
                let max_value = input.max().unwrap_or(0.0);
                Ok(max_value * params[1])
            },
            _ => Ok(0.0),
        }
    }

    fn extract_functional_feature(&self, input: &Array2<f64>, name: &str, params: &[f64]) -> Result<f64, AfiyahError> {
        // Simulate functional feature extraction
        let mean_value = input.mean().unwrap_or(0.0);
        Ok(mean_value * params[0])
    }

    fn classify_diseases(&self, features: &[Feature]) -> Result<Vec<DiseaseClassification>, AfiyahError> {
        let mut classifications = Vec::new();

        for classifier in &self.disease_classifiers {
            let score = self.calculate_disease_score(features, classifier)?;
            let confidence = if score >= classifier.threshold {
                (score - classifier.threshold) / (1.0 - classifier.threshold)
            } else {
                0.0
            };

            classifications.push(DiseaseClassification {
                disease_type: classifier.disease_type,
                score,
                confidence,
                sensitivity: classifier.sensitivity,
                specificity: classifier.specificity,
            });
        }

        Ok(classifications)
    }

    fn calculate_disease_score(&self, features: &[Feature], classifier: &DiseaseClassifier) -> Result<f64, AfiyahError> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, feature_name) in classifier.features.iter().enumerate() {
            if let Some(feature) = features.iter().find(|f| f.name == *feature_name) {
                let weight = if i < self.diagnostic_config.feature_weights.len() {
                    self.diagnostic_config.feature_weights[i]
                } else {
                    0.25 // Default weight
                };
                weighted_sum += feature.value * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            Ok(0.0)
        }
    }

    fn determine_primary_diagnosis(&self, classifications: &[DiseaseClassification]) -> Result<DiseaseType, AfiyahError> {
        if classifications.is_empty() {
            return Ok(DiseaseType::Normal);
        }

        let best_classification = classifications
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        if best_classification.confidence >= self.diagnostic_config.confidence_threshold {
            Ok(best_classification.disease_type)
        } else {
            Ok(DiseaseType::Normal)
        }
    }

    fn calculate_confidence(&self, classifications: &[DiseaseClassification], primary_diagnosis: &DiseaseType) -> Result<f64, AfiyahError> {
        if let Some(classification) = classifications.iter().find(|c| c.disease_type == *primary_diagnosis) {
            Ok(classification.confidence)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_severity(&self, features: &[Feature], primary_diagnosis: &DiseaseType) -> Result<f64, AfiyahError> {
        match primary_diagnosis {
            DiseaseType::Normal => Ok(0.0),
            DiseaseType::DiabeticRetinopathy => {
                self.calculate_diabetic_retinopathy_severity(features)
            },
            DiseaseType::Glaucoma => {
                self.calculate_glaucoma_severity(features)
            },
            DiseaseType::MacularDegeneration => {
                self.calculate_macular_degeneration_severity(features)
            },
            DiseaseType::RetinalDetachment => {
                self.calculate_retinal_detachment_severity(features)
            },
            DiseaseType::RetinopathyOfPrematurity => {
                self.calculate_rop_severity(features)
            },
            DiseaseType::Other => Ok(0.5),
        }
    }

    fn calculate_diabetic_retinopathy_severity(&self, features: &[Feature]) -> Result<f64, AfiyahError> {
        let mut severity = 0.0;
        let mut count = 0;

        for feature in features {
            if feature.name == "vessel_tortuosity" || feature.name == "microaneurysms" {
                severity += feature.value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(severity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_glaucoma_severity(&self, features: &[Feature]) -> Result<f64, AfiyahError> {
        let mut severity = 0.0;
        let mut count = 0;

        for feature in features {
            if feature.name == "cup_to_disc_ratio" || feature.name == "nerve_fiber_layer" {
                severity += feature.value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(severity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_macular_degeneration_severity(&self, features: &[Feature]) -> Result<f64, AfiyahError> {
        let mut severity = 0.0;
        let mut count = 0;

        for feature in features {
            if feature.name == "drusen_density" || feature.name == "geographic_atrophy" {
                severity += feature.value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(severity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_retinal_detachment_severity(&self, features: &[Feature]) -> Result<f64, AfiyahError> {
        let mut severity = 0.0;
        let mut count = 0;

        for feature in features {
            if feature.name == "retinal_elevation" || feature.name == "subretinal_fluid" {
                severity += feature.value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(severity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_rop_severity(&self, features: &[Feature]) -> Result<f64, AfiyahError> {
        let mut severity = 0.0;
        let mut count = 0;

        for feature in features {
            if feature.name == "vascular_tortuosity" || feature.name == "plus_disease" {
                severity += feature.value;
                count += 1;
            }
        }

        if count > 0 {
            Ok(severity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn generate_recommendations(&self, diagnosis: &DiseaseType, severity: f64) -> Result<Vec<String>, AfiyahError> {
        let mut recommendations = Vec::new();

        match diagnosis {
            DiseaseType::Normal => {
                recommendations.push("Continue regular eye examinations".to_string());
                recommendations.push("Maintain healthy lifestyle".to_string());
            },
            DiseaseType::DiabeticRetinopathy => {
                recommendations.push("Monitor blood sugar levels closely".to_string());
                recommendations.push("Regular ophthalmologic follow-up".to_string());
                if severity > 0.7 {
                    recommendations.push("Consider laser treatment".to_string());
                    recommendations.push("Anti-VEGF therapy may be indicated".to_string());
                }
            },
            DiseaseType::Glaucoma => {
                recommendations.push("Regular intraocular pressure monitoring".to_string());
                recommendations.push("Consider medication or surgery".to_string());
                if severity > 0.8 {
                    recommendations.push("Urgent surgical intervention may be needed".to_string());
                }
            },
            DiseaseType::MacularDegeneration => {
                recommendations.push("Nutritional supplements (AREDS formula)".to_string());
                recommendations.push("Regular monitoring".to_string());
                if severity > 0.6 {
                    recommendations.push("Consider anti-VEGF treatment".to_string());
                }
            },
            DiseaseType::RetinalDetachment => {
                recommendations.push("Urgent surgical intervention required".to_string());
                recommendations.push("Avoid strenuous activities".to_string());
            },
            DiseaseType::RetinopathyOfPrematurity => {
                recommendations.push("Specialized pediatric ophthalmologic care".to_string());
                recommendations.push("Regular monitoring of progression".to_string());
                if severity > 0.7 {
                    recommendations.push("Consider laser treatment or anti-VEGF".to_string());
                }
            },
            DiseaseType::Other => {
                recommendations.push("Consult with ophthalmologist".to_string());
                recommendations.push("Further diagnostic testing may be needed".to_string());
            },
        }

        Ok(recommendations)
    }

    fn extract_metadata(&self, input: &Array2<f64>) -> Result<MedicalMetadata, AfiyahError> {
        Ok(MedicalMetadata {
            image_resolution: input.dim(),
            acquisition_date: chrono::Utc::now(),
            patient_id: "ANONYMOUS".to_string(),
            study_id: "STUDY_001".to_string(),
            equipment: "Afiyah Medical Scanner".to_string(),
        })
    }

    /// Updates diagnostic configuration
    pub fn update_config(&mut self, config: DiagnosticConfig) {
        self.diagnostic_config = config;
    }

    /// Gets current diagnostic configuration
    pub fn get_config(&self) -> &DiagnosticConfig {
        &self.diagnostic_config
    }
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct Feature {
    pub name: String,
    pub value: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct DiseaseClassification {
    pub disease_type: DiseaseType,
    pub score: f64,
    pub confidence: f64,
    pub sensitivity: f64,
    pub specificity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_tool_creation() {
        let tool = DiagnosticTool::new();
        assert!(tool.is_ok());
    }

    #[test]
    fn test_retinal_image_analysis() {
        let mut tool = DiagnosticTool::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = tool.analyze_retinal_image(&input);
        assert!(result.is_ok());
        
        let diagnostic_result = result.unwrap();
        assert!(diagnostic_result.confidence >= 0.0);
        assert!(diagnostic_result.confidence <= 1.0);
        assert!(diagnostic_result.severity >= 0.0);
        assert!(diagnostic_result.severity <= 1.0);
    }

    #[test]
    fn test_disease_classification() {
        let tool = DiagnosticTool::new().unwrap();
        let features = vec![
            Feature { name: "vessel_tortuosity".to_string(), value: 0.8, confidence: 0.9 },
            Feature { name: "microaneurysms".to_string(), value: 0.7, confidence: 0.8 },
        ];
        
        let result = tool.classify_diseases(&features);
        assert!(result.is_ok());
        
        let classifications = result.unwrap();
        assert!(!classifications.is_empty());
    }

    #[test]
    fn test_feature_extraction() {
        let tool = DiagnosticTool::new().unwrap();
        let input = Array2::ones((16, 16));
        
        let result = tool.extract_features(&input);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_configuration_update() {
        let mut tool = DiagnosticTool::new().unwrap();
        let config = DiagnosticConfig {
            enable_ai_classification: false,
            enable_rule_based_classification: true,
            confidence_threshold: 0.9,
            severity_threshold: 0.6,
            feature_weights: vec![0.4, 0.3, 0.2, 0.1],
        };
        
        tool.update_config(config);
        assert!(!tool.get_config().enable_ai_classification);
        assert_eq!(tool.get_config().confidence_threshold, 0.9);
    }
}
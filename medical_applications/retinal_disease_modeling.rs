//! Retinal Disease Modeling Module

use ndarray::Array2;
use crate::AfiyahError;

/// Retinal disease model for simulating disease progression
pub struct RetinalDiseaseModel {
    disease_models: Vec<DiseaseModel>,
    progression_config: ProgressionConfig,
}

/// Individual disease model
#[derive(Debug, Clone)]
pub struct DiseaseModel {
    pub disease_type: DiseaseType,
    pub progression_rate: f64,
    pub severity_factors: Vec<f64>,
    pub treatment_effects: Vec<TreatmentEffect>,
    pub risk_factors: Vec<RiskFactor>,
}

/// Disease progression over time
#[derive(Debug, Clone)]
pub struct DiseaseProgression {
    pub timepoints: Vec<Timepoint>,
    pub final_severity: f64,
    pub progression_rate: f64,
    pub treatment_response: Option<TreatmentResponse>,
}

/// Individual timepoint in disease progression
#[derive(Debug, Clone)]
pub struct Timepoint {
    pub time: usize,
    pub state: Array2<f64>,
    pub metrics: ProgressionMetrics,
}

/// Progression metrics for a timepoint
#[derive(Debug, Clone)]
pub struct ProgressionMetrics {
    pub change_rate: f64,
    pub severity_change: f64,
    pub area_affected: f64,
    pub timepoint: chrono::DateTime<chrono::Utc>,
}

/// Treatment response modeling
#[derive(Debug, Clone)]
pub struct TreatmentResponse {
    pub treatment_type: TreatmentType,
    pub effectiveness: f64,
    pub side_effects: Vec<SideEffect>,
    pub duration: f64,
    pub cost: f64,
}

/// Treatment types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreatmentType {
    Medication,
    Laser,
    Surgery,
    AntiVEGF,
    Nutritional,
    Lifestyle,
}

/// Side effects of treatment
#[derive(Debug, Clone)]
pub struct SideEffect {
    pub name: String,
    pub severity: f64,
    pub probability: f64,
    pub duration: f64,
}

/// Risk factors for disease progression
#[derive(Debug, Clone)]
pub struct RiskFactor {
    pub name: String,
    pub weight: f64,
    pub current_value: f64,
    pub target_value: f64,
}

/// Treatment effects
#[derive(Debug, Clone)]
pub struct TreatmentEffect {
    pub treatment_type: TreatmentType,
    pub effectiveness: f64,
    pub duration: f64,
    pub cost: f64,
}

/// Disease types that can be modeled
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiseaseType {
    DiabeticRetinopathy,
    Glaucoma,
    MacularDegeneration,
    RetinalDetachment,
    RetinopathyOfPrematurity,
}

/// Progression configuration
#[derive(Debug, Clone)]
pub struct ProgressionConfig {
    pub time_step: f64,
    pub max_time: f64,
    pub enable_treatment: bool,
    pub enable_risk_factors: bool,
    pub enable_side_effects: bool,
}

impl Default for ProgressionConfig {
    fn default() -> Self {
        Self {
            time_step: 1.0, // 1 month
            max_time: 60.0, // 5 years
            enable_treatment: true,
            enable_risk_factors: true,
            enable_side_effects: true,
        }
    }
}

impl RetinalDiseaseModel {
    /// Creates a new retinal disease model
    pub fn new() -> Result<Self, AfiyahError> {
        let disease_models = Self::initialize_disease_models()?;
        let progression_config = ProgressionConfig::default();

        Ok(Self {
            disease_models,
            progression_config,
        })
    }

    /// Simulates disease progression
    pub fn simulate_progression(&self, initial_state: &Array2<f64>, time: f64) -> Result<Array2<f64>, AfiyahError> {
        let mut current_state = initial_state.clone();
        let mut current_time = 0.0;

        while current_time < time {
            // Apply disease progression
            current_state = self.apply_disease_progression(&current_state, current_time)?;
            
            // Apply treatments if enabled
            if self.progression_config.enable_treatment {
                current_state = self.apply_treatments(&current_state, current_time)?;
            }
            
            // Apply risk factors if enabled
            if self.progression_config.enable_risk_factors {
                current_state = self.apply_risk_factors(&current_state, current_time)?;
            }
            
            current_time += self.progression_config.time_step;
        }

        Ok(current_state)
    }

    /// Models treatment response
    pub fn model_treatment_response(&self, disease_type: DiseaseType, treatment_type: TreatmentType) -> Result<TreatmentResponse, AfiyahError> {
        let disease_model = self.disease_models
            .iter()
            .find(|m| m.disease_type == disease_type)
            .ok_or_else(|| AfiyahError::MedicalApplication {
                message: format!("Disease model not found for {:?}", disease_type)
            })?;

        let treatment_effect = disease_model.treatment_effects
            .iter()
            .find(|e| e.treatment_type == treatment_type)
            .ok_or_else(|| AfiyahError::MedicalApplication {
                message: format!("Treatment effect not found for {:?}", treatment_type)
            })?;

        let side_effects = if self.progression_config.enable_side_effects {
            self.generate_side_effects(treatment_type)?
        } else {
            Vec::new()
        };

        Ok(TreatmentResponse {
            treatment_type,
            effectiveness: treatment_effect.effectiveness,
            side_effects,
            duration: treatment_effect.duration,
            cost: treatment_effect.cost,
        })
    }

    /// Calculates disease risk score
    pub fn calculate_risk_score(&self, disease_type: DiseaseType, risk_factors: &[RiskFactor]) -> Result<f64, AfiyahError> {
        let disease_model = self.disease_models
            .iter()
            .find(|m| m.disease_type == disease_type)
            .ok_or_else(|| AfiyahError::MedicalApplication {
                message: format!("Disease model not found for {:?}", disease_type)
            })?;

        let mut risk_score = 0.0;
        let mut total_weight = 0.0;

        for risk_factor in risk_factors {
            if let Some(model_risk) = disease_model.risk_factors
                .iter()
                .find(|r| r.name == risk_factor.name) {
                let risk_contribution = (risk_factor.current_value - model_risk.target_value).abs() * model_risk.weight;
                risk_score += risk_contribution;
                total_weight += model_risk.weight;
            }
        }

        if total_weight > 0.0 {
            Ok(risk_score / total_weight)
        } else {
            Ok(0.0)
        }
    }

    fn initialize_disease_models() -> Result<Vec<DiseaseModel>, AfiyahError> {
        let mut models = Vec::new();

        // Diabetic Retinopathy model
        models.push(DiseaseModel {
            disease_type: DiseaseType::DiabeticRetinopathy,
            progression_rate: 0.1,
            severity_factors: vec![0.3, 0.4, 0.3], // Blood sugar, duration, control
            treatment_effects: vec![
                TreatmentEffect {
                    treatment_type: TreatmentType::Laser,
                    effectiveness: 0.8,
                    duration: 12.0,
                    cost: 5000.0,
                },
                TreatmentEffect {
                    treatment_type: TreatmentType::AntiVEGF,
                    effectiveness: 0.9,
                    duration: 6.0,
                    cost: 8000.0,
                },
            ],
            risk_factors: vec![
                RiskFactor {
                    name: "blood_sugar_control".to_string(),
                    weight: 0.4,
                    current_value: 0.5,
                    target_value: 0.8,
                },
                RiskFactor {
                    name: "diabetes_duration".to_string(),
                    weight: 0.3,
                    current_value: 0.6,
                    target_value: 0.0,
                },
                RiskFactor {
                    name: "blood_pressure".to_string(),
                    weight: 0.3,
                    current_value: 0.4,
                    target_value: 0.8,
                },
            ],
        });

        // Glaucoma model
        models.push(DiseaseModel {
            disease_type: DiseaseType::Glaucoma,
            progression_rate: 0.05,
            severity_factors: vec![0.5, 0.3, 0.2], // IOP, age, family history
            treatment_effects: vec![
                TreatmentEffect {
                    treatment_type: TreatmentType::Medication,
                    effectiveness: 0.7,
                    duration: 24.0,
                    cost: 2000.0,
                },
                TreatmentEffect {
                    treatment_type: TreatmentType::Surgery,
                    effectiveness: 0.9,
                    duration: 36.0,
                    cost: 15000.0,
                },
            ],
            risk_factors: vec![
                RiskFactor {
                    name: "intraocular_pressure".to_string(),
                    weight: 0.5,
                    current_value: 0.3,
                    target_value: 0.8,
                },
                RiskFactor {
                    name: "age".to_string(),
                    weight: 0.3,
                    current_value: 0.6,
                    target_value: 0.0,
                },
                RiskFactor {
                    name: "family_history".to_string(),
                    weight: 0.2,
                    current_value: 0.4,
                    target_value: 0.0,
                },
            ],
        });

        // Macular Degeneration model
        models.push(DiseaseModel {
            disease_type: DiseaseType::MacularDegeneration,
            progression_rate: 0.08,
            severity_factors: vec![0.4, 0.3, 0.3], // Age, genetics, lifestyle
            treatment_effects: vec![
                TreatmentEffect {
                    treatment_type: TreatmentType::AntiVEGF,
                    effectiveness: 0.85,
                    duration: 12.0,
                    cost: 10000.0,
                },
                TreatmentEffect {
                    treatment_type: TreatmentType::Nutritional,
                    effectiveness: 0.3,
                    duration: 60.0,
                    cost: 500.0,
                },
            ],
            risk_factors: vec![
                RiskFactor {
                    name: "age".to_string(),
                    weight: 0.4,
                    current_value: 0.7,
                    target_value: 0.0,
                },
                RiskFactor {
                    name: "smoking".to_string(),
                    weight: 0.3,
                    current_value: 0.2,
                    target_value: 0.0,
                },
                RiskFactor {
                    name: "nutrition".to_string(),
                    weight: 0.3,
                    current_value: 0.6,
                    target_value: 0.9,
                },
            ],
        });

        // Retinal Detachment model
        models.push(DiseaseModel {
            disease_type: DiseaseType::RetinalDetachment,
            progression_rate: 0.2,
            severity_factors: vec![0.6, 0.4], // Trauma, myopia
            treatment_effects: vec![
                TreatmentEffect {
                    treatment_type: TreatmentType::Surgery,
                    effectiveness: 0.95,
                    duration: 6.0,
                    cost: 20000.0,
                },
            ],
            risk_factors: vec![
                RiskFactor {
                    name: "myopia".to_string(),
                    weight: 0.6,
                    current_value: 0.8,
                    target_value: 0.0,
                },
                RiskFactor {
                    name: "trauma_history".to_string(),
                    weight: 0.4,
                    current_value: 0.3,
                    target_value: 0.0,
                },
            ],
        });

        // Retinopathy of Prematurity model
        models.push(DiseaseModel {
            disease_type: DiseaseType::RetinopathyOfPrematurity,
            progression_rate: 0.15,
            severity_factors: vec![0.5, 0.3, 0.2], // Gestational age, birth weight, oxygen
            treatment_effects: vec![
                TreatmentEffect {
                    treatment_type: TreatmentType::Laser,
                    effectiveness: 0.8,
                    duration: 3.0,
                    cost: 3000.0,
                },
                TreatmentEffect {
                    treatment_type: TreatmentType::AntiVEGF,
                    effectiveness: 0.85,
                    duration: 2.0,
                    cost: 5000.0,
                },
            ],
            risk_factors: vec![
                RiskFactor {
                    name: "gestational_age".to_string(),
                    weight: 0.5,
                    current_value: 0.3,
                    target_value: 0.9,
                },
                RiskFactor {
                    name: "birth_weight".to_string(),
                    weight: 0.3,
                    current_value: 0.4,
                    target_value: 0.8,
                },
                RiskFactor {
                    name: "oxygen_exposure".to_string(),
                    weight: 0.2,
                    current_value: 0.6,
                    target_value: 0.0,
                },
            ],
        });

        Ok(models)
    }

    fn apply_disease_progression(&self, state: &Array2<f64>, time: f64) -> Result<Array2<f64>, AfiyahError> {
        let mut new_state = state.clone();
        let progression_factor = 1.0 + (time * 0.01); // Simple progression model

        // Apply progression to each element
        for i in 0..new_state.nrows() {
            for j in 0..new_state.ncols() {
                let current_value = new_state[[i, j]];
                let progressed_value = current_value * progression_factor;
                new_state[[i, j]] = progressed_value.min(1.0); // Cap at 1.0
            }
        }

        Ok(new_state)
    }

    fn apply_treatments(&self, state: &Array2<f64>, time: f64) -> Result<Array2<f64>, AfiyahError> {
        let mut new_state = state.clone();
        
        // Simulate treatment effects
        let treatment_factor = 0.95; // 5% improvement per time step
        
        for i in 0..new_state.nrows() {
            for j in 0..new_state.ncols() {
                let current_value = new_state[[i, j]];
                let treated_value = current_value * treatment_factor;
                new_state[[i, j]] = treated_value.max(0.0); // Ensure non-negative
            }
        }

        Ok(new_state)
    }

    fn apply_risk_factors(&self, state: &Array2<f64>, time: f64) -> Result<Array2<f64>, AfiyahError> {
        let mut new_state = state.clone();
        
        // Simulate risk factor effects
        let risk_factor = 1.02; // 2% increase per time step
        
        for i in 0..new_state.nrows() {
            for j in 0..new_state.ncols() {
                let current_value = new_state[[i, j]];
                let risk_adjusted_value = current_value * risk_factor;
                new_state[[i, j]] = risk_adjusted_value.min(1.0); // Cap at 1.0
            }
        }

        Ok(new_state)
    }

    fn generate_side_effects(&self, treatment_type: TreatmentType) -> Result<Vec<SideEffect>, AfiyahError> {
        let mut side_effects = Vec::new();

        match treatment_type {
            TreatmentType::Laser => {
                side_effects.push(SideEffect {
                    name: "Temporary vision loss".to_string(),
                    severity: 0.3,
                    probability: 0.2,
                    duration: 1.0,
                });
                side_effects.push(SideEffect {
                    name: "Eye pain".to_string(),
                    severity: 0.4,
                    probability: 0.3,
                    duration: 0.5,
                });
            },
            TreatmentType::AntiVEGF => {
                side_effects.push(SideEffect {
                    name: "Injection site pain".to_string(),
                    severity: 0.2,
                    probability: 0.4,
                    duration: 0.1,
                });
                side_effects.push(SideEffect {
                    name: "Increased eye pressure".to_string(),
                    severity: 0.3,
                    probability: 0.1,
                    duration: 1.0,
                });
            },
            TreatmentType::Surgery => {
                side_effects.push(SideEffect {
                    name: "Post-operative pain".to_string(),
                    severity: 0.5,
                    probability: 0.6,
                    duration: 2.0,
                });
                side_effects.push(SideEffect {
                    name: "Infection risk".to_string(),
                    severity: 0.7,
                    probability: 0.05,
                    duration: 7.0,
                });
            },
            TreatmentType::Medication => {
                side_effects.push(SideEffect {
                    name: "Dry eyes".to_string(),
                    severity: 0.3,
                    probability: 0.3,
                    duration: 12.0,
                });
                side_effects.push(SideEffect {
                    name: "Allergic reaction".to_string(),
                    severity: 0.6,
                    probability: 0.02,
                    duration: 1.0,
                });
            },
            _ => {
                // Minimal side effects for other treatments
                side_effects.push(SideEffect {
                    name: "Mild discomfort".to_string(),
                    severity: 0.1,
                    probability: 0.1,
                    duration: 0.5,
                });
            }
        }

        Ok(side_effects)
    }

    /// Updates progression configuration
    pub fn update_config(&mut self, config: ProgressionConfig) {
        self.progression_config = config;
    }

    /// Gets current progression configuration
    pub fn get_config(&self) -> &ProgressionConfig {
        &self.progression_config
    }
}

impl DiseaseProgression {
    /// Creates a new disease progression
    pub fn new() -> Self {
        Self {
            timepoints: Vec::new(),
            final_severity: 0.0,
            progression_rate: 0.0,
            treatment_response: None,
        }
    }

    /// Adds a timepoint to the progression
    pub fn add_timepoint(&mut self, time: usize, state: Array2<f64>, metrics: ProgressionMetrics) {
        self.timepoints.push(Timepoint {
            time,
            state,
            metrics,
        });
    }

    /// Calculates final severity
    pub fn calculate_final_severity(&mut self) {
        if let Some(last_timepoint) = self.timepoints.last() {
            self.final_severity = last_timepoint.metrics.severity_change;
        }
    }

    /// Calculates progression rate
    pub fn calculate_progression_rate(&mut self) {
        if self.timepoints.len() >= 2 {
            let first = &self.timepoints[0];
            let last = &self.timepoints[self.timepoints.len() - 1];
            
            let time_diff = last.time - first.time;
            if time_diff > 0 {
                self.progression_rate = (last.metrics.severity_change - first.metrics.severity_change) / time_diff as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disease_model_creation() {
        let model = RetinalDiseaseModel::new();
        assert!(model.is_ok());
    }

    #[test]
    fn test_disease_progression_simulation() {
        let model = RetinalDiseaseModel::new().unwrap();
        let initial_state = Array2::ones((16, 16));
        
        let result = model.simulate_progression(&initial_state, 12.0);
        assert!(result.is_ok());
        
        let final_state = result.unwrap();
        assert_eq!(final_state.dim(), (16, 16));
    }

    #[test]
    fn test_treatment_response_modeling() {
        let model = RetinalDiseaseModel::new().unwrap();
        
        let result = model.model_treatment_response(DiseaseType::DiabeticRetinopathy, TreatmentType::Laser);
        assert!(result.is_ok());
        
        let treatment_response = result.unwrap();
        assert_eq!(treatment_response.treatment_type, TreatmentType::Laser);
        assert!(treatment_response.effectiveness > 0.0);
    }

    #[test]
    fn test_risk_score_calculation() {
        let model = RetinalDiseaseModel::new().unwrap();
        let risk_factors = vec![
            RiskFactor {
                name: "blood_sugar_control".to_string(),
                weight: 0.4,
                current_value: 0.5,
                target_value: 0.8,
            },
        ];
        
        let result = model.calculate_risk_score(DiseaseType::DiabeticRetinopathy, &risk_factors);
        assert!(result.is_ok());
        
        let risk_score = result.unwrap();
        assert!(risk_score >= 0.0);
    }

    #[test]
    fn test_disease_progression_creation() {
        let mut progression = DiseaseProgression::new();
        let state = Array2::ones((8, 8));
        let metrics = ProgressionMetrics {
            change_rate: 0.1,
            severity_change: 0.2,
            area_affected: 0.3,
            timepoint: chrono::Utc::now(),
        };
        
        progression.add_timepoint(0, state, metrics);
        assert_eq!(progression.timepoints.len(), 1);
    }

    #[test]
    fn test_configuration_update() {
        let mut model = RetinalDiseaseModel::new().unwrap();
        let config = ProgressionConfig {
            time_step: 2.0,
            max_time: 120.0,
            enable_treatment: false,
            enable_risk_factors: true,
            enable_side_effects: false,
        };
        
        model.update_config(config);
        assert_eq!(model.get_config().time_step, 2.0);
        assert!(!model.get_config().enable_treatment);
    }
}
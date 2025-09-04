//! Rhodopsin Cascade Implementation
//! 
//! This module implements the rhodopsin phototransduction cascade based on
//! biological research. The cascade amplifies single photon events through
//! a series of biochemical reactions involving transducin, phosphodiesterase,
//! and cGMP.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;

/// Rhodopsin cascade for signal amplification
pub struct RhodopsinCascade {
    transducin_activation: TransducinActivation,
    phosphodiesterase_activation: PhosphodiesteraseActivation,
    cgmp_hydrolysis: CgmpHydrolysis,
    channel_closure: ChannelClosure,
    amplification_factor: f64,
}

impl RhodopsinCascade {
    /// Creates new rhodopsin cascade with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            transducin_activation: TransducinActivation::new()?,
            phosphodiesterase_activation: PhosphodiesteraseActivation::new()?,
            cgmp_hydrolysis: CgmpHydrolysis::new()?,
            channel_closure: ChannelClosure::new()?,
            amplification_factor: 1e6, // Million-fold amplification
        })
    }

    /// Amplifies opsin response through the phototransduction cascade
    pub fn amplify(&self, opsin_response: &super::opsin_response::OpsinResponse) -> Result<super::AmplifiedResponse, AfiyahError> {
        // Stage 1: Transducin activation
        let transducin_response = self.transducin_activation.activate(&opsin_response.rhodopsin_response)?;
        
        // Stage 2: Phosphodiesterase activation
        let pde_response = self.phosphodiesterase_activation.activate(&transducin_response)?;
        
        // Stage 3: cGMP hydrolysis
        let cgmp_response = self.cgmp_hydrolysis.hydrolyze(&pde_response)?;
        
        // Stage 4: Channel closure
        let channel_response = self.channel_closure.close(&cgmp_response)?;
        
        // Apply amplification factor
        let amplified_rhodopsin: Vec<f64> = channel_response.iter()
            .map(|&x| (x * self.amplification_factor).min(1.0))
            .collect();
        
        // For cone responses, use lower amplification
        let cone_amplification = self.amplification_factor * 0.1;
        let amplified_cone: Vec<f64> = opsin_response.cone_response.iter()
            .map(|&x| (x * cone_amplification).min(1.0))
            .collect();
        
        // Generate temporal response
        let temporal_response = self.generate_temporal_response(&amplified_rhodopsin)?;
        
        Ok(super::AmplifiedResponse {
            rod_signals: amplified_rhodopsin,
            cone_signals: amplified_cone,
            temporal_response,
            amplification_factor: self.amplification_factor,
        })
    }

    /// Calibrates rhodopsin cascade based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.transducin_activation.calibrate(params)?;
        self.phosphodiesterase_activation.calibrate(params)?;
        self.cgmp_hydrolysis.calibrate(params)?;
        self.channel_closure.calibrate(params)?;
        
        // Adjust amplification factor based on rod sensitivity
        self.amplification_factor = 1e6 * params.rod_sensitivity;
        Ok(())
    }

    fn generate_temporal_response(&self, rod_signals: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        // Generate temporal response based on rod signal characteristics
        let mut temporal_response = Vec::with_capacity(10);
        
        for i in 0..10 {
            let time_point = i as f64 / 10.0;
            let avg_signal = rod_signals.iter().sum::<f64>() / rod_signals.len() as f64;
            
            // Temporal response follows exponential decay
            let temporal_value = avg_signal * (-time_point * 2.0).exp();
            temporal_response.push(temporal_value);
        }
        
        Ok(temporal_response)
    }
}

/// Transducin activation stage of the cascade
#[derive(Debug, Clone)]
pub struct TransducinActivation {
    activation_rate: f64,
    deactivation_rate: f64,
    gdp_gtp_exchange_rate: f64,
}

impl TransducinActivation {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            activation_rate: 0.8,
            deactivation_rate: 0.1,
            gdp_gtp_exchange_rate: 0.9,
        })
    }

    pub fn activate(&self, rhodopsin_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut transducin_response = Vec::with_capacity(rhodopsin_response.len());
        
        for &rhodopsin in rhodopsin_response {
            // Transducin activation is proportional to rhodopsin activation
            let activation = rhodopsin * self.activation_rate * self.gdp_gtp_exchange_rate;
            transducin_response.push(activation.min(1.0).max(0.0));
        }
        
        Ok(transducin_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust activation rate based on rod sensitivity
        self.activation_rate = 0.8 * params.rod_sensitivity;
        Ok(())
    }
}

/// Phosphodiesterase activation stage of the cascade
#[derive(Debug, Clone)]
pub struct PhosphodiesteraseActivation {
    activation_rate: f64,
    catalytic_rate: f64,
    inhibition_rate: f64,
}

impl PhosphodiesteraseActivation {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            activation_rate: 0.7,
            catalytic_rate: 0.9,
            inhibition_rate: 0.05,
        })
    }

    pub fn activate(&self, transducin_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut pde_response = Vec::with_capacity(transducin_response.len());
        
        for &transducin in transducin_response {
            // PDE activation is proportional to transducin activation
            let activation = transducin * self.activation_rate * self.catalytic_rate;
            let inhibited_activation = activation * (1.0 - self.inhibition_rate);
            pde_response.push(inhibited_activation.min(1.0).max(0.0));
        }
        
        Ok(pde_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust activation rate based on rod sensitivity
        self.activation_rate = 0.7 * params.rod_sensitivity;
        Ok(())
    }
}

/// cGMP hydrolysis stage of the cascade
#[derive(Debug, Clone)]
pub struct CgmpHydrolysis {
    hydrolysis_rate: f64,
    cgmp_concentration: f64,
    basal_hydrolysis_rate: f64,
}

impl CgmpHydrolysis {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            hydrolysis_rate: 0.8,
            cgmp_concentration: 1.0, // Normalized concentration
            basal_hydrolysis_rate: 0.1,
        })
    }

    pub fn hydrolyze(&self, pde_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut cgmp_response = Vec::with_capacity(pde_response.len());
        
        for &pde in pde_response {
            // cGMP hydrolysis is proportional to PDE activity
            let hydrolysis = pde * self.hydrolysis_rate * self.cgmp_concentration;
            let basal_hydrolysis = self.basal_hydrolysis_rate * self.cgmp_concentration;
            let total_hydrolysis = hydrolysis + basal_hydrolysis;
            
            // cGMP response is inverse of hydrolysis (less cGMP = more response)
            let cgmp_level = 1.0 - total_hydrolysis;
            cgmp_response.push(cgmp_level.min(1.0).max(0.0));
        }
        
        Ok(cgmp_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust hydrolysis rate based on rod sensitivity
        self.hydrolysis_rate = 0.8 * params.rod_sensitivity;
        Ok(())
    }
}

/// Channel closure stage of the cascade
#[derive(Debug, Clone)]
pub struct ChannelClosure {
    channel_open_probability: f64,
    cgmp_sensitivity: f64,
    calcium_feedback: f64,
}

impl ChannelClosure {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            channel_open_probability: 0.1, // Low probability in dark
            cgmp_sensitivity: 0.9,
            calcium_feedback: 0.2,
        })
    }

    pub fn close(&self, cgmp_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut channel_response = Vec::with_capacity(cgmp_response.len());
        
        for &cgmp in cgmp_response {
            // Channel closure is proportional to cGMP reduction
            let closure = (1.0 - cgmp) * self.cgmp_sensitivity;
            let calcium_effect = closure * self.calcium_feedback;
            let total_closure = closure + calcium_effect;
            
            // Channel response is inverse of closure (more closure = more response)
            let channel_activity = total_closure;
            channel_response.push(channel_activity.min(1.0).max(0.0));
        }
        
        Ok(channel_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust cGMP sensitivity based on rod sensitivity
        self.cgmp_sensitivity = 0.9 * params.rod_sensitivity;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhodopsin_cascade_creation() {
        let cascade = RhodopsinCascade::new();
        assert!(cascade.is_ok());
    }

    #[test]
    fn test_transducin_activation() {
        let transducin = TransducinActivation::new().unwrap();
        let rhodopsin = vec![0.1, 0.5, 0.9];
        let result = transducin.activate(&rhodopsin);
        assert!(result.is_ok());
    }

    #[test]
    fn test_phosphodiesterase_activation() {
        let pde = PhosphodiesteraseActivation::new().unwrap();
        let transducin = vec![0.1, 0.5, 0.9];
        let result = pde.activate(&transducin);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cgmp_hydrolysis() {
        let cgmp = CgmpHydrolysis::new().unwrap();
        let pde = vec![0.1, 0.5, 0.9];
        let result = cgmp.hydrolyze(&pde);
        assert!(result.is_ok());
    }

    #[test]
    fn test_channel_closure() {
        let channel = ChannelClosure::new().unwrap();
        let cgmp = vec![0.1, 0.5, 0.9];
        let result = channel.close(&cgmp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_amplification_factor() {
        let cascade = RhodopsinCascade::new().unwrap();
        assert_eq!(cascade.amplification_factor, 1e6);
    }
}
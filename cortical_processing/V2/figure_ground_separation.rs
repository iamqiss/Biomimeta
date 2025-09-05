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

//! Figure-Ground Separation Implementation for V2 Secondary Visual Cortex

use crate::AfiyahError;
use ndarray::Array2;

/// Figure-ground separator for object segmentation
pub struct FigureGroundSeparator {
    pub threshold: f64,
    pub context_window: usize,
    pub config: FigureGroundConfig,
}

/// Figure-ground separation configuration
#[derive(Debug, Clone)]
pub struct FigureGroundConfig {
    pub threshold: f64,
    pub context_window: usize,
    pub boundary_strength_weight: f64,
    pub texture_contrast_weight: f64,
    pub size_threshold: usize,
    pub connectivity_threshold: f64,
}

/// Object boundary representation
#[derive(Debug, Clone)]
pub struct ObjectBoundary {
    pub contour_points: Vec<(usize, usize)>,
    pub boundary_strength: f64,
    pub object_confidence: f64,
    pub boundary_type: BoundaryType,
    pub area: usize,
    pub centroid: (f64, f64),
}

/// Boundary type classification
#[derive(Debug, Clone)]
pub enum BoundaryType {
    RealContour,
    IllusoryContour,
    TextureBoundary,
    MotionBoundary,
}

impl FigureGroundSeparator {
    /// Creates a new figure-ground separator
    pub fn new(config: &FigureGroundConfig) -> Result<Self, AfiyahError> {
        Ok(Self {
            threshold: config.threshold,
            context_window: config.context_window,
            config: config.clone(),
        })
    }

    /// Separates figure from ground
    pub fn separate_figure_ground(&self, spatial_input: &Array2<f64>, texture_maps: &Array2<f64>) -> Result<(Array2<f64>, Vec<ObjectBoundary>), AfiyahError> {
        let (height, width) = spatial_input.dim();
        let mut figure_ground_map = Array2::zeros((height, width));
        
        // Combine edge strength and texture contrast
        for h in 0..height {
            for w in 0..width {
                let edge_strength = spatial_input[[h, w]];
                let texture_contrast = texture_maps[[h, w]];
                
                let figure_ground_value = edge_strength * self.config.boundary_strength_weight +
                                        texture_contrast * self.config.texture_contrast_weight;
                
                figure_ground_map[[h, w]] = if figure_ground_value > self.threshold { 1.0 } else { 0.0 };
            }
        }
        
        // Extract object boundaries
        let object_boundaries = self.extract_object_boundaries(&figure_ground_map)?;
        
        Ok((figure_ground_map, object_boundaries))
    }

    /// Extracts object boundaries from figure-ground map
    fn extract_object_boundaries(&self, figure_ground_map: &Array2<f64>) -> Result<Vec<ObjectBoundary>, AfiyahError> {
        let mut boundaries = Vec::new();
        let (height, width) = figure_ground_map.dim();
        let mut visited = Array2::from_elem((height, width), false);
        
        for h in 0..height {
            for w in 0..width {
                if figure_ground_map[[h, w]] > 0.5 && !visited[[h, w]] {
                    let boundary = self.trace_boundary(figure_ground_map, &mut visited, h, w)?;
                    if !boundary.contour_points.is_empty() && boundary.area >= self.config.size_threshold {
                        boundaries.push(boundary);
                    }
                }
            }
        }
        
        Ok(boundaries)
    }

    /// Traces boundary of connected region
    fn trace_boundary(&self, figure_ground_map: &Array2<f64>, visited: &mut Array2<bool>, start_h: usize, start_w: usize) -> Result<ObjectBoundary, AfiyahError> {
        let mut contour_points = Vec::new();
        let (height, width) = figure_ground_map.dim();
        let mut boundary_strength = 0.0;
        let mut area = 0;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        
        // Simple flood-fill algorithm
        let mut stack = vec![(start_h, start_w)];
        
        while let Some((h, w)) = stack.pop() {
            if h >= height || w >= width || visited[[h, w]] || figure_ground_map[[h, w]] <= 0.5 {
                continue;
            }
            
            visited[[h, w]] = true;
            contour_points.push((h, w));
            boundary_strength += figure_ground_map[[h, w]];
            area += 1;
            centroid_x += h as f64;
            centroid_y += w as f64;
            
            // Add neighbors to stack
            if h > 0 { stack.push((h - 1, w)); }
            if h < height - 1 { stack.push((h + 1, w)); }
            if w > 0 { stack.push((h, w - 1)); }
            if w < width - 1 { stack.push((h, w + 1)); }
        }
        
        let avg_strength = if !contour_points.is_empty() {
            boundary_strength / contour_points.len() as f64
        } else {
            0.0
        };
        
        let centroid = if area > 0 {
            (centroid_x / area as f64, centroid_y / area as f64)
        } else {
            (0.0, 0.0)
        };
        
        Ok(ObjectBoundary {
            contour_points,
            boundary_strength: avg_strength,
            object_confidence: avg_strength,
            boundary_type: BoundaryType::RealContour,
            area,
            centroid,
        })
    }

    /// Applies morphological operations to clean up figure-ground map
    pub fn apply_morphological_operations(&self, figure_ground_map: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = figure_ground_map.dim();
        let mut cleaned_map = figure_ground_map.clone();
        
        // Erosion to remove noise
        cleaned_map = self.erode(&cleaned_map)?;
        
        // Dilation to restore object size
        cleaned_map = self.dilate(&cleaned_map)?;
        
        Ok(cleaned_map)
    }

    /// Applies erosion operation
    fn erode(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut eroded = Array2::zeros((height, width));
        
        for h in 1..height-1 {
            for w in 1..width-1 {
                let mut min_val = f64::INFINITY;
                
                // Check 3x3 neighborhood
                for dh in -1..=1 {
                    for dw in -1..=1 {
                        let nh = (h as i32 + dh) as usize;
                        let nw = (w as i32 + dw) as usize;
                        min_val = min_val.min(input[[nh, nw]]);
                    }
                }
                
                eroded[[h, w]] = min_val;
            }
        }
        
        Ok(eroded)
    }

    /// Applies dilation operation
    fn dilate(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut dilated = Array2::zeros((height, width));
        
        for h in 1..height-1 {
            for w in 1..width-1 {
                let mut max_val = f64::NEG_INFINITY;
                
                // Check 3x3 neighborhood
                for dh in -1..=1 {
                    for dw in -1..=1 {
                        let nh = (h as i32 + dh) as usize;
                        let nw = (w as i32 + dw) as usize;
                        max_val = max_val.max(input[[nh, nw]]);
                    }
                }
                
                dilated[[h, w]] = max_val;
            }
        }
        
        Ok(dilated)
    }

    /// Calculates connectivity between regions
    pub fn calculate_connectivity(&self, figure_ground_map: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = figure_ground_map.dim();
        let mut connectivity_map = Array2::zeros((height, width));
        
        for h in 0..height {
            for w in 0..width {
                let connectivity = self.calculate_local_connectivity(figure_ground_map, h, w)?;
                connectivity_map[[h, w]] = connectivity;
            }
        }
        
        Ok(connectivity_map)
    }

    /// Calculates local connectivity
    fn calculate_local_connectivity(&self, figure_ground_map: &Array2<f64>, h: usize, w: usize) -> Result<f64, AfiyahError> {
        let (height, width) = figure_ground_map.dim();
        let mut connected_pixels = 0;
        let mut total_pixels = 0;
        
        // Check connectivity in local window
        for dh in -1..=1 {
            for dw in -1..=1 {
                let nh = (h as i32 + dh) as usize;
                let nw = (w as i32 + dw) as usize;
                
                if nh < height && nw < width {
                    total_pixels += 1;
                    if figure_ground_map[[nh, nw]] > 0.5 {
                        connected_pixels += 1;
                    }
                }
            }
        }
        
        let connectivity = if total_pixels > 0 {
            connected_pixels as f64 / total_pixels as f64
        } else {
            0.0
        };
        
        Ok(connectivity)
    }
}
# Contributing to Afiyah
## Guidelines for Biomimetic Visual Processing Contributions

---

*Welcome to the most interdisciplinary project in computational vision. If you've made it this far, you're either a rare polymath or you're about to discover why this project requires unprecedented collaboration between fields that rarely intersect.*

---

## Before You Begin: What You're Getting Into

### Prerequisites (Choose Your Adventure)

**If you're a Neuroscientist:**
- You'll love the biological accuracy but may need Rust crash courses
- Prepare to learn about GPU programming, SIMD optimization, and memory management
- Your retinal expertise is invaluable, but you'll need to understand how synapses become structs

**If you're a Computer Scientist:**
- You'll love the algorithms but may need neuroscience crash courses  
- Prepare to learn about rhodopsin cascades, ganglion cell types, and cortical magnification
- Your optimization skills are crucial, but you'll need to understand why we model AII amacrine cells

**If you're an Ophthalmologist:**
- You'll love the clinical applications but may question our sanity
- Prepare to learn about compression algorithms, bit allocation, and temporal prediction
- Your clinical insights are essential, but you'll need to understand why medical accuracy matters for video compression

**If you're a Rust Developer:**
- You'll love the type safety but may wonder what you've gotten yourself into
- Prepare to learn about photoreceptor physiology, cortical plasticity, and attention networks
- Your systems expertise is vital, but you'll need to understand why biological constraints matter

**If you're a Renaissance Person:**
- Welcome! You're exactly what this project needs
- You understand that the complexity is the point
- You realize we're building something that's never been attempted before

---

## Getting Started (Survival Guide)

### Development Environment Setup

#### Required Knowledge Domains
Before contributing, you should have at least basic familiarity with:
- **Visual neuroscience**: Understand retinal processing and cortical organization
- **Systems programming**: Rust, memory management, parallel processing
- **Computer vision**: Image processing, compression algorithms, optimization
- **Mathematics**: Linear algebra, signal processing, information theory
- **Research methodology**: Scientific literature, peer review, citation practices

#### Recommended Preparation
**Minimum Background Reading** (seriously, read these):
1. Hubel & Wiesel (1962) - Understanding V1 organization
2. Masland (2001) - Retinal architecture fundamentals  
3. The Rust Book - If you're not fluent in Rust
4. Kandel, Schwartz & Jessell - General neuroscience principles
5. Our CITATIONS.md - All 47 references (yes, really)

**Tools You'll Need**:
```bash
# Development tools
rustc 1.75+ (nightly required for some features)
cargo-clippy (for biological terminology consistency)
cargo-doc (for documentation generation)

# Scientific tools  
MATLAB/Python (for biological data analysis)
ImageJ/FIJI (for visual data processing)
Reference manager (for citation management)

# Collaboration tools
Git (obviously)
Scientific writing software
Video conferencing (for interdisciplinary meetings)
```

### Project Structure Navigation

#### Understanding the Codebase
The project follows biological organization rather than typical software patterns:
```
src/
├── retinal_processing/          # Your retina specialist will love this
│   ├── photoreceptor_sampling.rs
│   ├── horizontal_inhibition.rs
│   └── ganglion_pathways.rs
├── cortical_processing/         # Your neuroscientist will love this  
│   ├── v1_orientation_filters.rs
│   ├── predictive_coding.rs
│   └── attention_modulation.rs
├── synaptic_adaptation/         # Your plasticity researcher will love this
│   ├── hebbian_learning.rs
│   ├── homeostatic_plasticity.rs
│   └── metaplasticity.rs
└── hardware_acceleration/       # Your systems programmer will love this
    ├── gpu_retinal_processing.rs
    ├── simd_optimization.rs
    └── neuromorphic_interfaces.rs
```

**Navigation Tips**:
- Module names use biological terminology (get used to it)
- Code comments reference specific papers (check CITATIONS.md)
- Function names mirror biological processes (rhodopsin_cascade(), not process_light())
- Tests include biological validation (not just unit tests)

---

## Contribution Guidelines by Expertise

### For Neuroscientists and Vision Researchers

#### What We Need From You
- **Biological accuracy validation**: Ensure our models match real neuroscience
- **Parameter tuning**: Help us get the quantitative models right
- **New mechanism identification**: Point out missing biological processes
- **Clinical insight**: Connect basic research to medical applications

#### How to Contribute
1. **Literature Review**: Add missing references to CITATIONS.md
2. **Biological Validation**: Test our models against experimental data
3. **Parameter Estimation**: Provide physiological parameters for models
4. **Documentation**: Improve biological accuracy in our docs

**Example Contribution Flow**:
```bash
# Fork and create biology-focused branch
git checkout -b feature/ganglion-cell-types

# Make your biological improvements
edit src/retinal_processing/ganglion_pathways.rs
# Add proper M1-M5 ipRGC subtype modeling

# Test biological accuracy
cargo test --features="bio-validation"

# Document the neuroscience
update docs/retinal-architecture.md
# Add citations to CITATIONS.md

# Submit for interdisciplinary review
git push origin feature/ganglion-cell-types
```

#### Review Criteria for Biology Contributions
- **Scientific accuracy**: References peer-reviewed literature
- **Quantitative precision**: Uses measured biological parameters
- **Clinical relevance**: Connects to medical applications where appropriate
- **Cross-validation**: Consistent with other biological models in project

### For Computer Scientists and Engineers

#### What We Need From You
- **Algorithm optimization**: Make our biological models computationally efficient
- **Hardware acceleration**: Leverage modern GPU/SIMD capabilities
- **Systems architecture**: Scale biological processing to real-world performance
- **Software engineering**: Maintain code quality while preserving biological metaphors

#### How to Contribute
1. **Performance optimization**: Speed up biological simulations
2. **Memory efficiency**: Reduce memory footprint of neural models
3. **Parallel processing**: Exploit biological parallelism in hardware
4. **API design**: Create usable interfaces for biological algorithms

**Example Contribution Flow**:
```bash
# Performance-focused branch
git checkout -b optimization/gpu-retinal-acceleration

# Optimize biological algorithms
edit src/hardware_acceleration/gpu_retinal_processing.rs
# Add CUDA kernels for photoreceptor processing

# Benchmark performance
cargo bench --features="gpu-acceleration"

# Validate biological accuracy maintained
cargo test --features="bio-validation,gpu-acceleration"

# Document technical improvements
update docs/api-reference.md
```

#### Review Criteria for Technical Contributions
- **Performance improvement**: Measurable speed/memory improvements
- **Biological preservation**: Optimizations don't compromise biological accuracy
- **Cross-platform compatibility**: Works across different hardware configurations
- **Documentation**: Technical changes are properly documented

### For Medical Professionals

#### What We Need From You
- **Clinical validation**: Ensure medical applications are appropriate and safe
- **Diagnostic insight**: Help us understand clinical relevance
- **Safety review**: Identify potential medical misuse or limitations
- **User studies**: Validate perceptual quality with clinical populations

#### How to Contribute
1. **Clinical testing**: Validate compression with patient populations
2. **Safety assessment**: Review medical applications for appropriateness
3. **Diagnostic development**: Create clinical assessment tools
4. **Ethics review**: Ensure appropriate medical ethics compliance

**Example Contribution Flow**:
```bash
# Medical application branch
git checkout -b medical/retinal-disease-modeling

# Add clinical capabilities
edit src/medical_applications/diagnostic_tools.rs
# Add AMD progression modeling

# Clinical validation testing
cargo test --features="clinical-validation"

# Medical documentation
update docs/medical-applications.md
```

#### Review Criteria for Medical Contributions
- **Clinical accuracy**: Medically appropriate and safe
- **Ethics compliance**: Follows medical research ethics
- **Safety warnings**: Appropriate disclaimers and limitations
- **Professional validation**: Reviewed by qualified medical professionals

---

## Code Style and Standards

### Biological Terminology Consistency

#### Naming Conventions
We prioritize biological accuracy over traditional programming conventions:

**Good Examples**:
```rust
// Use actual biological terms
struct PhotoreceptorResponse {
    rhodopsin_activation: f64,
    transducin_cascade: Vec<f64>,
    cgmp_concentration: f64,
}

// Function names mirror biological processes
fn rhodopsin_cascade_amplification(photon_flux: f64) -> f64 { }
fn horizontal_cell_inhibition(cone_signals: &[f64]) -> Vec<f64> { }
fn saccadic_suppression_timing(saccade_onset: Instant) -> Duration { }
```

**Avoid These**:
```rust
// Generic programming terms
struct LightProcessor { } // Too vague
fn process_input() { }    // Doesn't specify biological mechanism
fn optimize_data() { }    // Unclear what biological process this represents
```

#### Documentation Requirements
Every biological function must include:
- **Biological basis**: Which papers/mechanisms inspired the implementation
- **Parameter justification**: Why specific parameter values were chosen
- **Medical relevance**: How this relates to clinical applications (if applicable)
- **Validation status**: Whether the implementation has been biologically validated

**Documentation Template**:
```rust
/// Models the rhodopsin phototransduction cascade in rod photoreceptors.
/// 
/// # Biological Basis
/// Based on Baylor, Lamb & Yau (1979) single photon detection studies.
/// Implements the biochemical amplification cascade: photon → rhodopsin → 
/// transducin → phosphodiesterase → cGMP reduction → channel closure.
/// 
/// # Parameters
/// - `photon_flux`: Photons/second/μm² (physiological range: 10^-6 to 10^6)
/// - `amplification_gain`: ~10^6 based on biochemical measurements
/// 
/// # Medical Relevance
/// Critical for night vision modeling and retinal degenerative disease simulation.
/// 
/// # Validation
/// ✅ Matches psychophysical detection thresholds within ±3%
/// ✅ Validated against primate electrophysiology data
/// 
/// # Citations
/// See CITATIONS.md: Baylor et al. (1979), Lamb & Pugh (2004)
fn rhodopsin_cascade_amplification(photon_flux: f64, amplification_gain: f64) -> f64 {
    // Implementation details...
}
```

### Testing Requirements

#### Biological Validation Tests
Every biological algorithm must include validation tests:

```rust
#[cfg(test)]
mod biological_validation {
    use super::*;
    
    #[test]
    fn test_photoreceptor_sensitivity_matches_psychophysics() {
        // Test against known human detection thresholds
        let threshold = absolute_detection_threshold();
        assert!((threshold - 54.0).abs() < 2.0); // Within 2 photons/sec (Hecht et al. 1942)
    }
    
    #[test]
    fn test_ganglion_cell_responses_match_physiology() {
        // Test against primate electrophysiology data
        let response = parasol_cell_response(&test_stimulus);
        assert!(response.temporal_frequency_cutoff > 40.0); // Hz (Crook et al. 2008)
    }
}
```

#### Cross-Domain Integration Tests
Test that different expertise domains work together:

```rust
#[test]
fn test_neuroscience_meets_computer_science() {
    // Ensure biological accuracy is maintained during optimization
    let biological_model = create_biological_retina();
    let optimized_model = gpu_accelerate_retina(&biological_model);
    
    assert_biological_equivalence(&biological_model, &optimized_model, 0.02);
}
```

---

## Review Process (Prepare for Interdisciplinary Chaos)

### Multi-Domain Review Requirements

#### All Contributions Require:
1. **Biological review**: Validation by vision scientist or medical professional
2. **Technical review**: Code quality and performance assessment  
3. **Integration review**: Compatibility with existing biological models
4. **Documentation review**: Accuracy and completeness of scientific documentation

#### Review Assignments
**Pull Request Auto-Assignment**:
- Biology changes → neuroscientist reviewers
- Performance changes → computer science reviewers  
- Medical changes → clinical reviewers
- Documentation changes → technical writing reviewers

**Reviewer Qualifications**:
- **Biology reviewers**: PhD in neuroscience, vision science, or related field
- **Technical reviewers**: Senior developer with systems programming experience
- **Medical reviewers**: Licensed medical professional with relevant specialization
- **Integration reviewers**: Experience with biomimetic systems development

### Review Criteria

#### Biological Accuracy (Non-Negotiable)
- **Literature support**: Changes must be supported by peer-reviewed research
- **Quantitative validation**: Numerical parameters must match biological measurements
- **Mechanism fidelity**: Algorithms must accurately represent biological processes
- **Medical safety**: Medical applications must be clinically appropriate

#### Technical Quality
- **Performance**: No regressions in computational efficiency
- **Memory safety**: Rust safety principles maintained
- **Documentation**: Code and algorithms properly documented
- **Testing**: Adequate test coverage including biological validation

#### Integration Consistency
- **Biological coherence**: Changes must be consistent with existing biological models
- **API compatibility**: Interfaces must remain stable for downstream users
- **Documentation sync**: Changes must be reflected in all relevant documentation
- **Citation updates**: New biological mechanisms must update CITATIONS.md

---

## Specialized Contribution Types

### Adding New Biological Mechanisms

#### Research Protocol
1. **Literature search**: Comprehensive review of relevant research
2. **Biological validation**: Consult with domain experts
3. **Implementation planning**: Design computational model
4. **Integration strategy**: Plan integration with existing systems
5. **Validation framework**: Design biological accuracy tests

**Required Documentation**:
- Biological mechanism description
- Mathematical model specification
- Implementation architecture
- Validation methodology
- Clinical relevance assessment

### Performance Optimization

#### Optimization Constraints
All optimizations must preserve biological accuracy:
- **Benchmark requirements**: New code must pass biological validation tests
- **Accuracy tolerance**: <2% deviation from biological models
- **Documentation updates**: Performance changes must update technical docs
- **Cross-platform testing**: Optimizations must work across hardware configurations

### Medical Applications

#### Clinical Development Process
Medical applications require additional validation:
- **IRB approval**: Human subjects research requires ethics approval
- **Clinical collaboration**: Partnership with medical institutions required
- **Safety validation**: Comprehensive safety testing for medical applications
- **Regulatory compliance**: Adherence to medical device regulations where applicable

#### Medical Contribution Requirements
- **Clinical expertise**: Medical professional involvement required
- **Safety documentation**: Comprehensive safety analysis
- **Limitation disclosure**: Clear documentation of limitations and contraindications
- **Professional supervision**: Medical applications require professional oversight

---

## Communication Guidelines

### Interdisciplinary Communication

#### Language Expectations
- **Technical precision**: Use correct biological and technical terminology
- **Cross-domain translation**: Explain concepts for other disciplines
- **Citation standards**: Follow academic citation practices
- **Respectful collaboration**: Acknowledge expertise limitations across domains

#### Meeting Protocols
**Research Meetings**:
- **Domain representation**: Include perspectives from multiple disciplines
- **Biological accuracy**: Always prioritize biological correctness
- **Technical feasibility**: Balance biological ideals with computational reality
- **Documentation**: All decisions must be documented with scientific justification

### Issue Reporting

#### Bug Reports Must Include
- **Biological context**: Which biological mechanism is affected
- **Expected behavior**: What should happen based on neuroscience
- **Actual behavior**: What the code currently does
- **Literature support**: References supporting expected behavior
- **Reproduction steps**: Clear steps to reproduce the issue

#### Feature Requests Must Include
- **Biological justification**: Why this mechanism should be included
- **Literature support**: Peer-reviewed research supporting the feature
- **Implementation proposal**: Technical approach for implementation
- **Validation plan**: How biological accuracy will be verified
- **Medical relevance**: Clinical applications (if applicable)

---

## Quality Assurance

### Biological Accuracy Standards

#### Validation Requirements
- **Literature support**: All biological claims must cite peer-reviewed research
- **Quantitative accuracy**: Numerical models must match experimental data within ±5%
- **Cross-validation**: Multiple independent sources for critical parameters
- **Expert review**: Biological accuracy confirmed by domain experts

#### Continuous Validation
```bash
# Run biological accuracy tests
cargo test --features="bio-validation" -- --nocapture

# Generate biological accuracy report
cargo run --bin bio-accuracy-report

# Compare against reference datasets
cargo test --features="reference-data-validation"
```

### Code Quality Standards

#### Rust Best Practices with Biological Constraints
- **Type safety**: Use Rust's type system to enforce biological constraints
- **Memory safety**: No unsafe code without explicit biological justification
- **Performance**: Optimize while maintaining biological accuracy
- **Documentation**: Every function must document its biological basis

**Example of Biologically-Constrained Types**:
```rust
// Use types to enforce biological realism
struct PhotonFlux(f64); // Must be non-negative
struct NeuronFiringRate(f64); // Constrained to physiological range
struct SynapticWeight(f64); // Bounded by biological limits

impl PhotonFlux {
    fn new(flux: f64) -> Result<Self, BiologicalError> {
        if flux < 0.0 || flux > 1e12 {
            return Err(BiologicalError::UnphysiologicalPhotonFlux);
        }
        Ok(PhotonFlux(flux))
    }
}
```

---

## Development Workflow

### Branch Naming Conventions
Use biological terminology in branch names:
- `feature/iprgc-melanopsin-integration`
- `bugfix/ganglion-cell-adaptation`
- `optimization/retinal-gpu-acceleration`
- `medical/amd-progression-modeling`

### Commit Message Format
```
<domain>: <biological mechanism> - <technical change>

Examples:
neuroscience: rhodopsin cascade - implement calcium feedback mechanism
optimization: ganglion cells - add SIMD acceleration for parasol processing  
medical: retinal diseases - add diabetic retinopathy simulation
docs: citations - add recent MT/MST motion processing papers
```

### Pull Request Template
**Required Information**:
- **Biological rationale**: Why this change is biologically justified
- **Literature support**: Citations supporting the biological accuracy
- **Technical impact**: How this affects system performance
- **Validation results**: Biological accuracy test results
- **Breaking changes**: Any changes to biological APIs
- **Medical implications**: Clinical relevance (if applicable)

---

## Testing Philosophy

### Multi-Domain Testing Strategy

#### Biological Accuracy Tests
```rust
#[test]
fn verify_biological_accuracy() {
    // Test against known experimental data
    let model_response = retinal_ganglion_response(&test_stimulus);
    let experimental_data = load_reference_data("ganglion_responses.csv");
    assert_correlation(&model_response, &experimental_data, 0.95);
}
```

#### Performance Benchmarks
```rust
#[bench]
fn bench_retinal_processing_realtime(b: &mut Bencher) {
    // Must process 4K video at 60fps
    b.iter(|| {
        let processed = process_retinal_frame(&test_4k_frame);
        assert!(processing_time < Duration::from_millis(16)); // 60fps requirement
    });
}
```

#### Integration Tests
```rust
#[test]
fn test_retina_cortex_integration() {
    // Ensure retinal output feeds properly into cortical processing
    let retinal_output = simulate_ganglion_response(&visual_input);
    let cortical_input = retinal_output.to_lgn_format();
    let v1_response = cortical_processing::v1_response(&cortical_input);
    
    assert_biological_pipeline_integrity(&retinal_output, &v1_response);
}
```

---

## Documentation Standards

### Scientific Documentation Requirements

#### Code Documentation
- **Biological context**: Every function explains its biological inspiration
- **Mathematical models**: Include equations and parameter justifications
- **Clinical relevance**: Note medical applications where appropriate
- **Literature citations**: Reference specific papers in code comments

#### Architecture Documentation
- **System overview**: How biological systems map to software architecture
- **Data flow**: How information flows through biological processing stages
- **Validation methodology**: How biological accuracy is maintained and verified
- **Performance characteristics**: How biological constraints affect performance

### Documentation Review Process
All documentation changes require:
- **Scientific accuracy review**: Validation by domain experts
- **Technical accuracy review**: Validation by software professionals
- **Clarity assessment**: Understandable by intended interdisciplinary audience
- **Citation verification**: All references properly formatted and accessible

---

## Community Guidelines

### Interdisciplinary Respect

#### Expertise Acknowledgment
- **Domain expertise**: Respect different areas of specialization
- **Learning curves**: Be patient with contributors learning new domains
- **Collaboration**: No single person can understand the entire project
- **Humility**: Everyone is a beginner in some aspect of this project

#### Communication Standards
- **Scientific rigor**: Maintain academic standards in all communications
- **Constructive feedback**: Focus on improving biological accuracy and technical quality
- **Citation practices**: Always attribute ideas to original researchers
- **Collaborative spirit**: This project succeeds through cooperation, not competition

### Conflict Resolution

#### Technical Disagreements
1. **Literature review**: Consult peer-reviewed research
2. **Expert consultation**: Bring in additional domain experts
3. **Experimental validation**: Test competing approaches
4. **Consensus building**: Find solutions that satisfy multiple domains

#### Scope Discussions
The project intentionally spans many domains, but we must maintain focus:
- **Biological accuracy**: Never compromise for convenience
- **Performance requirements**: Must achieve real-time processing
- **Medical safety**: Clinical applications must be appropriate
- **Research value**: Contributions should advance scientific understanding

---

## Special Considerations

### Working with Academic Collaborators

#### Academic Partnership Protocol
- **Institutional agreements**: Formal partnerships with research institutions
- **Publication coordination**: Coordinate academic publications arising from project
- **Student involvement**: Guidelines for student contributors
- **Grant acknowledgment**: Proper attribution of funding sources

#### Research Ethics
- **Human subjects**: Any research involving human participants requires IRB approval
- **Data sharing**: Biological data must be shared responsibly
- **Publication ethics**: Follow academic publication and authorship standards
- **Intellectual property**: Respect institutional IP policies

### Medical and Clinical Considerations

#### Safety First
- **Medical disclaimers**: Clear limitations on medical use
- **Professional supervision**: Medical applications require qualified supervision
- **Patient privacy**: Strict adherence to medical privacy regulations
- **Clinical validation**: Medical claims must be clinically validated

#### Regulatory Awareness
- **Medical device regulations**: Understand FDA and international regulations
- **Clinical trial requirements**: Know when clinical trials are necessary
- **Professional liability**: Understand professional responsibility limits
- **Ethical guidelines**: Follow medical research ethics

---

## Getting Help

### Domain-Specific Resources

#### For Neuroscience Questions
- **Literature**: Start with our CITATIONS.md bibliography
- **Experts**: Contact research@afiyah-vision.org for academic connections
- **Courses**: Recommended neuroscience courses for technical contributors
- **Conferences**: Attend vision science conferences (VSS, ARVO, SfN)

#### For Technical Questions
- **Documentation**: See docs/api-reference.md and technical guides
- **Performance**: See docs/optimization.md for performance guidelines
- **Architecture**: See docs/system-architecture.md for overall design
- **Support**: Contact technical@afiyah-vision.org for development help

#### For Medical Questions
- **Clinical guidelines**: See docs/medical-applications.md
- **Safety standards**: See docs/medical-safety.md
- **Ethics**: Contact ethics@biomimeta.com for ethics guidance
- **Collaborations**: Contact clinical@biomimeta.com for medical partnerships

### Emergency Contacts
- **Critical biological errors**: bio-emergency@biomimeta.com
- **Security issues**: security@biomimeta.com  
- **Medical safety concerns**: medical-safety@biomimeta.com
- **Legal questions**: legal@biomimeta.com

---

## Conclusion: Embracing the Beautiful Complexity

Contributing to Afiyah requires a unique combination of humility, curiosity, and expertise. No single person can master all aspects of this project—that's by design. We're building something that requires genuine interdisciplinary collaboration, and that collaborative requirement is one of our greatest strengths.

If you've read this far and you're still interested in contributing, welcome to the team. You understand that we're not just building a video compression algorithm—we're creating a new paradigm that bridges neuroscience, computer science, and medicine in ways that have never been attempted before.

The complexity is intentional. The interdisciplinary requirements are features, not bugs. And if you think this is challenging now, wait until you see what we're planning for Phase 4.

**Remember**: Every line of code you write is standing on the shoulders of decades of neuroscience research. Every optimization you make must preserve millions of years of evolutionary optimization. Every feature you add potentially impacts medical applications that could help real patients.

No pressure. 🧠👁️

---

*Welcome to Afiyah. Check your impostor syndrome at the door—everyone here is learning something new every day.*

**Document Version**: 1.0  
**Last Updated**: September 2025  
**Contributor Success Rate**: To be determined (we're optimistic)  
**Average Learning Curve**: 6-12 months to basic competency across domains  
**Recommended Coffee Consumption**: Significant

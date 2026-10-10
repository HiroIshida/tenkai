use rkern::{
    cuda::{self, MathMode, Precision, SphereCenter},
    fk::FkDag,
    scheduling, urdf,
};

#[path = "cuda/cases.rs"]
mod cases;
mod common;
#[path = "cuda/harness.rs"]
mod harness;

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F64];

#[test]
fn invalid_kernel_inputs_return_errors() {
    let fk =
        FkDag::from_robot(&urdf::read_from_string(include_str!("fixtures/fk_chain.urdf")).unwrap())
            .unwrap();
    for precision in PRECISIONS {
        let options = cuda::Options {
            math: precision.into(),
            ..cuda::Options::default()
        };
        for index in [fk.link_poses().len(), usize::MAX] {
            let error = cuda::InvalidLinkIndex {
                index,
                link_count: fk.link_poses().len(),
            };
            assert_eq!(
                cuda::generate_fk_kernel(&fk, index, options).unwrap_err(),
                error
            );
            let spheres = [
                SphereCenter {
                    link_index: 0,
                    position: [0.0; 3],
                },
                SphereCenter {
                    link_index: index,
                    position: [0.0; 3],
                },
            ];
            assert_eq!(
                cuda::generate_centers_kernel(&fk, &spheres, options).unwrap_err(),
                cuda::CentersError::InvalidLink {
                    sphere_index: 1,
                    source: error
                }
            );
        }
        for coordinate in 0..3 {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut spheres = [SphereCenter {
                    link_index: 0,
                    position: [0.0; 3],
                }; 2];
                spheres[1].position[coordinate] = value;
                assert_eq!(
                    cuda::generate_centers_kernel(&fk, &spheres, options).unwrap_err(),
                    cuda::CentersError::NonFinitePosition { sphere_index: 1 }
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a C++ compiler; set CXX if needed"]
fn generated_batch_arithmetic_matches_the_reference() {
    for (math, precision) in [
        (MathMode::F32, Precision::F32),
        (MathMode::F64, Precision::F64),
    ] {
        for scheduling in [
            scheduling::Mode::Live(scheduling::Options::default()),
            scheduling::Mode::DagOrder,
        ] {
            harness::check_arithmetic(&cases::suite(cuda::Options { math, scheduling }), precision);
        }
    }
}

#[test]
#[ignore = "requires the CUDA toolkit; set NVCC and CXX if needed"]
fn generated_source_compiles_with_nvcc() {
    for math in [MathMode::F32, MathMode::F64, MathMode::F32Sfu] {
        for scheduling in [
            scheduling::Mode::Live(scheduling::Options::default()),
            scheduling::Mode::DagOrder,
        ] {
            harness::check_compilation(&cases::suite(cuda::Options { math, scheduling }), math);
        }
    }
}

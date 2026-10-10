use std::f64::consts::FRAC_PI_2;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rkern::{
    cuda::{self, Precision, SphereCenter},
    fk::FkDag,
    urdf,
};

mod common;

const CHAIN: &str = include_str!("fixtures/fk_chain.urdf");
const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F64];

enum Output {
    Link(usize),
    Centers(Vec<SphereCenter>),
}

struct Case {
    name: String,
    source: String,
    entry: &'static str,
    components: usize,
    count: usize,
    inputs: Vec<f64>,
    expected: Vec<f64>,
}

fn test_directory(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("cuda-{name}-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn run(command: &mut Command) {
    let output = command.output().expect("failed to execute test command");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn sphere_center(fk: &FkDag, values: &[f64], sphere: &SphereCenter) -> [f64; 3] {
    let pose = &fk.link_poses()[sphere.link_index];
    let [x, y, z, w] = pose.rotation.map(|id| values[id.index()]);
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    [0, 1, 2].map(|i| {
        values[pose.translation[i].index()]
            + rotation[i]
                .iter()
                .zip(sphere.position)
                .map(|(a, b)| a * b)
                .sum::<f64>()
    })
}

fn case(
    name: String,
    fk: &FkDag,
    output: Output,
    configurations: &[Vec<f64>],
    precision: Precision,
) -> Case {
    let (entry, source, components) = match &output {
        Output::Link(link) => (
            "rkern_fk",
            cuda::generate_fk_kernel_with_precision(fk, *link, precision).unwrap(),
            7,
        ),
        Output::Centers(spheres) => (
            "rkern_centers",
            cuda::generate_centers_kernel_with_precision(fk, spheres, precision).unwrap(),
            3 * spheres.len(),
        ),
    };
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for configuration in configurations {
        let q: Vec<_> = configuration
            .iter()
            .map(|&v| match precision {
                Precision::F32 => f64::from(v as f32),
                Precision::F64 => v,
            })
            .collect();
        let values = common::evaluate(fk, &q);
        match &output {
            Output::Link(link) => {
                let pose = &fk.link_poses()[*link];
                expected.extend(
                    pose.translation
                        .iter()
                        .chain(&pose.rotation)
                        .map(|id| values[id.index()]),
                );
            }
            Output::Centers(spheres) => {
                for sphere in spheres {
                    expected.extend(sphere_center(fk, &values, sphere));
                }
            }
        }
        inputs.extend(q);
    }
    Case {
        name,
        source,
        entry,
        components,
        count: configurations.len(),
        inputs,
        expected,
    }
}

fn cases(precision: Precision) -> Vec<Case> {
    let chain = FkDag::from_robot(&urdf::read_from_string(CHAIN).unwrap()).unwrap();
    let configurations = [
        vec![0.0, 0.0, 0.0, 0.0],
        vec![FRAC_PI_2, FRAC_PI_2, FRAC_PI_2, 0.5],
        vec![0.31, -0.72, 1.17, 0.24],
        vec![-1.2, 0.8, -0.4, -0.5],
        vec![std::f64::consts::PI, -FRAC_PI_2, 0.4, 0.3],
    ];
    let mut cases = Vec::new();
    for (link, pose) in chain.link_poses().iter().enumerate() {
        cases.push(case(
            format!("chain_fk_{}", pose.name),
            &chain,
            Output::Link(link),
            &configurations,
            precision,
        ));
    }
    let spheres = (0..chain.link_poses().len())
        .rev()
        .flat_map(|link_index| {
            [[0.0; 3], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.7]].map(|position| SphereCenter {
                link_index,
                position,
            })
        })
        .collect();
    cases.push(case(
        "chain_centers".into(),
        &chain,
        Output::Centers(spheres),
        &configurations,
        precision,
    ));
    cases.push(case(
        "empty_centers".into(),
        &chain,
        Output::Centers(vec![]),
        &configurations,
        precision,
    ));

    let root = FkDag::from_robot(
        &urdf::read_from_string(r#"<robot name="one"><link name="root"/></robot>"#).unwrap(),
    )
    .unwrap();
    let no_inputs = vec![vec![]; configurations.len()];
    cases.push(case(
        "root_fk".into(),
        &root,
        Output::Link(0),
        &no_inputs,
        precision,
    ));
    cases.push(case(
        "root_centers".into(),
        &root,
        Output::Centers(vec![SphereCenter {
            link_index: 0,
            position: [0.25, -0.5, 1.0],
        }]),
        &no_inputs,
        precision,
    ));

    let fetch = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf")).unwrap(),
    )
    .unwrap();
    let gripper = fetch
        .link_poses()
        .iter()
        .position(|pose| pose.name == "gripper_link")
        .unwrap();
    let inputs: Vec<Vec<f64>> = (0..5)
        .map(|batch| {
            (0..fetch.joint_names().len())
                .map(|joint| 0.5 * (batch as f64 * 0.7 + joint as f64 * 0.31).sin())
                .collect()
        })
        .collect();
    cases.push(case(
        "fetch_gripper".into(),
        &fetch,
        Output::Link(gripper),
        &inputs,
        precision,
    ));
    cases
}

fn kernel_source(cases: &[Case]) -> String {
    let mut source = String::new();
    for (i, case) in cases.iter().enumerate() {
        writeln!(source, "#define {} kernel_{i}", case.entry).unwrap();
        source.push_str(&case.source);
        writeln!(source, "#undef {}", case.entry).unwrap();
    }
    source
}

fn cpp_array(values: &[f64]) -> String {
    if values.is_empty() {
        return "0.0".into();
    }
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn invalid_kernel_inputs_return_errors() {
    let fk = FkDag::from_robot(&urdf::read_from_string(CHAIN).unwrap()).unwrap();
    for precision in PRECISIONS {
        for index in [fk.link_poses().len(), usize::MAX] {
            let error = cuda::InvalidLinkIndex {
                index,
                link_count: fk.link_poses().len(),
            };
            assert_eq!(
                cuda::generate_fk_kernel_with_precision(&fk, index, precision).unwrap_err(),
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
                cuda::generate_centers_kernel_with_precision(&fk, &spheres, precision).unwrap_err(),
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
                    cuda::generate_centers_kernel_with_precision(&fk, &spheres, precision)
                        .unwrap_err(),
                    cuda::CentersError::NonFinitePosition { sphere_index: 1 }
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a C++ compiler; set CXX if needed"]
fn generated_batch_arithmetic_matches_the_reference() {
    for precision in PRECISIONS {
        let cases = cases(precision);
        let (scalar, tolerance) = match precision {
            Precision::F32 => ("float", "1e-5"),
            Precision::F64 => ("double", "1e-12"),
        };
        let mut data = format!("using Scalar = {scalar};\nconst double tolerance = {tolerance};\n");
        for (i, case) in cases.iter().enumerate() {
            writeln!(
                data,
                "const Scalar inputs_{i}[] = {{{}}};",
                cpp_array(&case.inputs)
            )
            .unwrap();
            writeln!(
                data,
                "const double expected_{i}[] = {{{}}};",
                cpp_array(&case.expected)
            )
            .unwrap();
        }
        data.push_str("using Kernel = void (*)(const Scalar*, Scalar*, size_t);\nstruct Case { const char* name; Kernel kernel; size_t components; size_t count; const Scalar* inputs; const double* expected; };\nconst Case cases[] = {\n");
        for (i, case) in cases.iter().enumerate() {
            writeln!(
                data,
                "{{\"{}\", kernel_{i}, {}, {}, inputs_{i}, expected_{i}}},",
                case.name, case.components, case.count
            )
            .unwrap();
        }
        data.push_str("};\n");
        let directory = test_directory(&format!("host-{precision:?}"));
        fs::write(directory.join("kernels.cuh"), kernel_source(&cases)).unwrap();
        fs::write(directory.join("expected.h"), data).unwrap();
        fs::write(
            directory.join("main.cpp"),
            include_str!("fixtures/cuda_host.cpp"),
        )
        .unwrap();
        let executable = directory.join("check");
        let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
        run(Command::new(compiler)
            .args(["-std=c++11", "-O2", "-ffp-contract=off", "-Wall", "-Werror"])
            .arg(directory.join("main.cpp"))
            .arg("-o")
            .arg(&executable));
        run(&mut Command::new(executable));
    }
}

#[test]
#[ignore = "requires the CUDA toolkit; set NVCC and CXX if needed"]
fn generated_source_compiles_with_nvcc() {
    for precision in PRECISIONS {
        let directory = test_directory(&format!("nvcc-{precision:?}"));
        let path = directory.join("kernels.cu");
        fs::write(&path, kernel_source(&cases(precision))).unwrap();
        let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
        let mut command = Command::new(nvcc);
        command.args(["--ptx", "-std=c++11"]);
        if let Some(compiler) = std::env::var_os("CXX") {
            command.arg("-ccbin").arg(compiler);
        }
        run(command
            .arg(path)
            .arg("-o")
            .arg(directory.join("kernels.ptx")));
    }
}

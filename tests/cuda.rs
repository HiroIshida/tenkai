use std::f64::consts::FRAC_PI_2;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rkern::{cuda, fk::FkDag, urdf};

mod common;

const CHAIN: &str = include_str!("fixtures/fk_chain.urdf");

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

fn append_kernel(
    source: &mut String,
    fk: &FkDag,
    link: usize,
    kernel: usize,
    precision: cuda::Precision,
) {
    writeln!(source, "#define rkern_fk kernel_{kernel}").unwrap();
    source.push_str(&cuda::generate_fk_kernel_with_precision(fk, link, precision).unwrap());
    source.push_str("#undef rkern_fk\n");
}

fn cpp_array(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn invalid_link_indices_return_errors() {
    let fk = FkDag::from_robot(&urdf::read_from_string(CHAIN).unwrap()).unwrap();
    for index in [fk.link_poses().len(), usize::MAX] {
        let error = cuda::generate_fk_kernel(&fk, index).unwrap_err();
        assert_eq!(error.index, index);
        assert_eq!(error.link_count, fk.link_poses().len());
    }
}

#[test]
#[ignore = "requires a C++ compiler; set CXX if needed"]
fn generated_batch_arithmetic_matches_the_dag() {
    for precision in [cuda::Precision::F32, cuda::Precision::F64] {
        check_batch_arithmetic(precision);
    }
}

fn check_batch_arithmetic(precision: cuda::Precision) {
    let robot = urdf::read_from_string(CHAIN).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    let configurations = [
        [0.0, 0.0, 0.0, 0.0],
        [FRAC_PI_2, FRAC_PI_2, FRAC_PI_2, 0.5],
        [0.31, -0.72, 1.17, 0.24],
        [-1.2, 0.8, -0.4, -0.5],
        [std::f64::consts::PI, -FRAC_PI_2, 0.4, 0.3],
    ];
    let mut source = String::new();
    let mut expected = Vec::new();
    for (link, pose) in fk.link_poses().iter().enumerate() {
        append_kernel(&mut source, &fk, link, link, precision);
        for configuration in configurations {
            let values = common::evaluate(&fk, &configuration);
            expected.extend(
                pose.translation
                    .iter()
                    .chain(&pose.rotation)
                    .map(|id| values[id.index()]),
            );
        }
    }
    let root = urdf::read_from_string(r#"<robot name="one"><link name="root"/></robot>"#).unwrap();
    let root_fk = FkDag::from_robot(&root).unwrap();
    append_kernel(&mut source, &root_fk, 0, fk.link_poses().len(), precision);
    for _ in configurations {
        expected.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }
    let kernel_count = fk.link_poses().len() + 1;
    let kernel_names = (0..kernel_count)
        .map(|i| format!("kernel_{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let inputs = configurations.into_iter().flatten().collect::<Vec<_>>();
    let data = format!(
        "const size_t kernel_count = {kernel_count};\n\
         const size_t configuration_count = {};\n\
         const double inputs[16 * 4] = {{{}}};\n\
         const double expected[] = {{{}}};\n\
         using Kernel = void (*)(const double*, double*, size_t);\n\
         const Kernel kernels[] = {{{kernel_names}}};\n",
        configurations.len(),
        cpp_array(&inputs),
        cpp_array(&expected),
    );
    let mut harness = include_str!("fixtures/cuda_host.cpp").to_owned();
    let data = if precision == cuda::Precision::F32 {
        harness = harness.replace("double", "float").replace("1e-12", "1e-5");
        data.replace("const double inputs", "const float inputs")
            .replace(
                "void (*)(const double*, double*",
                "void (*)(const float*, float*",
            )
    } else {
        data
    };
    let directory = test_directory(&format!("host-{precision:?}"));
    fs::write(directory.join("kernels.cuh"), source).unwrap();
    fs::write(directory.join("expected.h"), data).unwrap();
    fs::write(directory.join("main.cpp"), harness).unwrap();
    let executable = directory.join("check");
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    run(Command::new(compiler)
        .args(["-std=c++11", "-O2", "-ffp-contract=off", "-Wall", "-Werror"])
        .arg(directory.join("main.cpp"))
        .arg("-o")
        .arg(&executable));
    run(&mut Command::new(executable));
}

#[test]
#[ignore = "requires the CUDA toolkit; set NVCC and CXX if needed"]
fn generated_source_compiles_with_nvcc() {
    for precision in [cuda::Precision::F32, cuda::Precision::F64] {
        check_nvcc(precision);
    }
}

fn check_nvcc(precision: cuda::Precision) {
    let chain = FkDag::from_robot(&urdf::read_from_string(CHAIN).unwrap()).unwrap();
    let fetch = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf")).unwrap(),
    )
    .unwrap();
    let mut source = String::new();
    append_kernel(&mut source, &chain, 0, 0, precision);
    let tool = chain
        .link_poses()
        .iter()
        .position(|pose| pose.name == "tool")
        .unwrap();
    append_kernel(&mut source, &chain, tool, 1, precision);
    let gripper = fetch
        .link_poses()
        .iter()
        .position(|pose| pose.name == "gripper_link")
        .unwrap();
    append_kernel(&mut source, &fetch, gripper, 2, precision);
    let directory = test_directory(&format!("nvcc-{precision:?}"));
    let path = directory.join("kernels.cu");
    fs::write(&path, source).unwrap();
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

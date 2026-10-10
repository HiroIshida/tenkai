use super::cases::Case;
use rkern::cuda::{MathMode, Precision};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn check_arithmetic(cases: &[Case], precision: Precision) {
    let data = reference_data(cases, precision);
    let directory = test_directory(&format!("host-{precision:?}"));
    fs::write(directory.join("kernels.cuh"), kernel_source(cases)).unwrap();
    fs::write(directory.join("expected.h"), data).unwrap();
    fs::write(
        directory.join("main.cpp"),
        include_str!("../fixtures/cuda_host.cpp"),
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

pub(super) fn check_compilation(cases: &[Case], math: MathMode) {
    let directory = test_directory(&format!("nvcc-{math:?}"));
    let path = directory.join("kernels.cu");
    fs::write(&path, kernel_source(cases)).unwrap();
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let mut command = Command::new(nvcc);
    command.args(["--ptx", "--gpu-architecture=compute_75", "-std=c++11"]);
    if let Some(compiler) = std::env::var_os("CXX") {
        command.arg("-ccbin").arg(compiler);
    }
    run(command
        .arg(path)
        .arg("-o")
        .arg(directory.join("kernels.ptx")));
    if math == MathMode::F32Sfu {
        let ptx = fs::read_to_string(directory.join("kernels.ptx")).unwrap();
        for operation in ["sin", "cos"] {
            assert!(
                ptx.contains(&format!("{operation}.approx.f32"))
                    || ptx.contains(&format!("{operation}.approx.ftz.f32")),
                "missing SFU {operation} instruction",
            );
        }
    }
}

fn reference_data(cases: &[Case], precision: Precision) -> String {
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
    data
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

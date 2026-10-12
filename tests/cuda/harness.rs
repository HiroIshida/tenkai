use super::cases::Case;
use super::compile::{cuda_compiler, directory, run};
use rkern::cuda::{MathMode, Precision};
use std::fmt::Write;
use std::fs;
use std::process::Command;

pub(super) fn check_arithmetic(cases: &[Case], precision: Precision) {
    let data = reference_data(cases, precision);
    let directory = directory(&format!("cuda-host-{precision:?}"));
    fs::write(directory.join("kernels.cuh"), kernel_source(cases)).unwrap();
    fs::write(directory.join("expected.h"), data).unwrap();
    fs::write(
        directory.join("sdf_checks.h"),
        include_str!("../fixtures/sdf_checks.h"),
    )
    .unwrap();
    fs::write(
        directory.join("broad_phase_checks.h"),
        include_str!("../fixtures/broad_phase_checks.h"),
    )
    .unwrap();
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
    let directory = directory(&format!("cuda-nvcc-{math:?}"));
    let path = directory.join("kernels.cu");
    fs::write(&path, kernel_source(cases)).unwrap();
    let mut command = cuda_compiler("c++11");
    command.arg("--ptx");
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
    let namespace = match precision {
        Precision::F32 => "rkern_f32",
        Precision::F64 => "rkern_f64",
    };
    writeln!(data, "namespace sdf = {namespace};").unwrap();
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
        writeln!(
            data,
            "void invoke_{i}(const Scalar* inputs, void* output, size_t count) {{"
        )
        .unwrap();
        if let Some(setup) = &case.collision_setup {
            data.push_str(setup);
            writeln!(data, "kernel_{i}(inputs, environment_ids, scene, static_cast<unsigned char*>(output), count);").unwrap();
        } else {
            writeln!(
                data,
                "kernel_{i}(inputs, static_cast<Scalar*>(output), count);"
            )
            .unwrap();
        }
        data.push_str("}\n");
    }
    data.push_str("using Kernel = void (*)(const Scalar*, void*, size_t);\nstruct Case { const char* name; Kernel kernel; size_t components; size_t count; const Scalar* inputs; const double* expected; bool collision; };\nconst Case cases[] = {\n");
    for (i, case) in cases.iter().enumerate() {
        writeln!(
            data,
            "{{\"{}\", invoke_{i}, {}, {}, inputs_{i}, expected_{i}, {}}},",
            case.name,
            case.components,
            case.count,
            case.collision_setup.is_some()
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
        writeln!(source, "#define rkern_collision_one collision_one_{i}").unwrap();
        source.push_str(&case.source);
        source.push_str("#undef rkern_collision_one\n");
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

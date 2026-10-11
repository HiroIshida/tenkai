use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn directory(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("{name}-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

pub fn cuda_compiler(standard: &str) -> Command {
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let mut command = Command::new(nvcc);
    command.args(["-arch=compute_75", "-code=compute_75"]);
    command.arg(format!("-std={standard}"));
    if let Some(compiler) = std::env::var_os("CXX") {
        command.arg("-ccbin").arg(compiler);
    }
    command
}

pub fn run(command: &mut Command) {
    let output = command.output().expect("failed to execute test command");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

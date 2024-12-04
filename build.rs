use std::{
    env,
    io::{Error, ErrorKind},
    path::{Path, PathBuf},
    process::Command,
};

const CORE_HEADER: &str = "core/include/triton/core/tritonserver.h";
const CORE_BINDINGS_FILE_NAME: &str = "tritonserver.rs";

fn main() -> Result<(), Error> {
    let no_header_file = !Path::new(CORE_HEADER).is_file();
    if no_header_file {
        Command::new("git")
            .arg("submodule")
            .arg("update")
            .arg("--init")
            .arg("--recursive")
            .status()
            .and_then(|status| {
                if status.success() {
                    Ok(())
                } else {
                    Err(Error::new(
                        ErrorKind::Other,
                        format!("Git init return error: {}", status),
                    ))
                }
            })?;
    }

    let out_dir = Path::new(&env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap();

    println!("cargo:rerun-if-changed={}", CORE_HEADER);
    bindgen::builder()
        .header(CORE_HEADER)
        .clang_args(["-x", "c++"])
        .layout_tests(false)
        .dynamic_link_require_all(true)
        .generate()
        .map_err(|_| Error::new(ErrorKind::Other, "Bindgen generate error"))
        .and_then(|bindings| {
            bindings.write_to_file(
                [out_dir.as_path(), Path::new(CORE_BINDINGS_FILE_NAME)]
                    .iter()
                    .collect::<PathBuf>(),
            )
        })?;

    println!("cargo:rustc-link-lib=dylib=tritonserver");

    Ok(())
}

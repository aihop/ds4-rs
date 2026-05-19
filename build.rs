fn main() {
    println!("cargo:rerun-if-changed=sys/ds4_metal.m");
    println!("cargo:rerun-if-changed=sys/ds4_gpu.h");

    // Compile ds4_metal.m
    cc::Build::new()
        .file("sys/ds4_metal.m")
        .compiler("clang")
        .flag("-fobjc-arc")
        .flag("-O3")
        .flag("-ffast-math")
        .flag("-Wall")
        .flag("-Wextra")
        .compile("ds4_metal");

    // Link Metal and Foundation frameworks
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("sys/ds4_gpu.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
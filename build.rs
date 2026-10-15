fn main() {
    cc::Build::new()
        .file("vendor/harrier/harrier.c")
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c11")
        .flag("-ffreestanding")
        .flag("-fno-strict-aliasing")
        .flag("-ffp-contract=off")
        .compile("harrier");
    cc::Build::new()
        .file("vendor/harrier/harrier_tok.c")
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c11")
        .flag("-ffreestanding")
        .flag("-fno-strict-aliasing")
        .flag("-ffp-contract=off")
        .compile("harrier_tok");
    cc::Build::new()
        .file("vendor/harrier/harrier_tok_data.c")
        .flag("-std=c23") // #embed
        .opt_level(2)
        .compile("harrier_tok_data");
    cc::Build::new()
        .file("vendor/harrier/harrier_weights.c")
        .flag("-std=c23") // #embed — clang >= 19 / gcc >= 15
        .opt_level(2)
        .compile("harrier_weights");
}

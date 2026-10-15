use em0::harrier::Harrier;
use std::time::Instant;

fn main() {
    let h = Harrier::new().expect("init");

    // 10 sentences x ~1000 Chinese chars each (realistic memory-doc size)
    let base = "在软件开发的过程中，工程师需要不断权衡性能与可维护性。\
内存数据库通过向量化执行引擎显著提升了查询吞吐量，同时也带来了更高的内存占用。\
嵌入式系统则更加关注实时性与代码体积，开发者常常需要在有限的缓存中做极致优化。\
近年来，大语言模型的兴起改变了知识工作的形态，检索增强生成成为降低幻觉的关键手段。";
    let texts: Vec<String> = (0..10)
        .map(|i| format!("第{i}段。{}", base.repeat(7)))
        .collect();
    let texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let chars: usize = texts.iter().map(|t| t.chars().count()).sum();
    let toks: usize = texts.iter().map(|t| h.token_count(t)).sum();
    println!(
        "batch: {} texts, {} chars total, {} tokens total",
        texts.len(),
        chars,
        toks
    );

    let t0 = Instant::now();
    let seq = h.embed(&texts).expect("embed");
    let t_seq = t0.elapsed();
    println!(
        "sequential: {:>8.2} s  ({:.0} tok/s)",
        t_seq.as_secs_f64(),
        toks as f64 / t_seq.as_secs_f64()
    );

    let t0 = Instant::now();
    let par = h.embed_par(&texts).expect("embed_par");
    let t_par = t0.elapsed();
    println!(
        "parallel:   {:>8.2} s  ({:.0} tok/s)  speedup {:.1}x",
        t_par.as_secs_f64(),
        toks as f64 / t_par.as_secs_f64(),
        t_seq.as_secs_f64() / t_par.as_secs_f64()
    );

    // results must be identical
    let max_diff = seq
        .iter()
        .zip(par.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
    println!("seq vs par max diff: {max_diff:.9}");
}

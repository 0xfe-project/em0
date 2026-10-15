use em0::harrier::Harrier;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: rerank <query> <doc1> <doc2> ...");
        std::process::exit(2);
    }
    let h = Harrier::new().expect("init");
    let docs: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
    let scores = h.rerank(&args[1], &docs).expect("rerank");
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    for (rank, &i) in order.iter().enumerate() {
        println!("[{}] score={:.4} {}", rank + 1, scores[i], docs[i]);
    }
}

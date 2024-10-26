use std::fs::File;
use std::io::Write;
use std::time::Instant;

use ark_bls12_381::Fr;
use buvc_rs::poly::poly_fft;
use buvc_rs::poly::poly_ifft;
use buvc_rs::vc_context::VcContext;
use buvc_rs::vc_parameter::tests::test_parameter;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use criterion::SamplingMode;

fn benchmark_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Benchmark");
    let mut times: Vec<(usize, f64)> = Vec::new(); // For storing sizes and execution times

    // Set up Criterion to use fewer samples and handle long-running benchmarks
    group.sampling_mode(SamplingMode::Flat).sample_size(10); // Reduce sample size for long tests

    // Open a file for writing benchmarking data
    let mut file = File::create("fft_benchmark_results.csv").expect("Could not create file");
    writeln!(file, "n,execution_time").expect("Could not write to file");

    // Iterate over values of logn to get n from 2 to 100
    for logn in 1..=18 {
        // log2(2) = 1, log2(100) ≈ 6.64, so iterate through logn from 1 to 7
        let n = 1 << logn; // n = 2^logn, e.g., n = 2, 4, 8, 16, ...

        // Prepare test parameters using logn
        let (_s, vc_p) = test_parameter(logn); // Pass logn directly to test_parameter
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        // Ensure vectors are consistent with n
        let v: Vec<Fr> = (1..=n).map(|x| Fr::from(x as u64)).collect();
        let gq = vc_c.build_commitment(&v).1; // This line will now have consistent input

        let alpha_len = n / 2;
        let alpha: Vec<usize> = (0..alpha_len).collect();
        let beta: Vec<usize> = (alpha_len..n).collect();
        let delta_value: Vec<Fr> = (1..=alpha.len()).map(|x| Fr::from(x as u64)).collect();

        let start_time = Instant::now(); // Start timing
        let _updated_gq = vc_c.update_witnesses_batch(&alpha, &gq, &beta, &delta_value);
        let elapsed_time = start_time.elapsed().as_secs_f64(); // Time in seconds

        // Write benchmark data to the file
        writeln!(file, "{}, {:.6}", n, elapsed_time).expect("Could not write to file");

        times.push((n, elapsed_time)); // Store logn, n, and time

        // Benchmark the function
        group.bench_function(format!("fft n = {}", n), |b| {
            b.iter(|| {
                let mut gqq = gq.clone();
                poly_fft(&vc_c.unity, &mut gqq, n);
            });
        });

        group.bench_function(format!("ifft n = {}", n), |b| {
            b.iter(|| {
                let mut gqq = gq.clone();
                poly_ifft(&vc_c.unity, &mut gqq, n);
            });
        });
    }

    // Close the file after benchmarking
    drop(file);

    // Finish the benchmark group
    group.finish();
}

criterion_group!(benches, benchmark_fft);
criterion_main!(benches);

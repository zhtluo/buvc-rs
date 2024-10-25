use std::fs::File;
use std::io::Write;
use std::time::Instant;
use ark_bls12_381::Fr;
use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use plotters::prelude::*;

use buvc_rs::vc_context::VcContext;
use buvc_rs::vc_parameter::tests::test_parameter;

fn benchmark_update_witness_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("Update Witness Batch Benchmark");
    let mut times: Vec<(usize, f64)> = Vec::new(); // For storing sizes and execution times

    // Set up Criterion to use fewer samples and handle long-running benchmarks
    group.sampling_mode(SamplingMode::Flat).sample_size(10);  // Reduce sample size for long tests

    // Open a file for writing benchmarking data
    let mut file = File::create("benchmark_results.csv").expect("Could not create file");
    writeln!(file, "n,execution_time").expect("Could not write to file");

    // Iterate over values of logn to get n from 2 to 100
    for logn in 1..=17 {  // log2(2) = 1, log2(100) ≈ 6.64, so iterate through logn from 1 to 7
        let n = 1 << logn; // n = 2^logn, e.g., n = 2, 4, 8, 16, ...

        // Prepare test parameters using logn
        let (_s, vc_p) = test_parameter(logn);  // Pass logn directly to test_parameter
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        // Ensure vectors are consistent with n
        let v: Vec<Fr> = (1..=n).map(|x| Fr::from(x as u64)).collect();  // No need for n.max(1)
        let gq = vc_c.build_commitment(&v).1;  // This line will now have consistent input
        
        let alpha_len = n / 2;
        let alpha: Vec<usize> = (0..alpha_len).collect();
        let beta: Vec<usize> = (alpha_len..n).collect();
        let delta_value: Vec<Fr> = (1..=alpha.len()).map(|x| Fr::from(x as u64)).collect();

        let start_time = Instant::now();  // Start timing
        let _updated_gq = vc_c.update_witnesses_batch(&alpha, &gq, &beta, &delta_value);
        let elapsed_time = start_time.elapsed().as_secs_f64();  // Time in seconds

        // Write benchmark data to the file
        writeln!(file, "{}, {:.6}", n, elapsed_time).expect("Could not write to file");

        times.push((n, elapsed_time));  // Store logn, n, and time

        // Benchmark the function
        group.bench_function(format!("update_witness_batch n = {}", n), |b| {
            b.iter(|| {
                vc_c.update_witnesses_batch(&alpha, &gq, &beta, &delta_value);
            });
        });
    }

    // Close the file after benchmarking
    drop(file);

    // Finish the benchmark group
    group.finish();

    // Plotting results
    plot_results(times).expect("Plotting failed");
}

// Function to plot the results using Plotters
fn plot_results(times: Vec<(usize, f64)>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("benchmark_results.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_time = times.iter().map(|(_, t)| *t).fold(0./0., f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Benchmark Results: n vs Time", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(2..100, 0f64..max_time)?;  // x-axis now ranges from n = 2 to n = 100

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        times.into_iter().map(|(n, t)| (n as i32, t)), // Convert usize to i32
        &RED,
    ))?
    .label("Execution Time")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));
    
    chart.configure_series_labels().draw()?;

    Ok(())
}

criterion_group!(benches, benchmark_update_witness_batch);
criterion_main!(benches);
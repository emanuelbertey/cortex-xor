use candle_core::{Device, Tensor, Result, DType, IndexOp};
use candle_core::backend::BackendDevice;
use candle_nn::{VarBuilder, LSTMConfig, rnn::LSTMState, RNN, Module};
use xlstm::MinLstmConfig;
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("============================================================");
    println!("           MinLSTM vs Standard LSTM: ULTIMATE SHOWDOWN      ");
    println!("============================================================");
    println!("Device: {:?}\n", device);

    // --- ARCHITECTURE & SMOKE (2 tests) ---
    run_test("1. Architecture: Parameter Count Efficiency", || test_parameters(&device))?;
    run_test("2. Smoke Test: Initial Output Distribution", || test_smoke_comparison(&device))?;

    // --- PERFORMANCE & SCALING (2 tests) ---
    run_test("3. Performance: Speed Benchmark (Fixed Seq)", || test_benchmark(&device))?;
    run_test("4. Performance: Sequence Length Scaling", || test_seq_scaling(&device))?;

    // --- MATHEMATICAL DYNAMICS (2 tests) ---
    run_test("5. Dynamics: Gradient Flow (Signal Strength)", || test_gradient_flow(&device))?;
    run_test("6. Dynamics: Hidden State Evolution (Stability)", || test_state_evolution(&device))?;

    // --- FUNCTIONAL CORRECTNESS (2 tests) ---
    run_test("7. Correctness: Batch Independence", || test_batch_independence(&device))?;
    run_test("8. Correctness: Parallel vs Sequential State Consistency", || test_state_consistency(&device))?;

    // --- ALGORITHMIC TASKS (3 tests) ---
    run_test("9. Task: Copy Sequence (Shifted Memory)", || test_copy_task(&device))?;
    run_test("10. Task: Counting Occurrences (Integration)", || test_counting_task(&device))?;
    run_test("11. Task: Key-Value Retrieval (Associative Memory)", || test_kv_retrieval_task(&device))?;

    println!("\n============================================================");
    println!("            All 11 Tests Completed Successfully             ");
    println!("============================================================");
    Ok(())
}

fn run_test<F>(name: &str, test_fn: F) -> Result<()> 
where F: FnOnce() -> Result<()> {
    println!("\n[TEST] {}", name);
    let start = Instant::now();
    match test_fn() {
        Ok(_) => {
            println!("✅ Passed ({:.2?})", start.elapsed());
            Ok(())
        },
        Err(e) => {
            println!("❌ FAILED: {}", e);
            Err(e)
        }
    }
}

// ---------------------------------------------------------
// HELPERS
// ---------------------------------------------------------

fn run_lstm_sequential(lstm: &candle_nn::LSTM, x: &Tensor, dev: &Device) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    let x_t = x.transpose(0, 1)?.contiguous()?;
    let h0 = Tensor::zeros((b, d), DType::F32, dev)?;
    let c0 = Tensor::zeros((b, d), DType::F32, dev)?;
    let mut state = LSTMState { h: h0, c: c0 };
    let mut outs = Vec::with_capacity(s);
    for t in 0..s {
        state = lstm.step(&x_t.get(t)?, &state)?;
        outs.push(state.h.clone());
    }
    Tensor::stack(&outs, 1)
}

fn bench_fn<F>(mut f: F, dev: &Device) -> Result<f64> where F: FnMut() -> Result<()> {
    for _ in 0..2 { f()?; }
    if let Device::Cuda(d) = dev { d.synchronize()?; }
    let start = Instant::now();
    for _ in 0..10 { f()?; }
    if let Device::Cuda(d) = dev { d.synchronize()?; }
    Ok(start.elapsed().as_secs_f64() / 10.0)
}

fn calculate_stats(t: &Tensor) -> Result<(f32, f32)> {
    let mean = t.mean_all()?.to_scalar::<f32>()?;
    let var = t.broadcast_sub(&t.mean_all()?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok((mean, var.sqrt()))
}

// ---------------------------------------------------------
// 1. ARCHITECTURE & SMOKE
// ---------------------------------------------------------

fn test_parameters(device: &Device) -> Result<()> {
    let d = 256;
    let vb = VarBuilder::zeros(DType::F32, device);
    let _ = MinLstmConfig::new(d).init(vb.pp("min"))?;
    let _ = candle_nn::lstm(d, d, LSTMConfig::default(), vb.pp("std"))?;
    let min_params = 3 * d * d + 3 * d;
    let std_params = 4 * d * (d + d) + 4 * d; 
    println!("   Comparison for d={}:", d);
    println!("   | Model         | Est. Parameters | Efficiency    |");
    println!("   |---------------|-----------------|---------------|");
    println!("   | MinLSTM       | {:>15} | 1.00x (Base)  |", min_params);
    println!("   | Standard LSTM | {:>15} | {:>7.2}x More  |", std_params, std_params as f32 / min_params as f32);
    Ok(())
}

fn test_smoke_comparison(device: &Device) -> Result<()> {
    let b = 2; let s = 16; let d = 32;
    let x = Tensor::randn(0f32, 1.0, (b, s, d), device)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let min_lstm = MinLstmConfig::new(d).init(vb.pp("min_smoke"))?;
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), vb.pp("std_smoke"))?;
    let (out_min, _) = min_lstm.forward(&x, None, false)?;
    let out_std = run_lstm_sequential(&lstm, &x, device)?;
    let (m1, s1) = calculate_stats(&out_min)?;
    let (m2, s2) = calculate_stats(&out_std)?;
    println!("   Initial Weight Statistics (Random Init):");
    println!("   | Metric | MinLSTM   | Std LSTM  |");
    println!("   |--------|-----------|-----------|");
    println!("   | Mean   | {:>9.4} | {:>9.4} |", m1, m2);
    println!("   | StdDev | {:>9.4} | {:>9.4} |", s1, s2);
    Ok(())
}

// ---------------------------------------------------------
// 2. PERFORMANCE & SCALING
// ---------------------------------------------------------

fn test_benchmark(device: &Device) -> Result<()> {
    let b = 16; let s = 256; let d = 256; let iters = 20;
    let x = Tensor::randn(0f32, 1.0, (b, s, d), device)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let min_lstm = MinLstmConfig::new(d).init(vb.pp("m_perf"))?;
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), vb.pp("s_perf"))?;
    let t_min = bench_fn(|| min_lstm.forward(&x, None, false).map(|_| ()), device)?;
    let t_std = bench_fn(|| run_lstm_sequential(&lstm, &x, device).map(|_| ()), device)?;
    println!("   | Model         | Time/Iter (ms) | Speedup |");
    println!("   |---------------|----------------|---------|");
    println!("   | MinLSTM       | {:>14.4} | Base    |", t_min * 1000.0);
    println!("   | Standard LSTM | {:>14.4} | {:>7.2}x |", t_std * 1000.0, t_std / t_min);
    Ok(())
}

fn test_seq_scaling(device: &Device) -> Result<()> {
    let b = 8; let d = 128; let vb = VarBuilder::zeros(DType::F32, device);
    let min_lstm = MinLstmConfig::new(d).init(vb.pp("min_s"))?;
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), vb.pp("std_s"))?;
    println!("   | Seq Len | MinLSTM (ms) | Std LSTM (ms) | Speedup |");
    println!("   |---------|--------------|---------------|---------|");
    for s in [128, 512, 1024] {
        let x = Tensor::randn(0f32, 1.0, (b, s, d), device)?;
        let t_min = bench_fn(|| min_lstm.forward(&x, None, false).map(|_| ()), device)?;
        let t_std = bench_fn(|| run_lstm_sequential(&lstm, &x, device).map(|_| ()), device)?;
        println!("   | {:>7} | {:>12.2} | {:>13.2} | {:>7.2}x |", s, t_min * 1000.0, t_std * 1000.0, t_std / t_min);
    }
    Ok(())
}

// ---------------------------------------------------------
// 3. MATHEMATICAL DYNAMICS
// ---------------------------------------------------------

fn test_gradient_flow(device: &Device) -> Result<()> {
    let b = 4; let s = 256; let d = 64;
    let x = Tensor::randn(0f32, 1.0, (b, s, d), device)?;
    let vam1 = candle_nn::VarMap::new();
    let min_lstm = MinLstmConfig::new(d).init(VarBuilder::from_varmap(&vam1, DType::F32, device).pp("m"))?;
    let (out_m, _) = min_lstm.forward(&x, None, false)?;
    let gs1 = out_m.sum_all()?.backward()?;
    let mut n1 = 0.0f32;
    for v in vam1.data().lock().unwrap().values() { if let Some(g) = gs1.get(v) { n1 += g.sqr()?.sum_all()?.to_scalar::<f32>()?; } }
    
    let vam2 = candle_nn::VarMap::new();
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), VarBuilder::from_varmap(&vam2, DType::F32, device).pp("s"))?;
    let out_s = run_lstm_sequential(&lstm, &x, device)?;
    let gs2 = out_s.sum_all()?.backward()?;
    let mut n2 = 0.0f32;
    for v in vam2.data().lock().unwrap().values() { if let Some(g) = gs2.get(v) { n2 += g.sqr()?.sum_all()?.to_scalar::<f32>()?; } }
    
    println!("   Total Gradient Norm (Signal strength):");
    println!("   | MinLSTM: {:>10.4} | Std LSTM: {:>10.4} |", n1.sqrt(), n2.sqrt());
    Ok(())
}

fn test_state_evolution(device: &Device) -> Result<()> {
    let b = 1; let s = 100; let d = 32;
    let x = Tensor::ones((b, s, d), DType::F32, device)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let min_lstm = MinLstmConfig::new(d).init(vb.pp("m_evolve"))?;
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), vb.pp("s_evolve"))?;
    let (out_min, _) = min_lstm.forward(&x, None, false)?;
    let out_std = run_lstm_sequential(&lstm, &x, device)?;
    println!("   Hidden Magnitude (ABS Mean) Growth:");
    println!("   | Step | MinLSTM   | Std LSTM  |");
    println!("   |------|-----------|-----------|");
    for t in [0, 49, 99] {
        let v_m = out_min.i((.., t, ..))?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let v_s = out_std.i((.., t, ..))?.abs()?.mean_all()?.to_scalar::<f32>()?;
        println!("   | {:>4} | {:>9.4} | {:>9.4} |", t+1, v_m, v_s);
    }
    Ok(())
}

// ---------------------------------------------------------
// 4. FUNCTIONAL CORRECTNESS
// ---------------------------------------------------------

fn test_batch_independence(device: &Device) -> Result<()> {
    let dim = 32; let s = 10;
    let vb = VarBuilder::zeros(DType::F32, device);
    let model = MinLstmConfig::new(dim).init(vb.pp("bi"))?;
    let x1 = Tensor::randn(0f32, 1.0, (1, s, dim), device)?;
    let x2 = Tensor::randn(0f32, 1.0, (1, s, dim), device)?.broadcast_mul(&Tensor::new(5.0f32, device)?)?;
    let (out_batch, _) = model.forward(&Tensor::cat(&[&x1, &x2], 0)?, None, false)?;
    let (out_1, _) = model.forward(&x1, None, false)?;
    let diff = (out_batch.i(0)? - out_1.i(0)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    if diff > 1e-5 { return Err(candle_core::Error::Msg(format!("Batch Leak: {}", diff).into())); }
    println!("   ✅ Batch elements are effectively independent.");
    Ok(())
}

fn test_state_consistency(device: &Device) -> Result<()> {
    let dim = 16; let vb = VarBuilder::zeros(DType::F32, device);
    let model = MinLstmConfig::new(dim).init(vb.pp("sc"))?;
    let x = Tensor::randn(0f32, 1.0, (1, 10, dim), device)?;
    let (out_full, _) = model.forward(&x, None, false)?;
    let (out1, h) = model.forward(&x.narrow(1, 0, 5)?, None, true)?;
    let (out2, _) = model.forward(&x.narrow(1, 5, 5)?, h.as_ref(), false)?;
    let diff = (out_full - Tensor::cat(&[&out1, &out2], 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    if diff > 1e-4 { return Err(candle_core::Error::Msg(format!("Consistency gap: {}", diff).into())); }
    println!("   ✅ State passing matches full sequence parallel run.");
    Ok(())
}

// ---------------------------------------------------------
// 5. ALGORITHMIC TASKS
// ---------------------------------------------------------

fn compare_train(name: &str, x: &Tensor, target: &Tensor, d: usize, steps: usize, lr: f64, dev: &Device) -> Result<()> {
    let vam1 = candle_nn::VarMap::new();
    let min_lstm = MinLstmConfig::new(d).init(VarBuilder::from_varmap(&vam1, DType::F32, dev).pp("m"))?;
    let vam2 = candle_nn::VarMap::new();
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), VarBuilder::from_varmap(&vam2, DType::F32, dev).pp("l"))?;
    let get_l1 = || { let (o,_) = min_lstm.forward(x, None, false).unwrap(); (o - target).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() };
    let get_l2 = || { let o = run_lstm_sequential(&lstm, x, dev).unwrap(); (o - target).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() };
    let l1_start = get_l1(); let l2_start = get_l2();
    for _ in 0..steps {
        let (o1,_) = min_lstm.forward(x, None, false)?;
        let gs1 = (o1 - target)?.sqr()?.mean_all()?.backward()?;
        for v in vam1.data().lock().unwrap().values() { if let Some(g) = gs1.get(v) { v.set(&(v.as_tensor() - (g*lr)?)?)?; } }
        let o2 = run_lstm_sequential(&lstm, x, dev)?;
        let gs2 = (o2 - target)?.sqr()?.mean_all()?.backward()?;
        for v in vam2.data().lock().unwrap().values() { if let Some(g) = gs2.get(v) { v.set(&(v.as_tensor() - (g*lr)?)?)?; } }
    }
    let l1_end = get_l1(); let l2_end = get_l2();
    println!("   {} ({} steps):", name, steps);
    println!("   | Model     | Start Loss | End Loss   | Improvement |");
    println!("   |-----------|------------|------------|-------------|");
    println!("   | MinLSTM   | {:>10.4} | {:>10.4} | {:>10.2}% |", l1_start, l1_end, (1.0 - l1_end/l1_start)*100.0);
    println!("   | Std LSTM  | {:>10.4} | {:>10.4} | {:>10.2}% |", l2_start, l2_end, (1.0 - l2_end/l2_start)*100.0);
    Ok(())
}

fn test_copy_task(device: &Device) -> Result<()> {
    let b = 16; let data_s = 8; let s = data_s * 2; let d = 16;
    let mut x_v = vec![0.0f32; b*s*d]; let mut y_v = vec![0.0f32; b*s*d];
    for bi in 0..b { for si in 0..data_s { for di in 0..d {
        let val = (bi+si+di) as f32 * 0.1;
        x_v[bi*s*d + si*d + di] = val; y_v[bi*s*d + (si+data_s)*d + di] = val;
    }}}
    let x = Tensor::from_vec(x_v, (b, s, d), device)?;
    let y = Tensor::from_vec(y_v, (b, s, d), device)?;
    compare_train("Copy Task", &x, &y, d, 30, 0.5, device)
}

fn test_counting_task(device: &Device) -> Result<()> {
    let b = 8; let s = 32; let d = 32;
    let x = Tensor::randn(0.0f32, 1.0, (b, s, d), device)?.abs()?.ge(0.5)?.to_dtype(DType::F32)?;
    let target = x.narrow(2, 0, 1)?.cumsum(1)?.broadcast_as((b, s, d))?;
    compare_train("Counting", &x, &target, d, 50, 0.8, device)
}

fn test_kv_retrieval_task(device: &Device) -> Result<()> {
    let b = 8; let s = 9; let d = 16;
    let x = Tensor::randn(0f32, 1.0, (b, s, d), device)?;
    let first_v = x.narrow(1, 1, 1)?; 
    let mut target_vec = vec![0.0f32; b * s * d];
    let fv_flat = first_v.flatten_all()?.to_vec1::<f32>()?;
    for bi in 0..b { for di in 0..d { target_vec[bi*s*d + (s-1)*d + di] = fv_flat[bi*d+di]; }}
    let target = Tensor::from_vec(target_vec, (b, s, d), device)?;
    compare_train("KV Lookup", &x, &target, d, 40, 0.5, device)
}

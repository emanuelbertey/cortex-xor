use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, LSTMConfig};
use xlstm::{MinLstmConfig, MinLstmThreaded, MinLstm};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("============================================================");
    println!("    MinLSTM THREADED vs SINGLE: LEARNING + SPEED            ");
    println!("============================================================");
    println!("Device: {:?}\n", device);

    let b = 16; 
    let s = 128;
    let d = 64;
    let threads = 4; // Configuración de hilos explícita
    
    // 1. GENERAR DATOS PARA TAREA DE APRENDIZAJE (Counting Task)
    let x = Tensor::randn(0.0f32, 1.0, (b, s, d), &device)?.abs()?.ge(0.5)?.to_dtype(DType::F32)?;
    let target = x.narrow(2, 0, 1)?.cumsum(1)?.broadcast_as((b, s, d))?;

    // 2. INICIALIZAR MODELOS
    let config = MinLstmConfig::new(d);
    
    let vam_s = candle_nn::VarMap::new();
    let vb_s = VarBuilder::from_varmap(&vam_s, DType::F32, &device);
    let min_single = config.init(vb_s.pp("m"))?;

    let vam_t = candle_nn::VarMap::new();
    let vb_t = VarBuilder::from_varmap(&vam_t, DType::F32, &device);
    let min_for_thread = config.init(vb_t.pp("m"))?; 
    
    // Inicializamos con 4 hilos como pediste
    let min_threaded = MinLstmThreaded::new(min_for_thread, config.clone(), threads);
    
    // Standard LSTM para referencia de velocidad
    let vam_l = candle_nn::VarMap::new();
    let vb_l = VarBuilder::from_varmap(&vam_l, DType::F32, &device);
    let lstm = candle_nn::lstm(d, d, LSTMConfig::default(), vb_l.pp("s"))?;

    println!("--- TEST A: SPEED IMPROVEMENT (Using {} Threads) ---", min_threaded.num_threads());
    let t_min = bench_fn(|| min_single.forward(&x, None, false).map(|_| ()), &device)?;
    let t_thread = bench_fn(|| min_threaded.forward(&x, None, false).map(|_| ()), &device)?;
    let t_std = bench_fn(|| run_lstm_sequential(&lstm, &x, &device).map(|_| ()), &device)?;
    
    println!("| Model              | Time/Iter (ms) | Speedup vs Single |");
    println!("|--------------------|----------------|-------------------|");
    println!("| MinLSTM (Single)   | {:>14.4} | 1.00x (Base)      |", t_min * 1000.0);
    println!("| MinLSTM ({} Th)  | {:>14.4} | {:>17.2}x |", min_threaded.num_threads(), t_thread * 1000.0, t_min / t_thread);
    println!("| Standard LSTM      | {:>14.4} | {:>17.2}x |", t_std * 1000.0, t_min / t_std);

    println!("\n--- TEST B: LEARNING COMPARISON ({} Steps, {} Threads) ---", 30, min_threaded.num_threads());
    let steps = 30;
    let lr = 0.8;

    println!("Entrenando MinLSTM (Single)...");
    let (l_s_start, l_s_end) = train_model(&min_single, &vam_s, &x, &target, steps, lr)?;
    
    println!("Entrenando MinLSTM (Threaded - {} hilos)...", min_threaded.num_threads());
    let (l_t_start, l_t_end) = train_threaded(&min_threaded, &vam_t, &x, &target, steps, lr)?;

    println!("\nLearning Results Comparison:");
    println!("| Model     | Start Loss | End Loss   | Improvement | Status |");
    println!("|-----------|------------|------------|-------------|--------|");
    println!("| Single    | {:>10.4} | {:>10.4} | {:>10.2}% | {} |", 
        l_s_start, l_s_end, (1.0 - l_s_end/l_s_start)*100.0, if l_s_end < l_s_start { "✅" } else { "❌" });
    println!("| Threaded  | {:>10.4} | {:>10.4} | {:>10.2}% | {} |", 
        l_t_start, l_t_end, (1.0 - l_t_end/l_t_start)*100.0, if l_t_end < l_t_start { "✅" } else { "❌" });

    // 3. VERIFICAR PARIDAD MATEMÁTICA FINAL
    // Reseteamos los pesos para que sean iguales y comparar una sola pasada
    {
        let data_s = vam_s.data().lock().unwrap();
        let data_t = vam_t.data().lock().unwrap();
        for (name, var_s) in data_s.iter() {
            if let Some(var_t) = data_t.get(name) {
                var_t.set(var_s.as_tensor())?;
            }
        }
    }
    
    let (out_s, _) = min_single.forward(&x, None, false)?;
    let (out_t, _) = min_threaded.forward(&x, None, false)?;
    let diff = (out_s.sub(&out_t)?).abs()?.max_all()?.to_scalar::<f32>()?;
    
    println!("\n--- TEST C: MATHEMATICAL PARITY ---");
    println!("Diferencia Máxima final: {:.8}", diff);
    if diff < 1e-5 {
        println!("✅ Los hilos ({}) mantienen paridad matemática absoluta.", min_threaded.num_threads());
    }

    Ok(())
}

fn train_model(m: &MinLstm, vm: &candle_nn::VarMap, x: &Tensor, t: &Tensor, steps: usize, lr: f32) -> Result<(f32, f32)> {
    let (start_out, _) = m.forward(x, None, false)?;
    let start_loss = (start_out.sub(t)?).sqr()?.mean_all()?.to_scalar::<f32>()?;
    for _ in 0..steps {
        let (out, _) = m.forward(x, None, false)?;
        let loss = (out.sub(t)?).sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for v in vm.data().lock().unwrap().values() {
            if let Some(g) = grads.get(v) {
                v.set(&(v.as_tensor() - (g * lr as f64)?)?)?;
            }
        }
    }
    let (end_out, _) = m.forward(x, None, false)?;
    let end_loss = (end_out.sub(t)?).sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok((start_loss, end_loss))
}

fn train_threaded(m: &MinLstmThreaded, vm: &candle_nn::VarMap, x: &Tensor, t: &Tensor, steps: usize, lr: f32) -> Result<(f32, f32)> {
    let (start_out, _) = m.forward(x, None, false)?;
    let start_loss = (start_out.sub(t)?).sqr()?.mean_all()?.to_scalar::<f32>()?;
    for _ in 0..steps {
        let (out, _) = m.forward(x, None, false)?;
        let loss = (out.sub(t)?).sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for v in vm.data().lock().unwrap().values() {
            if let Some(g) = grads.get(v) {
                v.set(&(v.as_tensor() - (g * lr as f64)?)?)?;
            }
        }
    }
    let (end_out, _) = m.forward(x, None, false)?;
    let end_loss = (end_out.sub(t)?).sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok((start_loss, end_loss))
}

fn bench_fn<F>(mut f: F, _dev: &Device) -> Result<f64> where F: FnMut() -> Result<()> {
    for _ in 0..2 { f()?; }
    let start = Instant::now();
    let iters = 5;
    for _ in 0..iters { f()?; }
    Ok(start.elapsed().as_secs_f64() / iters as f64)
}

fn run_lstm_sequential(lstm: &candle_nn::LSTM, x: &Tensor, dev: &Device) -> Result<Tensor> {
    use candle_nn::rnn::{LSTMState, RNN};
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

use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder};
use xlstm::{MinLstmConfig, MinLstmThreaded};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("============================================================");
    println!("    MinLSTM THREADED: SPEED + LEARNING CORRECTNESS          ");
    println!("============================================================");
    println!("Device: {:?}\n", device);

    let b = 16; 
    let s = 128;
    let d = 64;
    
    // 1. GENERAR DATOS PARA TAREA DE APRENDIZAJE (Counting Task)
    let x = Tensor::randn(0.0f32, 1.0, (b, s, d), &device)?.abs()?.ge(0.5)?.to_dtype(DType::F32)?;
    let target = x.narrow(2, 0, 1)?.cumsum(1)?.broadcast_as((b, s, d))?;

    // 2. INICIALIZAR MODELOS
    let vam = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vam, DType::F32, &device);
    let config = MinLstmConfig::new(d);
    
    // Inicializamos dos instancias que apuntan a la MISMA memoria en la VarMap
    let min_single = config.init(vb.pp("m"))?;
    let min_for_thread = config.init(vb.pp("m"))?; 
    
    let min_threaded = MinLstmThreaded::new(min_for_thread, config.clone());

    println!("--- TEST A: SPEED IMPROVEMENT ---");
    let t_min = bench_fn(|| min_single.forward(&x, None, false).map(|_| ()), &device)?;
    let t_thread = bench_fn(|| min_threaded.forward(&x, None, false).map(|_| ()), &device)?;
    
    println!("| Model              | Time/Iter (ms) | Speedup |");
    println!("|--------------------|----------------|---------|");
    println!("| MinLSTM (Single)   | {:>14.4} | Base    |", t_min * 1000.0);
    println!("| MinLSTM (Threaded) | {:>14.4} | {:>7.2}x |", t_thread * 1000.0, t_min / t_thread);

    println!("\n--- TEST B: LEARNING VERIFICATION (30 Steps) ---");
    println!("Verificando que la versión Threaded puede optimizar sus pesos correctamente...");
    
    let lr = 0.8;
    let l_start = get_loss(&min_threaded, &x, &target)?;
    
    for step in 1..=30 {
        let (out, _) = min_threaded.forward(&x, None, false)?;
        let loss = (out - &target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        
        // Actualizar pesos (Manual SGD)
        for v in vam.data().lock().unwrap().values() {
            if let Some(g) = grads.get(v) {
                v.set(&(v.as_tensor() - (g * lr)?)?)?;
            }
        }
        if step % 10 == 0 {
            println!("   Step {:>2} | Loss: {:.6}", step, loss.to_scalar::<f32>()?);
        }
    }
    
    let l_end = get_loss(&min_threaded, &x, &target)?;
    let improvement = (1.0 - l_end / l_start) * 100.0;

    println!("\nLearning Results (Threaded):");
    println!("| Start Loss | End Loss   | Improvement | Status      |");
    println!("|------------|------------|-------------|-------------|");
    println!("| {:>10.4} | {:>10.4} | {:>10.2}% | {} |", 
        l_start, l_end, improvement, 
        if improvement > 50.0 { "✅ LEARNS   " } else { "❌ NO LEARN  " });

    // 3. VERIFICAR PARIDAD (Single vs Threaded deben dar el mismo resultado exacto)
    let (out_s, _) = min_single.forward(&x, None, false)?;
    let (out_t, _) = min_threaded.forward(&x, None, false)?;
    let diff = (out_s - out_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
    
    println!("\n--- TEST C: MATHEMATICAL PARITY ---");
    println!("Diferencia Máxima (Single vs Threaded): {:.8}", diff);
    if diff < 1e-5 {
        println!("✅ Los hilos procesan los datos con precisión matemática perfecta.");
    } else {
        println!("❌ ERROR: Hay una discrepancia en el cálculo de los hilos!");
    }

    Ok(())
}

fn get_loss(m: &MinLstmThreaded, x: &Tensor, t: &Tensor) -> Result<f32> {
    let (out, _) = m.forward(x, None, false)?;
    (out - t)?.sqr()?.mean_all()?.to_scalar::<f32>()
}

fn bench_fn<F>(mut f: F, _dev: &Device) -> Result<f64> where F: FnMut() -> Result<()> {
    for _ in 0..2 { f()?; }
    let start = Instant::now();
    let iters = 5;
    for _ in 0..iters { f()?; }
    Ok(start.elapsed().as_secs_f64() / iters as f64)
}
